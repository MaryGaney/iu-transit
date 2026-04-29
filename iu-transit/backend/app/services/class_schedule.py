"""
app/services/class_schedule.py
───────────────────────────────
Parses the IU class schedule CSV and computes the "students releasing soon"
feature for each bus stop.

Two responsibilities:
  1. parse_schedule_csv(path)  — load the 8,337-row CSV into ClassSection rows
  2. compute_release_events()  — for every stop × day-of-week × 15-min window,
                                 count how many students are releasing nearby

The student-release computation is the key novel signal in this system.
It's done once per hour and cached in StudentReleaseEvent. The LSTM feature
builder queries that table — not the raw class data — during inference.

Column mapping from the provided CSV:
  "Instruction Mode"         → instruction_mode
  "Building & Room"          → raw string like "BLTH A217", parsed to code+room
  "Stu Enrl"                 → enrollment
  "Start & End Dt"           → start_date, end_date  e.g. "AUG-25 - DEC-19"
  "Start & End Time"         → start_time, end_time  e.g. "04:20PM - 07:20PM"
  "Cls Mtg Ptrn"             → meeting_days  e.g. "MWA" = Mon/Wed/unknown...
  "Class Nbr"                → class_nbr
  "Course"                   → course_id
  "Component"                → component
"""

import csv
import math
import io
import re
from datetime import datetime, date, time, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select, text

from app.core.config import settings
from app.core.logging import logger
from app.models.schedule import ClassSection, IUBuilding, StudentReleaseEvent
from app.models.gtfs import Stop


# ── Day-of-week parsing ───────────────────────────────────────────────────────
# IU uses letter codes: M=Mon, T=Tue, W=Wed, R=Thu, F=Fri, S=Sat, U=Sun
# "A" appears in some patterns (possibly "all days") — we treat conservatively
DAY_LETTER_TO_DOW = {
    "M": 0, "T": 1, "W": 2, "R": 3, "F": 4, "S": 5, "U": 6,
}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_days(day_str: str) -> list[int]:
    """Convert meeting day string like 'MWF' or 'TR' to list of DOW ints."""
    if not day_str:
        return []
    days = []
    for ch in day_str.upper():
        if ch in DAY_LETTER_TO_DOW:
            days.append(DAY_LETTER_TO_DOW[ch])
    return list(set(days))


def _parse_date(date_str: str) -> Optional[date]:
    """Parse 'AUG-25' → date(current_year, 8, 25). Assumes current academic year."""
    if not date_str:
        return None
    m = re.match(r"([A-Z]{3})-(\d{1,2})", date_str.strip().upper())
    if not m:
        return None
    month = MONTH_MAP.get(m.group(1))
    day = int(m.group(2))
    if not month:
        return None
    # Determine year: AUG-DEC = current year fall, JAN-MAY = next year spring
    now = datetime.now()
    year = now.year if month >= 8 else now.year + 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_time(time_str: str) -> Optional[time]:
    """Parse '04:20PM' → time(16, 20)."""
    if not time_str:
        return None
    time_str = time_str.strip().upper()
    m = re.match(r"(\d{1,2}):(\d{2})(AM|PM)", time_str)
    if not m:
        return None
    hour, minute, period = int(m.group(1)), int(m.group(2)), m.group(3)
    if period == "PM" and hour != 12:
        hour += 12
    elif period == "AM" and hour == 12:
        hour = 0
    try:
        return time(hour, minute)
    except ValueError:
        return None


def _parse_building_room(br_str: str) -> tuple[str, str]:
    """
    Parse 'BLTH A217' → ('BLTH', 'A217').
    Handles: 'BLTH A217', 'BLHD TBA', 'BLTH A111', etc.
    Returns ('', '') if unparseable.
    """
    if not br_str:
        return "", ""
    parts = br_str.strip().split(None, 1)
    if len(parts) == 2:
        return parts[0].upper(), parts[1]
    elif len(parts) == 1:
        return parts[0].upper(), ""
    return "", ""


def _parse_date_range(range_str: str) -> tuple[Optional[date], Optional[date]]:
    """Parse 'AUG-25 - DEC-19' → (date, date)."""
    if not range_str:
        return None, None
    parts = range_str.split(" - ")
    if len(parts) == 2:
        return _parse_date(parts[0].strip()), _parse_date(parts[1].strip())
    return None, None


def _parse_time_range(range_str: str) -> tuple[Optional[time], Optional[time]]:
    """Parse '04:20PM - 07:20PM' → (time, time)."""
    if not range_str:
        return None, None
    parts = range_str.split(" - ")
    if len(parts) == 2:
        return _parse_time(parts[0].strip()), _parse_time(parts[1].strip())
    return None, None


# ── CSV parser ────────────────────────────────────────────────────────────────

async def parse_schedule_csv(db: AsyncSession, csv_path: str) -> int:
    """
    Parse the IU schedule CSV and insert into class_sections table.
    Clears existing data first (idempotent reload).

    The CSV columns are pipe-delimited based on the provided sample; we also
    try comma and tab. The parser auto-detects.
    """
    logger.info(f"Parsing IU schedule from {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|")
            reader = csv.DictReader(f, dialect=dialect)

            await db.execute(delete(ClassSection))
            await db.flush()

            count = 0
            batch = []

            for row in reader:
                # Normalise keys (strip whitespace, handle multi-line headers)
                row = {k.strip(): v.strip() if v else "" for k, v in row.items()}

                instruction_mode = row.get("Instruction Mode", row.get("Instruction\nMode", ""))
                is_in_person = "in person" in instruction_mode.lower()

                # Skip fully online classes — they don't affect bus stops
                if not is_in_person:
                    continue

                # Enrollment
                enrl_str = row.get("Stu Enrl", row.get("StuEnrl", "0")).replace(",", "")
                try:
                    enrollment = int(enrl_str) if enrl_str.isdigit() else 0
                except ValueError:
                    enrollment = 0

                # Date/time
                date_range_str = row.get("Start & End Dt", row.get("Start &\nEnd Dt", ""))
                time_range_str = row.get("Start & End Time", row.get("Start &\nEnd Time", ""))
                start_date, end_date = _parse_date_range(date_range_str)
                start_time, end_time = _parse_time_range(time_range_str)

                # Building
                br_str = row.get("Building & Room", row.get("Building &\nRoom", ""))
                building_code, room = _parse_building_room(br_str)

                # Skip rows without a time (independent study, online labs, etc.)
                if not end_time:
                    continue

                # Total minutes
                total_min_str = row.get("Total Min", "0").replace(",", "")
                try:
                    total_minutes = int(total_min_str) if total_min_str.isdigit() else None
                except ValueError:
                    total_minutes = None

                # Meeting days
                meeting_days = row.get("Cls Mtg Ptrn", row.get("ClsMtgPtrn", ""))

                # Class number
                class_nbr_str = row.get("Class Nbr", row.get("ClassNbr", "0"))
                try:
                    class_nbr = int(class_nbr_str) if class_nbr_str.isdigit() else 0
                except ValueError:
                    class_nbr = 0

                section = ClassSection(
                    class_nbr=class_nbr,
                    course_id=row.get("Course", ""),
                    component=row.get("Component", ""),
                    instruction_mode=instruction_mode,
                    is_in_person=True,
                    meeting_days=meeting_days,
                    start_date=start_date,
                    end_date=end_date,
                    start_time=start_time,
                    end_time=end_time,
                    total_minutes=total_minutes,
                    building_code=building_code,
                    room=room,
                    enrollment=enrollment,
                    room_capacity=_safe_int(row.get("Room Cap", "")),
                    e_cap=_safe_int(row.get("E-Cap", "")),
                    academic_term=row.get("Academic Term Code", ""),
                    dept_name=row.get("Financial Organization Hierarchy Level Two Description Course", ""),
                )
                batch.append(section)

                if len(batch) >= 500:
                    db.add_all(batch)
                    await db.flush()
                    count += len(batch)
                    batch = []

            if batch:
                db.add_all(batch)
                await db.flush()
                count += len(batch)

            logger.info(f"Loaded {count} in-person class sections with times")
            return count

    except FileNotFoundError:
        logger.error(f"Schedule CSV not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Schedule CSV parse error: {e}")
        raise


# ── Student release event computation ────────────────────────────────────────

async def compute_release_events(db: AsyncSession) -> int:
    """
    For every bus stop, compute how many students will be releasing
    in each 15-minute window of each day of the week.

    Algorithm:
      For each in-person class section:
        1. Get the building's lat/lng from IUBuilding
        2. Find all bus stops within settings.class_release_radius_m metres
        3. For the section's end_time, find the 15-min window it falls in
        4. For each day in meeting_days, add enrollment to that (stop, dow, window)

    Result is stored in StudentReleaseEvent (cleared and recomputed each call).
    """
    logger.info("Computing student release events...")

    # Load all buildings into memory
    result = await db.execute(select(IUBuilding))
    buildings: dict[str, IUBuilding] = {b.building_code: b for b in result.scalars()}

    # Load all stops
    result = await db.execute(select(Stop))
    stops: list[Stop] = list(result.scalars())

    if not stops:
        logger.warning("No stops in DB — run GTFS static load first")
        return 0

    # Load all in-person sections with times
    result = await db.execute(
        select(ClassSection).where(
            ClassSection.is_in_person == True,
            ClassSection.end_time.is_not(None),
            ClassSection.building_code.is_not(None),
            ClassSection.enrollment > 0,
        )
    )
    sections: list[ClassSection] = list(result.scalars())

    logger.info(f"Processing {len(sections)} sections against {len(stops)} stops")

    # Accumulate: (stop_id, dow, window_start) → student count
    release_counts: dict[tuple, int] = {}

    for section in sections:
        building = buildings.get(section.building_code)
        if not building or building.latitude is None:
            continue

        # Find nearby stops
        nearby = _stops_within_radius(
            stops, building.latitude, building.longitude,
            settings.class_release_radius_m
        )
        if not nearby:
            continue

        # Determine which 15-minute window the end_time falls in
        window_start = _floor_to_15min(section.end_time)
        window_end = _add_15min(window_start)

        # Determine which days of week this class meets
        days = _parse_days(section.meeting_days or "")
        if not days:
            # If no days specified, assume MWF as a conservative default
            days = [0, 2, 4]

        for stop in nearby:
            for dow in days:
                key = (stop.stop_id, dow, window_start)
                release_counts[key] = release_counts.get(key, 0) + section.enrollment

    # Clear existing release events and write new ones
    await db.execute(delete(StudentReleaseEvent))
    await db.flush()

    events = []
    for (stop_id, dow, window_start), count in release_counts.items():
        events.append(StudentReleaseEvent(
            stop_id=stop_id,
            day_of_week=dow,
            window_start=window_start,
            window_end=_add_15min(window_start),
            students_releasing=count,
            computed_at=datetime.utcnow(),
        ))

    if events:
        db.add_all(events)
        await db.flush()
        logger.info(f"Wrote {len(events)} student release events")

    return len(events)


async def get_students_releasing(
    db: AsyncSession, stop_id: str, at_time: datetime
) -> int:
    """
    Return how many students are releasing near stop_id within the next
    settings.class_release_lookahead_min minutes of at_time.
    This is called by the feature builder every inference cycle.
    """
    dow = at_time.weekday()
    t = at_time.time()

    result = await db.execute(
        select(StudentReleaseEvent).where(
            StudentReleaseEvent.stop_id == stop_id,
            StudentReleaseEvent.day_of_week == dow,
            StudentReleaseEvent.window_start >= t,
            StudentReleaseEvent.window_start <= _add_minutes(t, settings.class_release_lookahead_min),
        )
    )
    events = result.scalars().all()
    return sum(e.students_releasing for e in events)


# ── Utility functions ─────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two lat/lng points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _stops_within_radius(stops: list, lat: float, lon: float, radius_m: float) -> list:
    """Return all stops within radius_m metres of (lat, lon)."""
    return [
        s for s in stops
        if _haversine_m(lat, lon, s.stop_lat, s.stop_lon) <= radius_m
    ]


def _floor_to_15min(t: time) -> time:
    """Floor a time to the nearest 15-minute boundary."""
    minute = (t.minute // 15) * 15
    return time(t.hour, minute)


def _add_15min(t: time) -> time:
    """Add 15 minutes to a time, wrapping at midnight."""
    dt = datetime.combine(date.today(), t) + timedelta(minutes=15)
    return dt.time()


def _add_minutes(t: time, minutes: int) -> time:
    dt = datetime.combine(date.today(), t) + timedelta(minutes=minutes)
    return dt.time()


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None
