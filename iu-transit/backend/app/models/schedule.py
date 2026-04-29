"""
app/models/schedule.py
──────────────────────
ORM models for IU class schedule data.

The source CSV has 8,337 rows. We care about:
  - In-person classes only (Instruction Mode = "In Person")
  - Building + room (to geocode to lat/lng)
  - Enrollment (Stu Enrl column)
  - Start/end dates and meeting days/times
  - Class end time drives the "students releasing soon" feature

Building geocodes are stored in IUBuilding, keyed by building code (e.g. "BLTH").
"""

from datetime import datetime, date, time
from sqlalchemy import Integer, String, Float, DateTime, Date, Time, Boolean, Index, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class IUBuilding(Base):
    """
    Map from IU building code to lat/lng.
    Populated once by scripts/geocode_buildings.py, then referenced by ClassSection.
    """
    __tablename__ = "iu_buildings"

    building_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    building_name: Mapped[str] = mapped_column(String(256), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)
    address: Mapped[str] = mapped_column(String(512), nullable=True)


class ClassSection(Base):
    """
    One row per class section from the IU schedule export.

    Key fields for the delay model:
      - end_time: when students start flooding out
      - enrollment: how many students are releasing
      - building_code → IUBuilding → lat/lng → proximity to bus stop
      - meeting_days: which days this class meets (M/T/W/R/F/S/U)
      - start_date / end_date: active semester window

    Fields planned for future syllabi integration:
      - attendance_required: scraped from syllabus (nullable until implemented)
      - next_exam_date: scraped from syllabus (affects how many actually show up)
    """
    __tablename__ = "class_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identifiers
    class_nbr: Mapped[int] = mapped_column(Integer, index=True)
    course_id: Mapped[str] = mapped_column(String(32))          # e.g. "CSCI-P 536"
    component: Mapped[str] = mapped_column(String(32))          # Lecture, Lab, Discussion
    instruction_mode: Mapped[str] = mapped_column(String(32))   # "In Person", "Online", etc.
    is_in_person: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Meeting pattern
    meeting_days: Mapped[str] = mapped_column(String(16), nullable=True)  # "MWF", "TR", etc.
    start_date: Mapped[date] = mapped_column(Date, nullable=True)
    end_date: Mapped[date] = mapped_column(Date, nullable=True)
    start_time: Mapped[time] = mapped_column(Time, nullable=True)
    end_time: Mapped[time] = mapped_column(Time, nullable=True, index=True)
    total_minutes: Mapped[int] = mapped_column(Integer, nullable=True)

    # Location
    building_code: Mapped[str] = mapped_column(String(16), nullable=True, index=True)
    room: Mapped[str] = mapped_column(String(32), nullable=True)

    # Enrollment / capacity
    enrollment: Mapped[int] = mapped_column(Integer, default=0)
    room_capacity: Mapped[int] = mapped_column(Integer, nullable=True)
    e_cap: Mapped[int] = mapped_column(Integer, nullable=True)

    # Academic context
    academic_term: Mapped[str] = mapped_column(String(16), nullable=True)  # e.g. "42584"
    dept_name: Mapped[str] = mapped_column(String(128), nullable=True)

    # Future: syllabus-derived fields (nullable until syllabi scraper is built)
    attendance_required: Mapped[bool] = mapped_column(Boolean, nullable=True)
    next_exam_date: Mapped[date] = mapped_column(Date, nullable=True)
    attendance_probability: Mapped[float] = mapped_column(Float, nullable=True)

    loaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_cs_inperson_time", "is_in_person", "end_time"),
        Index("ix_cs_building_time", "building_code", "end_time"),
    )


class StudentReleaseEvent(Base):
    """
    Pre-computed cache of "how many students release near stop X at time T".
    Recomputed hourly by the scheduler. This is what the LSTM feature builder
    queries in real time — we don't want to do the proximity math every 15 seconds.

    stop_id + window_start is effectively a composite key.
    """
    __tablename__ = "student_release_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stop_id: Mapped[str] = mapped_column(String(64), index=True)
    day_of_week: Mapped[int] = mapped_column(Integer)  # 0=Mon … 6=Sun
    # Time window (15-minute buckets)
    window_start: Mapped[time] = mapped_column(Time, index=True)
    window_end: Mapped[time] = mapped_column(Time)
    students_releasing: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sre_stop_day_window", "stop_id", "day_of_week", "window_start"),
    )
