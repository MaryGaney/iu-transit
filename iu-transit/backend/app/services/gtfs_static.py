"""
app/services/gtfs_static.py
────────────────────────────
Downloads the Bloomington Transit GTFS static zip, parses each text file,
and upserts into the DB.

Run once on startup (if DB is empty) and then daily to pick up schedule changes.

GTFS static contains:
  routes.txt, stops.txt, trips.txt, stop_times.txt, shapes.txt, calendar.txt
"""

import io
import zipfile
import csv
import httpx
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, text

from app.core.config import settings
from app.core.logging import logger
from app.models.gtfs import Route, Stop, Trip, StopTime, Shape, GTFSLoadLog


async def load_gtfs_static(db: AsyncSession, force: bool = False) -> dict:
    """
    Download and parse GTFS static feed from Bloomington Transit.
    Returns a summary dict with counts of loaded records.

    Args:
        db: async DB session
        force: if True, reload even if data already exists
    """
    # Check if we already have data — must have BOTH routes AND stops
    if not force:
        route_count = (await db.execute(text("SELECT COUNT(*) FROM routes"))).scalar() or 0
        stop_count  = (await db.execute(text("SELECT COUNT(*) FROM stops"))).scalar() or 0
        if route_count > 0 and stop_count > 0:
            logger.info(f"GTFS static already loaded ({route_count} routes, {stop_count} stops). Use force=True to reload.")
            return {"skipped": True, "route_count": route_count, "stop_count": stop_count}
        elif route_count > 0 and stop_count == 0:
            logger.warning("Routes exist but stops table is EMPTY — forcing GTFS reload to fix incomplete load.")
            force = True

    logger.info(f"Downloading GTFS static from {settings.gtfs_static_url}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(settings.gtfs_static_url)
            response.raise_for_status()
            zip_bytes = response.content
    except Exception as e:
        logger.error(f"Failed to download GTFS static: {e}")
        await _log_load(db, success=False, error=str(e))
        raise

    logger.info(f"Downloaded {len(zip_bytes):,} bytes. Parsing...")

    counts = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            available = zf.namelist()
            logger.info(f"GTFS zip contains: {available}")

            # Clear existing data in dependency order
            await db.execute(delete(StopTime))
            await db.execute(delete(Shape))
            await db.execute(delete(Trip))
            await db.execute(delete(Stop))
            await db.execute(delete(Route))
            await db.flush()

            if "routes.txt" in available:
                counts["routes"] = await _load_routes(db, zf)
                await db.flush()
            if "stops.txt" in available:
                counts["stops"] = await _load_stops(db, zf)
                await db.flush()
            if "trips.txt" in available:
                counts["trips"] = await _load_trips(db, zf)
                await db.flush()
            if "stop_times.txt" in available:
                counts["stop_times"] = await _load_stop_times(db, zf)
                await db.flush()
            if "shapes.txt" in available:
                counts["shapes"] = await _load_shapes(db, zf)
                await db.flush()

        await db.flush()
        await _log_load(
            db,
            success=True,
            route_count=counts.get("routes", 0),
            stop_count=counts.get("stops", 0),
            trip_count=counts.get("trips", 0),
        )
        logger.info(f"GTFS static loaded: {counts}")
        return counts

    except Exception as e:
        logger.error(f"Failed to parse GTFS static: {e}")
        await _log_load(db, success=False, error=str(e))
        raise


async def _load_routes(db: AsyncSession, zf: zipfile.ZipFile) -> int:
    rows = _read_csv(zf, "routes.txt")
    objects = []
    for row in rows:
        objects.append(Route(
            route_id=row["route_id"],
            route_short_name=row.get("route_short_name", ""),
            route_long_name=row.get("route_long_name", ""),
            route_color=row.get("route_color", ""),
            route_text_color=row.get("route_text_color", ""),
            route_type=int(row.get("route_type", 3)),
        ))
    db.add_all(objects)
    return len(objects)


async def _load_stops(db: AsyncSession, zf: zipfile.ZipFile) -> int:
    rows = _read_csv(zf, "stops.txt")
    logger.info(f"stops.txt has {len(rows)} rows, columns: {list(rows[0].keys()) if rows else 'none'}")
    objects = []
    skipped = 0
    for row in rows:
        try:
            # Strip ALL values of whitespace/BOM characters
            row = {k.strip().lstrip('\ufeff'): (v.strip() if v else '') for k, v in row.items()}

            stop_id  = row.get("stop_id", "")
            stop_lat = row.get("stop_lat", "")
            stop_lon = row.get("stop_lon", "")

            if not stop_id or not stop_lat or not stop_lon:
                skipped += 1
                continue

            lat = float(stop_lat)
            lon = float(stop_lon)

            # Skip stops clearly outside Bloomington area (sanity check)
            if not (38.5 < lat < 40.0 and -87.5 < lon < -85.5):
                skipped += 1
                continue

            wb_raw = row.get("wheelchair_boarding", "0")
            wb = int(wb_raw) if wb_raw.isdigit() else 0

            objects.append(Stop(
                stop_id=stop_id,
                stop_name=row.get("stop_name", ""),
                stop_lat=lat,
                stop_lon=lon,
                stop_desc=row.get("stop_desc", ""),
                wheelchair_boarding=wb,
            ))
        except (ValueError, KeyError, TypeError) as e:
            skipped += 1
            logger.debug(f"Skipping stop row: {e} — row keys: {list(row.keys())[:5]}")

    logger.info(f"Stops: {len(objects)} loaded, {skipped} skipped")
    if objects:
        # Insert in batches to avoid SQLite limits
        BATCH = 200
        for i in range(0, len(objects), BATCH):
            db.add_all(objects[i:i+BATCH])
            await db.flush()
    return len(objects)


async def _load_trips(db: AsyncSession, zf: zipfile.ZipFile) -> int:
    rows = _read_csv(zf, "trips.txt")
    objects = []
    for row in rows:
        direction = row.get("direction_id", "")
        objects.append(Trip(
            trip_id=row["trip_id"],
            route_id=row["route_id"],
            service_id=row.get("service_id", ""),
            trip_headsign=row.get("trip_headsign", ""),
            direction_id=int(direction) if direction.isdigit() else None,
            shape_id=row.get("shape_id", ""),
        ))
    db.add_all(objects)
    return len(objects)


async def _load_stop_times(db: AsyncSession, zf: zipfile.ZipFile) -> int:
    """
    stop_times.txt can be very large (100k+ rows for a transit system).
    We batch-insert in chunks to avoid memory pressure.
    """
    rows = _read_csv(zf, "stop_times.txt")
    BATCH = 2000
    batch = []
    total = 0
    for row in rows:
        batch.append(StopTime(
            trip_id=row["trip_id"],
            stop_id=row["stop_id"],
            arrival_time=row.get("arrival_time", ""),
            departure_time=row.get("departure_time", ""),
            stop_sequence=int(row.get("stop_sequence", 0)),
        ))
        if len(batch) >= BATCH:
            db.add_all(batch)
            await db.flush()
            total += len(batch)
            batch = []
    if batch:
        db.add_all(batch)
        await db.flush()
        total += len(batch)
    return total


async def _load_shapes(db: AsyncSession, zf: zipfile.ZipFile) -> int:
    rows = _read_csv(zf, "shapes.txt")
    BATCH = 2000
    batch = []
    total = 0
    for row in rows:
        try:
            batch.append(Shape(
                shape_id=row["shape_id"],
                shape_pt_lat=float(row["shape_pt_lat"]),
                shape_pt_lon=float(row["shape_pt_lon"]),
                shape_pt_sequence=int(row["shape_pt_sequence"]),
            ))
        except (ValueError, KeyError):
            continue
        if len(batch) >= BATCH:
            db.add_all(batch)
            await db.flush()
            total += len(batch)
            batch = []
    if batch:
        db.add_all(batch)
        await db.flush()
        total += len(batch)
    return total


async def _log_load(db: AsyncSession, success: bool, error: str = None,
                    route_count: int = 0, stop_count: int = 0, trip_count: int = 0):
    db.add(GTFSLoadLog(
        loaded_at=datetime.utcnow(),
        route_count=route_count,
        stop_count=stop_count,
        trip_count=trip_count,
        success=success,
        error_message=error,
    ))
    await db.flush()


def _read_csv(zf: zipfile.ZipFile, filename: str) -> list[dict]:
    with zf.open(filename) as f:
        content = f.read().decode("utf-8-sig")  # strip BOM if present
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
