"""
app/api/admin.py
─────────────────
Admin / setup endpoints. Not exposed to end users.
Protected by a simple token in production.

POST /api/admin/load-gtfs-static         — (re)load GTFS static feed
POST /api/admin/load-schedule            — upload IU class schedule CSV
POST /api/admin/geocode-buildings        — geocode all building codes
POST /api/admin/compute-release-events   — recompute student release cache
POST /api/admin/train-model              — trigger LSTM training
GET  /api/admin/status                   — overall system status
"""

import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func

from app.core.database import get_db
from app.core.config import settings
from app.services.gtfs_static import load_gtfs_static
from app.services.class_schedule import parse_schedule_csv, compute_release_events
from app.services.geocoder import geocode_all_buildings
from app.ml.trainer import train_model
from app.ml.lstm_model import predictor
from app.core.logging import logger

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/load-gtfs-static")
async def trigger_gtfs_load(
    force: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Download and load the BT GTFS static feed."""
    counts = await load_gtfs_static(db, force=force)
    return {"success": True, "counts": counts}


@router.post("/load-schedule")
async def upload_schedule(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload the IU class schedule CSV.
    Accepts the exported CSV/TSV from the IU course schedule tool.
    The file is saved temporarily, parsed, then deleted.
    """
    if not file.filename.endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(400, "Expected a .csv, .tsv, or .txt file")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".csv", delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        count = await parse_schedule_csv(db, tmp_path)
        bld_count = await geocode_all_buildings(db)
        # compute_release_events requires stops to be loaded first
        from sqlalchemy import text as sqtext
        stop_count = (await db.execute(sqtext("SELECT COUNT(*) FROM stops"))).scalar() or 0
        event_count = 0
        if stop_count > 0:
            event_count = await compute_release_events(db)
        return {
            "success": True,
            "sections_loaded": count,
            "buildings_geocoded": bld_count,
            "stop_count": stop_count,
            "release_events_computed": event_count,
            "note": (
                "Release events computed successfully"
                if event_count > 0
                else f"No release events computed — {stop_count} stops in DB. "
                     "If stops=0, run POST /api/admin/load-gtfs-static first, then re-upload schedule."
            ),
        }
    finally:
        os.unlink(tmp_path)


@router.post("/geocode-buildings")
async def trigger_geocoding(db: AsyncSession = Depends(get_db)):
    """Populate IUBuilding table from known codes and IU API."""
    count = await geocode_all_buildings(db)
    return {"success": True, "buildings_geocoded": count}


@router.post("/compute-release-events")
async def trigger_release_events(db: AsyncSession = Depends(get_db)):
    """Recompute student release event cache."""
    count = await compute_release_events(db)
    return {"success": True, "events_computed": count}


@router.post("/train-model")
async def trigger_training(db: AsyncSession = Depends(get_db)):
    """Trigger LSTM training. Runs synchronously (blocks until complete)."""
    path = await train_model(db)
    if path:
        return {"success": True, "model_path": path, "model_loaded": predictor.is_loaded}
    else:
        return {"success": False, "reason": "Insufficient training data (need 100+ samples)"}


@router.get("/status")
async def get_system_status(db: AsyncSession = Depends(get_db)):
    """Full system status: DB counts, model state, last poll times."""

    async def count(table: str) -> int:
        r = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
        return r.scalar() or 0

    return {
        "database": {
            "routes": await count("routes"),
            "stops": await count("stops"),
            "trips": await count("trips"),
            "stop_times": await count("stop_times"),
            "vehicle_positions": await count("vehicle_positions"),
            "trip_updates": await count("trip_updates"),
            "class_sections": await count("class_sections"),
            "student_release_events": await count("student_release_events"),
            "weather_observations": await count("weather_observations"),
            "delay_predictions": await count("delay_predictions"),
            "buildings": await count("iu_buildings"),
        },
        "model": {
            "loaded": predictor.is_loaded,
            "mode": "lstm" if predictor.is_loaded else "heuristic_fallback",
            "model_path": settings.lstm_model_path,
            "model_exists": os.path.exists(settings.lstm_model_path),
        },
        "config": {
            "gtfs_rt_poll_interval_s": settings.gtfs_rt_poll_interval,
            "class_release_radius_m": settings.class_release_radius_m,
            "class_release_lookahead_min": settings.class_release_lookahead_min,
            "lstm_seq_len": settings.lstm_sequence_length,
        },
    }


@router.get("/probe-gtfs-rt")
async def probe_gtfs_rt_endpoint():
    """
    Try several known GTFS-RT URL patterns for Bloomington Transit and report
    which ones return valid protobuf data. Use this to identify the correct
    vehicle positions endpoint before setting GTFS_RT_VEHICLE_URL in .env.
    """
    import httpx
    from google.transit import gtfs_realtime_pb2

    candidates = [
        "https://bloomingtontransit.com/gtfs/vehiclepositions.pb",
        "https://bloomingtontransit.com/gtfs/realtime/vehiclepositions",
        "https://bloomingtontransit.com/realtime/vehiclepositions.pb",
        "https://gtfs.bloomingtontransit.com/vehiclepositions",
        "https://api.transloc.com/api/2/vehicles.pb",
    ]

    results = []
    async with httpx.AsyncClient(timeout=8.0) as client:
        for url in candidates:
            try:
                resp = await client.get(url, headers={
                    "Accept": "application/x-protobuf, */*",
                    "User-Agent": "IUTransitTracker/1.0",
                })
                status = resp.status_code
                if status == 200:
                    try:
                        feed = gtfs_realtime_pb2.FeedMessage()
                        feed.ParseFromString(resp.content)
                        entity_count = len(feed.entity)
                        results.append({
                            "url": url, "status": status,
                            "valid_protobuf": True,
                            "entity_count": entity_count,
                        })
                    except Exception:
                        results.append({
                            "url": url, "status": status,
                            "valid_protobuf": False,
                            "note": "200 but not valid protobuf — may be HTML or JSON",
                        })
                else:
                    results.append({"url": url, "status": status, "valid_protobuf": False})
            except Exception as e:
                results.append({"url": url, "error": str(e), "valid_protobuf": False})

    working = [r for r in results if r.get("valid_protobuf")]
    return {
        "probe_results": results,
        "recommended_url": working[0]["url"] if working else None,
        "action": (
            f"Set GTFS_RT_VEHICLE_URL={working[0]['url']} in your .env"
            if working else
            "No working GTFS-RT endpoint found. Contact Bloomington Transit directly."
        ),
    }


@router.post("/recompute")
async def recompute_all(db: AsyncSession = Depends(get_db)):
    """
    Re-run geocoding and student release event computation without re-uploading data.
    Use this when:
      - You uploaded the schedule but release events = 0
      - You want to force a refresh after changing the radius/lookahead settings
    """
    from sqlalchemy import text as sqtext
    results = {}

    # 1. Re-geocode buildings
    bld_count = await geocode_all_buildings(db)
    results["buildings_geocoded"] = bld_count

    # 2. Check we have both sections and stops
    section_count = (await db.execute(sqtext("SELECT COUNT(*) FROM class_sections WHERE is_in_person=1 AND end_time IS NOT NULL"))).scalar() or 0
    stop_count = (await db.execute(sqtext("SELECT COUNT(*) FROM stops"))).scalar() or 0
    results["sections_with_time"] = section_count
    results["stops_in_db"] = stop_count

    if section_count == 0:
        return {"success": False, "error": "No class sections found. Upload schedule first via POST /api/admin/load-schedule", **results}
    if stop_count == 0:
        return {"success": False, "error": "No stops found. Load GTFS static first via POST /api/admin/load-gtfs-static", **results}

    # 3. Compute release events
    from app.services.class_schedule import compute_release_events
    event_count = await compute_release_events(db)
    results["release_events_computed"] = event_count

    # 4. Report which buildings are still missing coords (for debugging)
    missing = await db.execute(sqtext("""
        SELECT DISTINCT cs.building_code
        FROM class_sections cs
        LEFT JOIN iu_buildings ib ON cs.building_code = ib.building_code
        WHERE cs.is_in_person=1
          AND (ib.building_code IS NULL OR ib.latitude IS NULL)
        LIMIT 20
    """))
    missing_codes = [r[0] for r in missing.fetchall()]
    results["buildings_still_missing_coords"] = missing_codes

    return {
        "success": True,
        **results,
        "note": (
            f"✅ {event_count} release events computed across {stop_count} stops"
            if event_count > 0
            else f"⚠️ 0 events — {len(missing_codes)} building codes have no coordinates: {missing_codes[:10]}"
        )
    }


@router.get("/debug/routes")
async def debug_routes(db: AsyncSession = Depends(get_db)):
    """
    Show exactly what route_id and route_short_name are stored in the DB,
    and what vehicle route_ids are currently being reported by the RT feed.
    Use this to diagnose why bus labels show vehicle IDs instead of route names.
    """
    from app.services.gtfs_realtime import get_current_positions

    # What's in the routes table
    result = await db.execute(text(
        "SELECT route_id, route_short_name, route_long_name FROM routes ORDER BY route_id LIMIT 30"
    ))
    db_routes = [
        {"route_id": r[0], "short_name": r[1], "long_name": r[2]}
        for r in result.fetchall()
    ]

    # What route_ids are live vehicles reporting right now
    positions = await get_current_positions()
    live_route_ids = list({v.get("route_id") for v in positions if v.get("route_id")})

    # Check which live route_ids match DB routes
    db_route_ids = {r["route_id"] for r in db_routes}
    matched   = [rid for rid in live_route_ids if rid in db_route_ids]
    unmatched = [rid for rid in live_route_ids if rid not in db_route_ids]

    return {
        "db_routes": db_routes,
        "live_vehicle_route_ids": sorted(live_route_ids),
        "matched_to_db": sorted(matched),
        "unmatched_in_db": sorted(unmatched),
        "diagnosis": (
            "✅ All live route_ids match DB routes"
            if not unmatched else
            f"⚠️ {len(unmatched)} live route_id(s) not found in DB: {unmatched}. "
            "This means vehicle positions use different IDs than the static GTFS. "
            "Run POST /api/admin/load-gtfs-static?force=true to reload."
        ),
        "short_name_populated": sum(1 for r in db_routes if r["short_name"]),
        "short_name_empty": sum(1 for r in db_routes if not r["short_name"]),
        "label_advice": (
            "short_name is populated — frontend should use it"
            if any(r["short_name"] for r in db_routes)
            else "⚠️ short_name is EMPTY for all routes — frontend will fall back to route_id as label"
        ),
    }


@router.get("/debug/schedule")
async def debug_schedule(db: AsyncSession = Depends(get_db)):
    """
    Test that the class schedule is loaded and the student-release signal is working.
    Shows:
      - How many sections are loaded
      - Classes releasing RIGHT NOW (next 15 min)
      - Classes starting RIGHT NOW (next 20 min)
      - Sample of StudentReleaseEvent rows near known campus stops
    """
    from datetime import datetime, timedelta
    from app.services.class_schedule import get_students_releasing
    from app.models.schedule import ClassSection, StudentReleaseEvent

    now = datetime.utcnow()
    # Convert UTC to Eastern (IU is EDT = UTC-4 in fall, EST = UTC-5 in spring)
    # Simple approximation: subtract 4 hours for EDT
    local_now = datetime.now()  # use system local time

    # Count sections
    total = (await db.execute(text("SELECT COUNT(*) FROM class_sections"))).scalar()
    in_person = (await db.execute(text(
        "SELECT COUNT(*) FROM class_sections WHERE is_in_person=1"
    ))).scalar()
    with_time = (await db.execute(text(
        "SELECT COUNT(*) FROM class_sections WHERE is_in_person=1 AND end_time IS NOT NULL"
    ))).scalar()
    release_events = (await db.execute(text(
        "SELECT COUNT(*) FROM student_release_events"
    ))).scalar()

    # Classes releasing in next 15 min (using local time)
    t_start = local_now.strftime("%H:%M:%S")
    t_end   = (local_now + timedelta(minutes=15)).strftime("%H:%M:%S")
    releasing_now = await db.execute(text("""
        SELECT cs.course_id, cs.end_time, cs.enrollment, cs.building_code
        FROM class_sections cs
        WHERE cs.is_in_person = 1
          AND cs.end_time >= :t_start
          AND cs.end_time <= :t_end
          AND cs.enrollment > 0
        ORDER BY cs.enrollment DESC
        LIMIT 10
    """), {"t_start": t_start, "t_end": t_end})
    releasing_rows = [
        {"course": r[0], "end_time": str(r[1]), "enrollment": r[2], "building": r[3]}
        for r in releasing_now.fetchall()
    ]

    # Classes starting in next 20 min
    t_end2 = (local_now + timedelta(minutes=20)).strftime("%H:%M:%S")
    starting_now = await db.execute(text("""
        SELECT cs.course_id, cs.start_time, cs.enrollment, cs.building_code
        FROM class_sections cs
        WHERE cs.is_in_person = 1
          AND cs.start_time >= :t_start
          AND cs.start_time <= :t_end
          AND cs.enrollment > 0
        ORDER BY cs.enrollment DESC
        LIMIT 10
    """), {"t_start": t_start, "t_end": t_end2})
    starting_rows = [
        {"course": r[0], "start_time": str(r[1]), "enrollment": r[2], "building": r[3]}
        for r in starting_now.fetchall()
    ]

    # Sample release events for a well-known stop (IMU stop area)
    sample_events = await db.execute(text("""
        SELECT sre.stop_id, sre.day_of_week, sre.window_start, sre.students_releasing
        FROM student_release_events sre
        ORDER BY sre.students_releasing DESC
        LIMIT 10
    """))
    top_events = [
        {"stop_id": r[0], "dow": r[1], "window": str(r[2]), "students": r[3]}
        for r in sample_events.fetchall()
    ]

    # Next 3 class end times (to show what's coming up)
    upcoming = await db.execute(text("""
        SELECT DISTINCT end_time, COUNT(*) as sections, SUM(enrollment) as total_students
        FROM class_sections
        WHERE is_in_person=1 AND end_time IS NOT NULL
        GROUP BY end_time
        ORDER BY end_time
        LIMIT 20
    """))
    schedule_times = [
        {"end_time": str(r[0]), "sections": r[1], "total_students": r[2]}
        for r in upcoming.fetchall()
    ]

    return {
        "current_local_time": local_now.strftime("%H:%M:%S"),
        "current_day_of_week": local_now.strftime("%A"),
        "sections": {
            "total_rows": total,
            "in_person": in_person,
            "in_person_with_time": with_time,
            "student_release_events_computed": release_events,
        },
        "classes_releasing_next_15min": releasing_rows,
        "classes_starting_next_20min": starting_rows,
        "top_crowding_events_in_db": top_events,
        "all_class_end_times": schedule_times,
        "diagnosis": (
            "✅ Schedule loaded and release events computed"
            if with_time > 0 and release_events > 0
            else "⚠️ No release events computed yet"
            if with_time > 0
            else "❌ No in-person sections with times found — check CSV upload"
        ),
    }
