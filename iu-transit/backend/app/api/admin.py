"""
app/api/admin.py
─────────────────
Admin / setup endpoints.

POST /api/admin/load-gtfs-static
POST /api/admin/load-schedule
POST /api/admin/geocode-buildings
POST /api/admin/compute-release-events
POST /api/admin/recompute
POST /api/admin/train-model
GET  /api/admin/status
GET  /api/admin/probe-gtfs-rt
GET  /api/admin/debug/routes
GET  /api/admin/debug/schedule
"""

import os
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.core.config import settings
from app.core.logging import logger
from app.services.gtfs_static import load_gtfs_static
from app.services.class_schedule import parse_schedule_csv, compute_release_events
from app.services.geocoder import geocode_all_buildings

# train_model is imported lazily inside the endpoint to avoid
# pulling in trainer.py at module load time (trainer.py imports torch)
# ml.lstm_model.predictor is safe — it has TORCH_AVAILABLE guards
from app.ml.lstm_model import predictor

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Data loading endpoints ─────────────────────────────────────────────────────

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
    """Upload the IU class schedule CSV."""
    import tempfile

    if not file.filename.endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(400, "Expected a .csv, .tsv, or .txt file")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".csv", delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        count     = await parse_schedule_csv(db, tmp_path)
        bld_count = await geocode_all_buildings(db)

        stop_count = (
            await db.execute(text("SELECT COUNT(*) FROM stops"))
        ).scalar() or 0

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
                else (
                    f"No release events — {stop_count} stops in DB. "
                    "Run POST /api/admin/load-gtfs-static first."
                    if stop_count == 0
                    else "Stops loaded but no events computed — check building geocoding."
                )
            ),
        }
    finally:
        os.unlink(tmp_path)


@router.post("/geocode-buildings")
async def trigger_geocoding(db: AsyncSession = Depends(get_db)):
    count = await geocode_all_buildings(db)
    return {"success": True, "buildings_geocoded": count}


@router.post("/compute-release-events")
async def trigger_release_events(db: AsyncSession = Depends(get_db)):
    count = await compute_release_events(db)
    return {"success": True, "events_computed": count}


@router.post("/recompute")
async def recompute_all(db: AsyncSession = Depends(get_db)):
    """Re-run geocoding and student release event computation."""
    results = {}

    bld_count = await geocode_all_buildings(db)
    results["buildings_geocoded"] = bld_count

    section_count = (
        await db.execute(
            text("SELECT COUNT(*) FROM class_sections "
                 "WHERE is_in_person=1 AND end_time IS NOT NULL")
        )
    ).scalar() or 0
    stop_count = (
        await db.execute(text("SELECT COUNT(*) FROM stops"))
    ).scalar() or 0

    results["sections_with_time"] = section_count
    results["stops_in_db"]        = stop_count

    if section_count == 0:
        return {
            "success": False,
            "error": "No class sections found. Upload schedule first.",
            **results,
        }
    if stop_count == 0:
        return {
            "success": False,
            "error": "No stops found. Load GTFS static first.",
            **results,
        }

    event_count = await compute_release_events(db)
    results["release_events_computed"] = event_count

    missing_result = await db.execute(text("""
        SELECT DISTINCT cs.building_code
        FROM class_sections cs
        LEFT JOIN iu_buildings ib ON cs.building_code = ib.building_code
        WHERE cs.is_in_person=1
          AND (ib.building_code IS NULL OR ib.latitude IS NULL)
        LIMIT 20
    """))
    missing_codes = [r[0] for r in missing_result.fetchall()]
    results["buildings_still_missing_coords"] = missing_codes

    return {
        "success": True,
        **results,
        "note": (
            f"✅ {event_count} release events computed"
            if event_count > 0
            else f"⚠️ 0 events — {len(missing_codes)} buildings missing coords: {missing_codes[:10]}"
        ),
    }


# ── ML training ────────────────────────────────────────────────────────────────

@router.post("/train-model")
async def trigger_training(db: AsyncSession = Depends(get_db)):
    """Trigger LSTM training (requires torch to be installed)."""
    # Lazy import — keeps trainer.py from being imported at app startup
    # when torch is not installed
    try:
        from app.ml.trainer import train_model, TORCH_AVAILABLE
    except ImportError:
        return {
            "success": False,
            "reason": (
                "PyTorch is not installed. "
                "Add 'torch==2.3.0' to requirements.txt and redeploy."
            ),
        }

    if not TORCH_AVAILABLE:
        return {
            "success": False,
            "reason": (
                "PyTorch is not installed on this deployment. "
                "Add 'torch==2.3.0' to requirements.txt and redeploy to enable training."
            ),
        }

    path = await train_model(db)
    if path:
        return {
            "success": True,
            "model_path": path,
            "model_loaded": predictor.is_loaded,
        }
    return {
        "success": False,
        "reason": "Insufficient training data (need 100+ delay samples)",
    }


# ── Status ──────────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_system_status(db: AsyncSession = Depends(get_db)):
    """Full system status — DB counts, model state, config."""

    async def count(table: str) -> int:
        r = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
        return r.scalar() or 0

    # Check torch availability without importing trainer
    try:
        import torch as _torch
        torch_available = True
    except ImportError:
        torch_available = False

    return {
        "database": {
            "routes":                  await count("routes"),
            "stops":                   await count("stops"),
            "trips":                   await count("trips"),
            "stop_times":              await count("stop_times"),
            "vehicle_positions":       await count("vehicle_positions"),
            "trip_updates":            await count("trip_updates"),
            "class_sections":          await count("class_sections"),
            "student_release_events":  await count("student_release_events"),
            "weather_observations":    await count("weather_observations"),
            "delay_predictions":       await count("delay_predictions"),
            "buildings":               await count("iu_buildings"),
        },
        "model": {
            "loaded":        predictor.is_loaded,
            "mode":          "lstm" if predictor.is_loaded else "heuristic_fallback",
            "model_path":    settings.lstm_model_path,
            "model_exists":  os.path.exists(settings.lstm_model_path),
            "torch_installed": torch_available,
        },
        "config": {
            "gtfs_rt_poll_interval_s":      settings.gtfs_rt_poll_interval,
            "class_release_radius_m":       settings.class_release_radius_m,
            "class_release_lookahead_min":  settings.class_release_lookahead_min,
            "lstm_seq_len":                 settings.lstm_sequence_length,
        },
    }


# ── GTFS-RT probe ──────────────────────────────────────────────────────────────

@router.get("/probe-gtfs-rt")
async def probe_gtfs_rt_endpoint():
    """Try known GTFS-RT URL patterns and report which ones return valid protobuf."""
    import httpx
    from google.transit import gtfs_realtime_pb2

    candidates = [
        "https://s3.amazonaws.com/etatransit.gtfs/bloomingtontransit.etaspot.net/position_updates.pb",
        "https://bloomingtontransit.com/gtfs/vehiclepositions.pb",
        "https://bloomingtontransit.com/gtfs/realtime/vehiclepositions.pb",
    ]

    results = []
    async with httpx.AsyncClient(timeout=8.0) as client:
        for url in candidates:
            try:
                resp = await client.get(
                    url,
                    headers={"Accept": "application/x-protobuf, */*"},
                )
                if resp.status_code == 200:
                    try:
                        feed = gtfs_realtime_pb2.FeedMessage()
                        feed.ParseFromString(resp.content)
                        results.append({
                            "url": url, "status": 200,
                            "valid_protobuf": True,
                            "entity_count": len(feed.entity),
                        })
                    except Exception:
                        results.append({
                            "url": url, "status": 200,
                            "valid_protobuf": False,
                            "note": "200 but not valid protobuf",
                        })
                else:
                    results.append({"url": url, "status": resp.status_code, "valid_protobuf": False})
            except Exception as e:
                results.append({"url": url, "error": str(e), "valid_protobuf": False})

    working = [r for r in results if r.get("valid_protobuf")]
    return {
        "probe_results": results,
        "recommended_url": working[0]["url"] if working else None,
    }


# ── Debug endpoints ────────────────────────────────────────────────────────────

@router.get("/debug/routes")
async def debug_routes(db: AsyncSession = Depends(get_db)):
    """Show route_id / short_name from DB vs live vehicle route_ids."""
    from app.services.gtfs_realtime import get_current_positions

    result = await db.execute(text(
        "SELECT route_id, route_short_name, route_long_name "
        "FROM routes ORDER BY route_id LIMIT 30"
    ))
    db_routes = [
        {"route_id": r[0], "short_name": r[1], "long_name": r[2]}
        for r in result.fetchall()
    ]

    positions   = await get_current_positions()
    live_ids    = list({v.get("route_id") for v in positions if v.get("route_id")})
    db_ids      = {r["route_id"] for r in db_routes}
    matched     = [rid for rid in live_ids if rid in db_ids]
    unmatched   = [rid for rid in live_ids if rid not in db_ids]

    return {
        "db_routes":               db_routes,
        "live_vehicle_route_ids":  sorted(live_ids),
        "matched_to_db":           sorted(matched),
        "unmatched_in_db":         sorted(unmatched),
        "diagnosis": (
            "✅ All live route_ids match DB routes"
            if not unmatched
            else f"⚠️ {len(unmatched)} live route_id(s) not in DB: {unmatched}"
        ),
        "short_name_populated": sum(1 for r in db_routes if r["short_name"]),
        "short_name_empty":     sum(1 for r in db_routes if not r["short_name"]),
    }


@router.get("/debug/schedule")
async def debug_schedule(db: AsyncSession = Depends(get_db)):
    """Show class schedule status and current release/starting events."""
    from datetime import datetime, timedelta

    now = datetime.now()

    total       = (await db.execute(text("SELECT COUNT(*) FROM class_sections"))).scalar()
    in_person   = (await db.execute(text(
        "SELECT COUNT(*) FROM class_sections WHERE is_in_person=1"
    ))).scalar()
    with_time   = (await db.execute(text(
        "SELECT COUNT(*) FROM class_sections "
        "WHERE is_in_person=1 AND end_time IS NOT NULL"
    ))).scalar()
    rel_events  = (await db.execute(text(
        "SELECT COUNT(*) FROM student_release_events"
    ))).scalar()

    t_start = now.strftime("%H:%M:%S")
    t_end15 = (now + timedelta(minutes=15)).strftime("%H:%M:%S")
    t_end20 = (now + timedelta(minutes=20)).strftime("%H:%M:%S")

    releasing = await db.execute(text("""
        SELECT cs.course_id, cs.end_time, cs.enrollment, cs.building_code
        FROM class_sections cs
        WHERE cs.is_in_person=1 AND cs.end_time>=:s AND cs.end_time<=:e
          AND cs.enrollment>0
        ORDER BY cs.enrollment DESC LIMIT 10
    """), {"s": t_start, "e": t_end15})

    starting = await db.execute(text("""
        SELECT cs.course_id, cs.start_time, cs.enrollment, cs.building_code
        FROM class_sections cs
        WHERE cs.is_in_person=1 AND cs.start_time>=:s AND cs.start_time<=:e
          AND cs.enrollment>0
        ORDER BY cs.enrollment DESC LIMIT 10
    """), {"s": t_start, "e": t_end20})

    top_events = await db.execute(text("""
        SELECT stop_id, day_of_week, window_start, students_releasing
        FROM student_release_events
        ORDER BY students_releasing DESC LIMIT 10
    """))

    return {
        "current_local_time":         now.strftime("%H:%M:%S"),
        "current_day_of_week":        now.strftime("%A"),
        "sections": {
            "total_rows":                        total,
            "in_person":                         in_person,
            "in_person_with_time":               with_time,
            "student_release_events_computed":   rel_events,
        },
        "classes_releasing_next_15min": [
            {"course": r[0], "end_time": str(r[1]),
             "enrollment": r[2], "building": r[3]}
            for r in releasing.fetchall()
        ],
        "classes_starting_next_20min": [
            {"course": r[0], "start_time": str(r[1]),
             "enrollment": r[2], "building": r[3]}
            for r in starting.fetchall()
        ],
        "top_crowding_events_in_db": [
            {"stop_id": r[0], "dow": r[1],
             "window": str(r[2]), "students": r[3]}
            for r in top_events.fetchall()
        ],
        "diagnosis": (
            "✅ Schedule loaded and release events computed"
            if with_time > 0 and rel_events > 0
            else "⚠️ No release events — run POST /api/admin/recompute"
            if with_time > 0
            else "❌ No in-person sections with times — re-upload CSV"
        ),
    }
