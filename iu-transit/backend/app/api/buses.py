"""
app/api/buses.py
─────────────────
REST + WebSocket endpoints for live bus data.

Endpoints:
  GET  /api/buses/routes          — all routes with colour
  GET  /api/buses/stops           — all stops with lat/lng
  GET  /api/buses/shapes/{route}  — route polyline points
  GET  /api/buses/vehicles        — current vehicle positions (HTTP poll fallback)
  WS   /api/buses/live            — WebSocket stream of vehicle positions
  GET  /api/buses/stop/{stop_id}/schedule — scheduled arrivals for a stop
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, timedelta

from app.core.database import get_db
from app.services.weather import get_current_weather
from app.services.class_schedule import get_students_releasing
from app.core.websocket import manager
from app.services.gtfs_realtime import get_current_positions
from app.models.gtfs import Route, Stop, Shape, Trip, StopTime, TripUpdate
from app.core.logging import logger

router = APIRouter(prefix="/api/buses", tags=["buses"])


@router.get("/routes")
async def get_routes(db: AsyncSession = Depends(get_db)):
    """All transit routes with display colours."""
    result = await db.execute(select(Route))
    routes = result.scalars().all()
    return [
        {
            "route_id": r.route_id,
            "short_name": r.route_short_name,
            "long_name": r.route_long_name,
            "color": f"#{r.route_color}" if r.route_color else "#1D4ED8",
            "text_color": f"#{r.route_text_color}" if r.route_text_color else "#FFFFFF",
        }
        for r in routes
    ]


@router.get("/stops")
async def get_stops(db: AsyncSession = Depends(get_db)):
    """All stops with coordinates."""
    result = await db.execute(select(Stop))
    stops = result.scalars().all()
    return [
        {
            "stop_id": s.stop_id,
            "name": s.stop_name,
            "lat": s.stop_lat,
            "lng": s.stop_lon,
            "description": s.stop_desc,
        }
        for s in stops
    ]


@router.get("/shapes/{route_id}")
async def get_route_shape(route_id: str, db: AsyncSession = Depends(get_db)):
    """
    Return ALL shape polylines for a route (one per direction/variant).
    Routes like 1N have separate outbound and inbound shapes — we return
    all of them so the frontend can draw the complete route coverage.
    """
    # Get ALL distinct shape_ids for this route
    result = await db.execute(
        select(Trip.shape_id)
        .where(Trip.route_id == route_id, Trip.shape_id.is_not(None))
        .distinct()
    )
    shape_ids = [row[0] for row in result.all()]
    if not shape_ids:
        return {"route_id": route_id, "shapes": [], "points": []}

    # Fetch points for every shape, keyed by shape_id
    shapes_out = []
    all_points = []  # merged flat list for backwards compat
    for shape_id in shape_ids:
        result = await db.execute(
            select(Shape)
            .where(Shape.shape_id == shape_id)
            .order_by(Shape.shape_pt_sequence)
        )
        pts = result.scalars().all()
        if pts:
            point_list = [{"lat": p.shape_pt_lat, "lng": p.shape_pt_lon} for p in pts]
            shapes_out.append({"shape_id": shape_id, "points": point_list})
            all_points.extend(point_list)

    return {
        "route_id": route_id,
        "shapes": shapes_out,          # multiple polylines, one per direction
        "points": all_points,          # flat merge (legacy fallback)
    }


@router.get("/vehicles")
async def get_vehicles():
    """
    HTTP fallback for current vehicle positions.
    Clients should prefer the WebSocket endpoint for live updates.
    """
    positions = await get_current_positions()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "vehicles": positions,
        "count": len(positions),
    }


@router.websocket("/live")
async def vehicle_position_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live vehicle positions.
    The client connects once and receives position updates every ~15 seconds
    as buses are polled from the GTFS-RT feed.

    On connect: immediately send the last known positions.
    On each GTFS-RT poll: the scheduler calls broadcast_vehicle_positions()
    which pushes to all connected clients through this manager.
    """
    await manager.connect(websocket)
    try:
        # Send current positions immediately on connect
        positions = await get_current_positions()
        if positions:
            import json
            await websocket.send_text(json.dumps({
                "type": "vehicle_positions",
                "timestamp": datetime.utcnow().isoformat(),
                "vehicles": positions,
            }))

        # Keep alive — client messages are ignored but disconnects are detected
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WS error: {e}")
        manager.disconnect(websocket)


@router.get("/stop/{stop_id}/schedule")
async def get_stop_schedule(
    stop_id: str,
    db: AsyncSession = Depends(get_db),
    lookahead_minutes: int = 60,
):
    """
    Return upcoming scheduled arrivals at a stop for the next N minutes.
    Joins stop_times with trips and routes for display.
    """
    from datetime import time as dtime
    now = datetime.now()
    # GTFS times can exceed 24:00:00 for overnight service
    # For simplicity we query by HH:MM:SS string comparison in the next hour
    time_from = now.strftime("%H:%M:%S")
    time_to = (now + timedelta(minutes=lookahead_minutes)).strftime("%H:%M:%S")

    result = await db.execute(
        select(StopTime, Trip, Route)
        .join(Trip, StopTime.trip_id == Trip.trip_id)
        .join(Route, Trip.route_id == Route.route_id)
        .where(
            StopTime.stop_id == stop_id,
            StopTime.arrival_time >= time_from,
            StopTime.arrival_time <= time_to,
        )
        .order_by(StopTime.arrival_time)
        .limit(20)
    )
    rows = result.all()

    # If no results, widen to 3 hours — may be a low-frequency stop
    if not rows:
        time_to_wide = (now + timedelta(minutes=180)).strftime("%H:%M:%S")
        result2 = await db.execute(
            select(StopTime, Trip, Route)
            .join(Trip, StopTime.trip_id == Trip.trip_id)
            .join(Route, Trip.route_id == Route.route_id)
            .where(
                StopTime.stop_id == stop_id,
                StopTime.arrival_time >= time_from,
                StopTime.arrival_time <= time_to_wide,
            )
            .order_by(StopTime.arrival_time)
            .limit(10)
        )
        rows = result2.all()

    # Check total stop_times in DB for diagnostics
    from sqlalchemy import func as sqfunc
    total_st = (await db.execute(text("SELECT COUNT(*) FROM stop_times"))).scalar() or 0

    return {
        "stop_id": stop_id,
        "queried_at": now.isoformat(),
        "stop_times_in_db": total_st,
        "arrivals": [
            {
                "arrival_time": st.arrival_time,
                "route_id": route.route_id,
                "route_short_name": route.route_short_name,
                "route_long_name": route.route_long_name,
                "route_color": f"#{route.route_color}" if route.route_color else "#1D4ED8",
                "headsign": trip.trip_headsign,
                "trip_id": trip.trip_id,
            }
            for st, trip, route in rows
        ],
    }


@router.get("/stop/{stop_id}/recent-delays")
async def get_recent_delays(
    stop_id: str,
    db: AsyncSession = Depends(get_db),
    hours: int = 2,
):
    """
    Recent actual delays at a stop (from trip_updates log).
    Used by the frontend to show historical context in the stop popup.
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    result = await db.execute(
        select(TripUpdate)
        .where(
            TripUpdate.stop_id == stop_id,
            TripUpdate.timestamp >= since,
            TripUpdate.arrival_delay_seconds.is_not(None),
        )
        .order_by(desc(TripUpdate.timestamp))
        .limit(50)
    )
    updates = result.scalars().all()

    if not updates:
        return {"stop_id": stop_id, "avg_delay_seconds": 0, "samples": 0, "delays": []}

    delays = [u.arrival_delay_seconds for u in updates]
    return {
        "stop_id": stop_id,
        "avg_delay_seconds": round(sum(delays) / len(delays), 1),
        "max_delay_seconds": round(max(delays), 1),
        "samples": len(delays),
        "delays": [
            {
                "route_id": u.route_id,
                "delay_seconds": u.arrival_delay_seconds,
                "timestamp": u.timestamp.isoformat(),
            }
            for u in updates[:10]
        ],
    }


@router.get("/vehicle/{vehicle_id}/occupancy")
async def get_vehicle_occupancy(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Estimate occupancy / delay probability for a specific bus.

    Factors (research basis):
    - Class RELEASING: Cats et al. (2016) — class-release events increase dwell
      time 40-120s at university stops. We check a 15-min lookahead window.
    - Class STARTING: symmetric effect — students board 10-20 min before class.
      We check a 20-min lookahead for upcoming class starts.
    - Weather severity: rain/snow add 15-30% to boarding time (TRB 2019).
    - Temperature: >90°F or <20°F drives significantly more ridership
      (Guo et al. 2020 — every 10°F above 85 adds ~8% ridership).
    All temperatures reported in Fahrenheit for US users.
    """
    from datetime import datetime as dt, timedelta
    now = dt.utcnow()

    positions = await get_current_positions()
    vehicle = next((v for v in positions if v["vehicle_id"] == vehicle_id), None)
    route_id = vehicle["route_id"] if vehicle else None

    # ── Weather ───────────────────────────────────────────────────────────────
    weather = get_current_weather()
    weather_severity = (weather or {}).get("weather_severity", 0.0)
    temp_c = (weather or {}).get("temperature_c", 20.0)
    temp_f = round(temp_c * 9 / 5 + 32, 1)

    # Temperature discomfort factor (Fahrenheit thresholds)
    temp_factor = 0.0
    if temp_f >= 90:
        temp_factor = min((temp_f - 90) / 20.0, 0.5)   # up to 0.5 at 110°F
    elif temp_f <= 20:
        temp_factor = min((20 - temp_f) / 30.0, 0.45)  # up to 0.45 at -10°F
    elif temp_f <= 32:
        temp_factor = (32 - temp_f) / 60.0              # mild cold boost

    # ── Classes RELEASING (students boarding NOW) ─────────────────────────────
    students_releasing = 0
    # ── Classes STARTING (students rushing to class in next 20 min) ───────────
    students_starting = 0

    if route_id:
        from sqlalchemy import text as sqtext
        result = await db.execute(sqtext("""
            SELECT DISTINCT s.stop_id FROM stops s
            JOIN stop_times st ON st.stop_id = s.stop_id
            JOIN trips t ON t.trip_id = st.trip_id
            WHERE t.route_id = :route_id LIMIT 20
        """), {"route_id": route_id})
        stop_ids = [r[0] for r in result.all()]

        for stop_id in stop_ids[:10]:
            students_releasing += await get_students_releasing(db, stop_id, now)

        # Classes starting: look for sections whose start_time is within next 20 min
        start_window_begin = now.time()
        start_window_end   = (now + timedelta(minutes=20)).time()
        from app.models.schedule import ClassSection, IUBuilding, StudentReleaseEvent
        from app.services.class_schedule import _haversine_m
        # Get building coords for stops on this route then sum enrollment
        for stop_id in stop_ids[:10]:
            stop_result = await db.execute(
                sqtext("SELECT stop_lat, stop_lon FROM stops WHERE stop_id=:sid"),
                {"sid": stop_id}
            )
            stop_row = stop_result.fetchone()
            if not stop_row:
                continue
            slat, slng = stop_row
            # Find in-person sections starting soon near this stop
            sect_result = await db.execute(
                sqtext("""
                    SELECT cs.enrollment, ib.latitude, ib.longitude
                    FROM class_sections cs
                    JOIN iu_buildings ib ON cs.building_code = ib.building_code
                    WHERE cs.is_in_person = 1
                      AND cs.start_time >= :t_start
                      AND cs.start_time <= :t_end
                      AND cs.enrollment > 0
                      AND ib.latitude IS NOT NULL
                """),
                {"t_start": start_window_begin.strftime("%H:%M:%S"),
                 "t_end":   start_window_end.strftime("%H:%M:%S")}
            )
            for enrl, blat, blng in sect_result.fetchall():
                if blat and blng:
                    dist = _haversine_m(slat, slng, blat, blng)
                    if dist <= 400:
                        students_starting += enrl

    releasing_norm = min(students_releasing / 800.0, 1.0)
    starting_norm  = min(students_starting  / 600.0, 1.0)

    # ── Overall occupancy score ────────────────────────────────────────────────
    # Weights: releasing > starting > weather > temperature
    occupancy_score = min(
        releasing_norm * 0.45 +
        starting_norm  * 0.25 +
        weather_severity * 0.20 +
        temp_factor * 0.10,
        1.0
    )
    delay_probability = min(occupancy_score * 1.3, 1.0)

    # Classify
    if occupancy_score < 0.20:
        level, label = "low",      "Seats available"
    elif occupancy_score < 0.50:
        level, label = "moderate", "Filling up"
    elif occupancy_score < 0.75:
        level, label = "high",     "Nearly full"
    else:
        level, label = "very_high","Standing room only"

    return {
        "vehicle_id": vehicle_id,
        "route_id": route_id,
        "occupancy_score": round(occupancy_score, 3),
        "occupancy_level": level,
        "occupancy_label": label,
        "delay_probability": round(delay_probability, 3),
        "factors": {
            "class_release": {
                "score": round(releasing_norm, 3),
                "students_releasing_nearby": students_releasing,
                "label": (
                    f"~{students_releasing} students just let out nearby"
                    if students_releasing > 0
                    else "No classes releasing soon"
                ),
                "active": releasing_norm > 0.08,
            },
            "class_starting": {
                "score": round(starting_norm, 3),
                "students_starting_nearby": students_starting,
                "label": (
                    f"~{students_starting} students heading to class soon"
                    if students_starting > 0
                    else "No classes starting soon"
                ),
                "active": starting_norm > 0.08,
            },
            "weather": {
                "score": round(weather_severity, 3),
                "label": _weather_label(weather),
                "is_raining": (weather or {}).get("is_raining", False),
                "is_snowing": (weather or {}).get("is_snowing", False),
                "active": weather_severity > 0.15,
            },
            "temperature": {
                "score": round(temp_factor, 3),
                "temp_f": temp_f,
                "temp_c": round(temp_c, 1),
                "label": _temp_label_f(temp_f),
                "active": temp_factor > 0.05,
            },
        },
        "computed_at": now.isoformat(),
        "data_source": "heuristic",
    }


@router.get("/heatmap")
async def get_crowding_heatmap(db: AsyncSession = Depends(get_db)):
    """
    Return stop-level crowding intensity for the heatmap layer.
    Each stop gets a weight (0-1) based on students releasing + weather.
    The frontend uses this to render a Mapbox heatmap layer.
    """
    from datetime import datetime as dt
    from sqlalchemy import text
    now = dt.utcnow()

    weather = get_current_weather()
    weather_severity = (weather or {}).get("weather_severity", 0.0)

    result = await db.execute(
        select(Stop).where(Stop.stop_lat.is_not(None))
    )
    stops = result.scalars().all()

    features = []
    for stop in stops:
        students = await get_students_releasing(db, stop.stop_id, now)
        students_norm = min(students / 300.0, 1.0)

        # Weight combines student release and weather
        weight = students_norm * 0.7 + weather_severity * 0.3
        if weight < 0.05:
            weight = 0.05  # baseline so all stops appear faintly

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [stop.stop_lon, stop.stop_lat],
            },
            "properties": {
                "stop_id": stop.stop_id,
                "stop_name": stop.stop_name,
                "weight": round(weight, 3),
                "students": students,
                "weather_factor": round(weather_severity, 3),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "computed_at": now.isoformat(),
        "weather_severity": weather_severity,
    }


def _weather_label(weather: dict | None) -> str:
    if not weather:
        return "No weather data"
    if weather.get("is_snowing"):
        return "Snowing — high bus demand"
    if weather.get("is_raining"):
        return f"Raining — {weather.get('precipitation_mm', 0):.1f}mm"
    if weather.get("weather_severity", 0) > 0:
        return "Adverse conditions"
    return "Clear weather"


def _temp_label(temp_c: float) -> str:
    temp_f = temp_c * 9 / 5 + 32
    return _temp_label_f(temp_f)


def _temp_label_f(temp_f: float) -> str:
    if temp_f >= 95:
        return f"{temp_f:.0f}°F — very hot, high ridership"
    if temp_f >= 85:
        return f"{temp_f:.0f}°F — hot, more students taking bus"
    if temp_f <= 10:
        return f"{temp_f:.0f}°F — extremely cold, high ridership"
    if temp_f <= 25:
        return f"{temp_f:.0f}°F — very cold, students avoiding walk"
    if temp_f <= 32:
        return f"{temp_f:.0f}°F — freezing, increased ridership"
    return f"{temp_f:.0f}°F — normal conditions"
