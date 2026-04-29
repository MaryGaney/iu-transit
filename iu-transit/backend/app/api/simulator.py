"""
app/api/simulator.py
─────────────────────
Bus position simulator for development/demo when no live GTFS-RT feed is available.

Loads real route shapes from the DB and moves fake buses along them,
broadcasting via the same WebSocket channel as real buses.

POST /api/simulator/start   — begin simulation
POST /api/simulator/stop    — stop simulation
GET  /api/simulator/status  — is simulation running?
"""

import asyncio
import math
import random
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db, AsyncSessionLocal
from app.core.websocket import broadcast_vehicle_positions
from app.core.logging import logger
from app.models.gtfs import Route, Shape, Trip

router = APIRouter(prefix="/api/simulator", tags=["simulator"])

# ── Simulator state ────────────────────────────────────────────────────────────
_sim_task: Optional[asyncio.Task] = None
_sim_running = False

# Fake bus state: { vehicle_id: { route_id, shape_points, progress (0-1), speed } }
_fake_buses: dict[str, dict] = {}


@router.post("/start")
async def start_simulation(
    bus_count: int = 8,
    db: AsyncSession = Depends(get_db),
):
    """Start the bus simulator. Creates `bus_count` fake buses on real routes."""
    global _sim_task, _sim_running

    if _sim_running:
        return {"status": "already_running", "bus_count": len(_fake_buses)}

    # Load routes and their shapes
    result = await db.execute(select(Route))
    routes = result.scalars().all()

    if not routes:
        return {"status": "error", "message": "No routes in DB. Load GTFS static first via /api/admin/load-gtfs-static"}

    # Build shape point lists per route
    route_shapes: dict[str, list] = {}
    for route in routes:
        # Get a shape_id for this route
        trip_result = await db.execute(
            select(Trip.shape_id)
            .where(Trip.route_id == route.route_id, Trip.shape_id.isnot(None))
            .limit(1)
        )
        shape_id = trip_result.scalar_one_or_none()
        if not shape_id:
            continue

        shape_result = await db.execute(
            select(Shape)
            .where(Shape.shape_id == shape_id)
            .order_by(Shape.shape_pt_sequence)
        )
        points = shape_result.scalars().all()
        if len(points) >= 2:
            route_shapes[route.route_id] = [
                {"lat": p.shape_pt_lat, "lng": p.shape_pt_lon}
                for p in points
            ]

    if not route_shapes:
        return {"status": "error", "message": "No route shapes found. GTFS static may not have shapes.txt"}

    # Assign buses to routes, spread across the route at different starting positions
    _fake_buses.clear()
    route_ids = list(route_shapes.keys())

    for i in range(bus_count):
        route_id = route_ids[i % len(route_ids)]
        vehicle_id = f"SIM-{route_id}-{i+1}"
        _fake_buses[vehicle_id] = {
            "route_id": route_id,
            "points": route_shapes[route_id],
            "progress": random.uniform(0, 1),   # start at random point on route
            "speed": random.uniform(0.0015, 0.003),  # progress units per tick
            "direction": 1,  # 1=forward, -1=reverse (buses loop)
        }

    _sim_running = True
    _sim_task = asyncio.create_task(_simulation_loop())
    logger.info(f"Simulator started: {len(_fake_buses)} buses on {len(route_shapes)} routes")

    return {
        "status": "started",
        "bus_count": len(_fake_buses),
        "routes": list(route_shapes.keys()),
    }


@router.post("/stop")
async def stop_simulation():
    """Stop the bus simulator."""
    global _sim_task, _sim_running
    _sim_running = False
    if _sim_task:
        _sim_task.cancel()
        _sim_task = None
    _fake_buses.clear()
    logger.info("Simulator stopped")
    return {"status": "stopped"}


@router.get("/status")
async def simulation_status():
    return {
        "running": _sim_running,
        "bus_count": len(_fake_buses),
        "buses": [
            {
                "vehicle_id": vid,
                "route_id": b["route_id"],
                "progress": round(b["progress"], 3),
            }
            for vid, b in _fake_buses.items()
        ],
    }


# ── Simulation loop ────────────────────────────────────────────────────────────

async def _simulation_loop():
    """
    Runs every 1 second, advances each bus along its route, broadcasts positions.
    Uses real shape points so buses follow actual road geometry.
    """
    global _sim_running

    while _sim_running:
        positions = []

        for vehicle_id, bus in _fake_buses.items():
            points = bus["points"]
            n = len(points)

            # Advance progress
            bus["progress"] += bus["speed"] * bus["direction"]

            # Bounce at ends (loop the route)
            if bus["progress"] >= 1.0:
                bus["progress"] = 1.0
                bus["direction"] = -1
            elif bus["progress"] <= 0.0:
                bus["progress"] = 0.0
                bus["direction"] = 1

            # Interpolate position along shape
            lat, lng, bearing = _interpolate_position(points, bus["progress"])

            positions.append({
                "vehicle_id": vehicle_id,
                "route_id": bus["route_id"],
                "trip_id": f"sim-trip-{vehicle_id}",
                "lat": lat,
                "lng": lng,
                "bearing": bearing,
                "speed": bus["speed"] * 5000,  # fake m/s for display
                "status": "IN_TRANSIT_TO",
                "timestamp": datetime.utcnow().isoformat(),
                "simulated": True,
            })

        if positions:
            await broadcast_vehicle_positions(positions)

        await asyncio.sleep(1.0)  # update every second for smooth animation


def _interpolate_position(
    points: list[dict], progress: float
) -> tuple[float, float, float]:
    """
    Linearly interpolate position along a list of lat/lng points.
    Returns (lat, lng, bearing_degrees).
    """
    if len(points) == 1:
        return points[0]["lat"], points[0]["lng"], 0.0

    # Map progress (0-1) to a segment index
    n_segments = len(points) - 1
    scaled = progress * n_segments
    seg_idx = min(int(scaled), n_segments - 1)
    seg_t = scaled - seg_idx  # 0-1 within this segment

    p1 = points[seg_idx]
    p2 = points[seg_idx + 1]

    lat = p1["lat"] + (p2["lat"] - p1["lat"]) * seg_t
    lng = p1["lng"] + (p2["lng"] - p1["lng"]) * seg_t
    bearing = _bearing(p1["lat"], p1["lng"], p2["lat"], p2["lng"])

    return lat, lng, bearing


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute compass bearing in degrees from point 1 to point 2."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
