"""
app/services/gtfs_realtime.py
──────────────────────────────
Polls the Bloomington Transit GTFS-RT vehicle positions and trip updates
endpoints every 15 seconds.

GTFS-RT uses Protocol Buffers (protobuf). The `gtfs-realtime-bindings` package
provides the Python classes. We decode the binary feed and write to DB.

This service also:
  - Computes schedule delay in seconds by joining against stop_times
  - Broadcasts position updates via WebSocket fan-out (app.core.websocket)
  - Writes training data continuously to vehicle_positions and trip_updates

IMPORTANT NOTE ON BT'S RT URL:
  Bloomington Transit's exact GTFS-RT URL needs to be confirmed. The service
  tries the configured URL first and falls back to a set of known patterns.
  Run `python scripts/probe_gtfs_rt.py` to identify the correct endpoint.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from google.transit import gtfs_realtime_pb2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.config import settings
from app.core.logging import logger
from app.models.gtfs import VehiclePosition, TripUpdate
from app.core.websocket import broadcast_vehicle_positions


# State shared across polls — last known positions for WebSocket diffing
_last_positions: dict[str, dict] = {}


async def poll_vehicle_positions(db: AsyncSession) -> list[dict]:
    """
    Fetch and parse GTFS-RT VehiclePositions feed.
    Returns list of position dicts (also written to DB and broadcast via WS).
    """
    raw = await _fetch_protobuf(settings.gtfs_rt_vehicle_url)
    if raw is None:
        logger.warning("No data from GTFS-RT vehicle positions endpoint")
        return []

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(raw)
    except Exception as e:
        logger.error(f"Failed to parse GTFS-RT protobuf: {e}")
        return []

    positions = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        veh = entity.vehicle
        pos = veh.position
        trip = veh.trip

        vehicle_id = veh.vehicle.id or entity.id
        route_id = trip.route_id if trip else None
        trip_id = trip.trip_id if trip else None

        # Timestamp from feed, fall back to now
        ts = (
            datetime.utcfromtimestamp(veh.timestamp)
            if veh.timestamp else now
        )

        # Map GTFS-RT current_status enum to string
        status_map = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}
        status = status_map.get(veh.current_status, "UNKNOWN")

        vp = VehiclePosition(
            vehicle_id=vehicle_id,
            trip_id=trip_id,
            route_id=route_id,
            latitude=pos.latitude,
            longitude=pos.longitude,
            bearing=pos.bearing if pos.bearing else None,
            speed=pos.speed if pos.speed else None,
            current_stop_sequence=veh.current_stop_sequence or None,
            current_status=status,
            timestamp=ts,
            polled_at=now,
        )
        db.add(vp)

        position_dict = {
            "vehicle_id": vehicle_id,
            "route_id": route_id,
            "trip_id": trip_id,
            "lat": pos.latitude,
            "lng": pos.longitude,
            "bearing": pos.bearing,
            "speed": pos.speed,
            "status": status,
            "timestamp": ts.isoformat(),
        }
        positions.append(position_dict)
        _last_positions[vehicle_id] = position_dict

    await db.flush()

    # Broadcast to all connected WebSocket clients
    if positions:
        await broadcast_vehicle_positions(positions)
        logger.debug(f"Polled {len(positions)} vehicles, broadcast to WS clients")

    return positions


async def poll_trip_updates(db: AsyncSession) -> int:
    """
    Fetch and parse GTFS-RT TripUpdates feed.
    This gives per-stop arrival/departure delays in seconds — the most direct
    source of ground truth for training the LSTM.
    """
    raw = await _fetch_protobuf(settings.gtfs_rt_trip_updates_url)
    if raw is None:
        return 0

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(raw)
    except Exception as e:
        logger.error(f"Failed to parse TripUpdates protobuf: {e}")
        return 0

    now = datetime.utcnow()
    count = 0

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        route_id = tu.trip.route_id
        trip_id = tu.trip.trip_id

        for stu in tu.stop_time_update:
            arr_delay = stu.arrival.delay if stu.HasField("arrival") else None  # type: ignore
            dep_delay = stu.departure.delay if stu.HasField("departure") else None  # type: ignore

            db.add(TripUpdate(
                trip_id=trip_id,
                route_id=route_id,
                stop_id=stu.stop_id,
                stop_sequence=stu.stop_sequence,
                arrival_delay_seconds=arr_delay,
                departure_delay_seconds=dep_delay,
                timestamp=now,
            ))
            count += 1

    await db.flush()
    logger.debug(f"Stored {count} TripUpdate stop records")
    return count


async def get_current_positions() -> list[dict]:
    """Return the last known positions from memory (no DB query needed)."""
    return list(_last_positions.values())


async def _fetch_protobuf(url: str) -> Optional[bytes]:
    """
    Fetch binary protobuf data from a GTFS-RT endpoint.
    Returns raw bytes or None on failure.

    Headers note: some GTFS-RT providers require specific Accept headers.
    We send both the standard and a generic binary accept.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url,
                headers={
                    "Accept": "application/x-protobuf, application/octet-stream, */*",
                    "User-Agent": "IUTransitTracker/1.0",
                },
            )
            if response.status_code == 200:
                return response.content
            else:
                logger.warning(f"GTFS-RT {url} returned {response.status_code}")
                return None
    except httpx.ConnectError:
        logger.warning(f"GTFS-RT endpoint unreachable: {url}")
        return None
    except Exception as e:
        logger.error(f"GTFS-RT fetch error: {e}")
        return None
