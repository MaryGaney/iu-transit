"""
app/core/websocket.py
──────────────────────
WebSocket connection manager.
All connected frontend clients receive bus position updates as JSON.

Message format sent to clients:
{
  "type": "vehicle_positions",
  "timestamp": "2024-11-01T14:23:45",
  "vehicles": [
    {
      "vehicle_id": "bus_42",
      "route_id": "4E",
      "trip_id": "...",
      "lat": 39.1686,
      "lng": -86.5225,
      "bearing": 180.0,
      "speed": 8.3,
      "status": "IN_TRANSIT_TO",
      "timestamp": "..."
    },
    ...
  ]
}
"""

import json
import asyncio
from datetime import datetime
from typing import Set
from fastapi import WebSocket

from app.core.logging import logger


class ConnectionManager:
    """Tracks all active WebSocket connections and broadcasts messages to them."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        logger.info(f"WS client connected. Total: {len(self._connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)
        logger.info(f"WS client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients. Dead connections are removed."""
        if not self._connections:
            return

        data = json.dumps(message)
        dead = set()

        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self._connections.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Module-level singleton
manager = ConnectionManager()


async def broadcast_vehicle_positions(vehicles: list[dict]) -> None:
    """Convenience wrapper to broadcast the standard vehicle positions message."""
    await manager.broadcast({
        "type": "vehicle_positions",
        "timestamp": datetime.utcnow().isoformat(),
        "vehicles": vehicles,
    })


async def broadcast_prediction_update(route_id: str, stop_id: str, prediction: dict) -> None:
    """Broadcast a new LSTM prediction to all clients."""
    await manager.broadcast({
        "type": "prediction_update",
        "timestamp": datetime.utcnow().isoformat(),
        "route_id": route_id,
        "stop_id": stop_id,
        "prediction": prediction,
    })
