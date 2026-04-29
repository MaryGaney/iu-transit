"""
app/models/gtfs.py
──────────────────
ORM models for GTFS static data (routes, stops, trips, shapes)
and realtime vehicle positions.
"""

from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, DateTime, Boolean,
    ForeignKey, Index, Text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Route(Base):
    """GTFS routes.txt"""
    __tablename__ = "routes"

    route_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    route_short_name: Mapped[str] = mapped_column(String(32), nullable=True)
    route_long_name: Mapped[str] = mapped_column(String(256), nullable=True)
    route_color: Mapped[str] = mapped_column(String(8), nullable=True)
    route_text_color: Mapped[str] = mapped_column(String(8), nullable=True)
    route_type: Mapped[int] = mapped_column(Integer, default=3)  # 3 = bus

    trips: Mapped[list["Trip"]] = relationship("Trip", back_populates="route")


class Stop(Base):
    """GTFS stops.txt"""
    __tablename__ = "stops"

    stop_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stop_name: Mapped[str] = mapped_column(String(256))
    stop_lat: Mapped[float] = mapped_column(Float)
    stop_lon: Mapped[float] = mapped_column(Float)
    stop_desc: Mapped[str] = mapped_column(Text, nullable=True)
    wheelchair_boarding: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_stops_latlon", "stop_lat", "stop_lon"),
    )


class Trip(Base):
    """GTFS trips.txt"""
    __tablename__ = "trips"

    trip_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.route_id"))
    service_id: Mapped[str] = mapped_column(String(64))
    trip_headsign: Mapped[str] = mapped_column(String(256), nullable=True)
    direction_id: Mapped[int] = mapped_column(Integer, nullable=True)
    shape_id: Mapped[str] = mapped_column(String(64), nullable=True)

    route: Mapped["Route"] = relationship("Route", back_populates="trips")


class StopTime(Base):
    """GTFS stop_times.txt — scheduled arrival/departure per stop per trip."""
    __tablename__ = "stop_times"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.trip_id"), index=True)
    stop_id: Mapped[str] = mapped_column(ForeignKey("stops.stop_id"), index=True)
    arrival_time: Mapped[str] = mapped_column(String(8))    # HH:MM:SS (may exceed 24h)
    departure_time: Mapped[str] = mapped_column(String(8))
    stop_sequence: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_stoptimes_trip_stop", "trip_id", "stop_id"),
        Index("ix_stoptimes_stop_arrival", "stop_id", "arrival_time"),
    )


class Shape(Base):
    """GTFS shapes.txt — polyline points for map display."""
    __tablename__ = "shapes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shape_id: Mapped[str] = mapped_column(String(64), index=True)
    shape_pt_lat: Mapped[float] = mapped_column(Float)
    shape_pt_lon: Mapped[float] = mapped_column(Float)
    shape_pt_sequence: Mapped[int] = mapped_column(Integer)


class VehiclePosition(Base):
    """
    Realtime vehicle position — one row per poll per vehicle.
    This table is the core of our training data accumulation.
    Every 15-second poll writes here; we use it to compute historical delays.
    """
    __tablename__ = "vehicle_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[str] = mapped_column(String(64))
    trip_id: Mapped[str] = mapped_column(String(128), nullable=True)
    route_id: Mapped[str] = mapped_column(String(64), nullable=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    bearing: Mapped[float] = mapped_column(Float, nullable=True)
    speed: Mapped[float] = mapped_column(Float, nullable=True)   # m/s
    current_stop_sequence: Mapped[int] = mapped_column(Integer, nullable=True)
    current_status: Mapped[str] = mapped_column(String(32), nullable=True)
    # Seconds of delay vs schedule (positive = late, negative = early)
    schedule_delay_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    polled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_vp_route_ts", "route_id", "timestamp"),
        Index("ix_vp_vehicle_ts", "vehicle_id", "timestamp"),
    )


class TripUpdate(Base):
    """
    GTFS-RT TripUpdate stop-level delay data.
    More precise than vehicle position — gives delay per stop in seconds.
    """
    __tablename__ = "trip_updates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str] = mapped_column(String(128), index=True)
    route_id: Mapped[str] = mapped_column(String(64), index=True)
    stop_id: Mapped[str] = mapped_column(String(64), index=True)
    stop_sequence: Mapped[int] = mapped_column(Integer)
    arrival_delay_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    departure_delay_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    __table_args__ = (
        Index("ix_tu_route_stop_ts", "route_id", "stop_id", "timestamp"),
    )


class GTFSLoadLog(Base):
    """Tracks when the static GTFS feed was last loaded, for cache invalidation."""
    __tablename__ = "gtfs_load_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    route_count: Mapped[int] = mapped_column(Integer)
    stop_count: Mapped[int] = mapped_column(Integer)
    trip_count: Mapped[int] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
