"""
app/core/config.py
──────────────────
Centralised settings loaded from environment / .env file.
All other modules import `settings` from here — never read os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/transit.db"

    # ── GTFS ──────────────────────────────────────────────────────────────────
    gtfs_static_url: str = "https://bloomingtontransit.com/gtfs/google_transit.zip"
    # Vehicle positions protobuf endpoint.
    # NOTE: Bloomington Transit's exact RT URL must be confirmed by inspecting
    # the bloomingtontransit.com/gtfs page or contacting BT directly.
    # Common patterns for TransLoc / Swiftly-backed systems are listed below.
    gtfs_rt_vehicle_url: str = (
        "https://bloomingtontransit.com/gtfs/vehiclepositions.pb"
    )
    gtfs_rt_trip_updates_url: str = (
        "https://bloomingtontransit.com/gtfs/tripupdates.pb"
    )

    # ── Geography ─────────────────────────────────────────────────────────────
    bloomington_lat: float = 39.1653
    bloomington_lng: float = -86.5264

    # ── Weather (Open-Meteo — no key needed) ─────────────────────────────────
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"

    # ── IU Buildings ──────────────────────────────────────────────────────────
    iu_buildings_url: str = "https://api.iub.edu/buildings/v1"

    # ── Mapbox (served to frontend) ───────────────────────────────────────────
    mapbox_token: str = ""

    # ── ML ────────────────────────────────────────────────────────────────────
    lstm_model_path: str = "./data/lstm_model.pt"
    lstm_scaler_path: str = "./data/feature_scaler.pkl"
    lstm_sequence_length: int = 10   # timesteps fed into the LSTM
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_input_features: int = 9    # matches feature vector below

    # ── Polling intervals (seconds) ───────────────────────────────────────────
    gtfs_rt_poll_interval: int = 20
    weather_poll_interval: int = 300
    class_schedule_refresh_interval: int = 3600

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── Class schedule ────────────────────────────────────────────────────────
    # Radius (metres) around a stop to count as "near" for class-release signal
    class_release_radius_m: float = 400.0
    # Minutes before class ends to start boosting delay prediction
    class_release_lookahead_min: int = 15


settings = Settings()
