"""
app/models/weather.py
─────────────────────
ORM model for weather observations logged every 5 minutes.
We use Open-Meteo (free, no key) for current conditions.

The weather features fed into the LSTM:
  - weather_code: WMO code (0=clear, 51-67=drizzle/rain, 71-77=snow, etc.)
  - temperature_c: current temp in Celsius
  - precipitation_mm: last-hour precipitation
  - wind_speed_ms: wind speed in m/s
  - is_precipitation: boolean derived from weather_code
"""

from datetime import datetime
from sqlalchemy import Integer, Float, Boolean, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


# WMO weather interpretation code → human label (subset relevant to delays)
WMO_CODE_LABELS = {
    0: "clear",
    1: "mainly_clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "fog",
    48: "icing_fog",
    51: "light_drizzle",
    53: "moderate_drizzle",
    55: "dense_drizzle",
    61: "slight_rain",
    63: "moderate_rain",
    65: "heavy_rain",
    71: "slight_snow",
    73: "moderate_snow",
    75: "heavy_snow",
    77: "snow_grains",
    80: "slight_showers",
    81: "moderate_showers",
    82: "violent_showers",
    85: "slight_snow_showers",
    86: "heavy_snow_showers",
    95: "thunderstorm",
    96: "thunderstorm_hail",
    99: "thunderstorm_heavy_hail",
}

PRECIPITATION_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82}
SNOW_CODES = {71, 73, 75, 77, 85, 86}
SEVERE_CODES = {95, 96, 99}


class WeatherObservation(Base):
    """
    One row per weather fetch (every 5 min).
    Joined to vehicle positions by nearest timestamp when building feature vectors.
    """
    __tablename__ = "weather_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    # Open-Meteo current_weather fields
    weather_code: Mapped[int] = mapped_column(Integer)
    temperature_c: Mapped[float] = mapped_column(Float)
    wind_speed_ms: Mapped[float] = mapped_column(Float, nullable=True)  # Open-Meteo gives km/h, we convert
    wind_direction: Mapped[float] = mapped_column(Float, nullable=True)

    # Hourly fields (look-ahead 1 hour for upcoming conditions)
    precipitation_mm: Mapped[float] = mapped_column(Float, default=0.0)
    precipitation_probability: Mapped[int] = mapped_column(Integer, default=0)  # 0-100

    # Derived boolean flags (for quick ML feature access)
    is_raining: Mapped[bool] = mapped_column(Boolean, default=False)
    is_snowing: Mapped[bool] = mapped_column(Boolean, default=False)
    is_severe: Mapped[bool] = mapped_column(Boolean, default=False)

    # Normalised 0-1 severity score (used as a single float feature in LSTM)
    # 0.0 = clear, 0.3 = light rain, 0.6 = heavy rain/snow, 1.0 = severe
    weather_severity: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index("ix_weather_at", "observed_at"),
    )

    @classmethod
    def compute_severity(cls, weather_code: int, precipitation_mm: float) -> float:
        if weather_code in SEVERE_CODES:
            return 1.0
        if weather_code in SNOW_CODES:
            return 0.7 + min(precipitation_mm / 20.0, 0.3)
        if weather_code in PRECIPITATION_CODES:
            return 0.3 + min(precipitation_mm / 10.0, 0.3)
        if weather_code in {45, 48}:  # fog
            return 0.2
        return 0.0
