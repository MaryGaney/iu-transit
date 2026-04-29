"""
app/services/weather.py
────────────────────────
Fetches current weather for Bloomington, IN from Open-Meteo (free, no API key).

Open-Meteo API docs: https://open-meteo.com/en/docs
We request:
  - current_weather: temperature, windspeed, weathercode
  - hourly: precipitation, precipitation_probability (next 2 hours)

Called every 5 minutes by the scheduler. The most recent observation is
cached in memory and served directly to the feature builder without a DB query.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.models.weather import WeatherObservation, PRECIPITATION_CODES, SNOW_CODES, SEVERE_CODES
from sqlalchemy.ext.asyncio import AsyncSession


# In-memory cache of the most recent observation (avoids a DB round-trip per prediction)
_current_weather: Optional[dict] = None


async def fetch_weather(db: AsyncSession) -> Optional[dict]:
    """
    Fetch current weather from Open-Meteo and persist to DB.
    Returns a dict with the weather features or None on failure.
    """
    global _current_weather

    params = {
        "latitude": settings.bloomington_lat,
        "longitude": settings.bloomington_lng,
        "current_weather": "true",
        "hourly": "precipitation,precipitation_probability",
        "timezone": "America/Indiana/Indianapolis",
        "forecast_days": 1,
        "windspeed_unit": "ms",  # get m/s directly
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(settings.weather_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return _current_weather  # return stale data rather than None

    try:
        cw = data["current_weather"]
        weather_code = int(cw["weathercode"])
        temperature_c = float(cw["temperature"])
        wind_speed_ms = float(cw["windspeed"])  # already m/s per our request
        wind_direction = float(cw.get("winddirection", 0))

        # Extract the current hour's precipitation from hourly data
        now_hour = datetime.now().hour
        precip_mm = 0.0
        precip_prob = 0
        if "hourly" in data:
            hourly = data["hourly"]
            times = hourly.get("time", [])
            # Find index matching current hour
            for i, t in enumerate(times):
                if f"T{now_hour:02d}:00" in t:
                    precip_mm = float(hourly["precipitation"][i] or 0)
                    precip_prob = int(hourly["precipitation_probability"][i] or 0)
                    break

        is_raining = weather_code in PRECIPITATION_CODES
        is_snowing = weather_code in SNOW_CODES
        is_severe = weather_code in SEVERE_CODES

        severity = WeatherObservation.compute_severity(weather_code, precip_mm)

        obs = WeatherObservation(
            observed_at=datetime.utcnow(),
            weather_code=weather_code,
            temperature_c=temperature_c,
            wind_speed_ms=wind_speed_ms,
            wind_direction=wind_direction,
            precipitation_mm=precip_mm,
            precipitation_probability=precip_prob,
            is_raining=is_raining,
            is_snowing=is_snowing,
            is_severe=is_severe,
            weather_severity=severity,
        )
        db.add(obs)
        await db.flush()

        weather_dict = {
            "weather_code": weather_code,
            "temperature_c": temperature_c,
            "wind_speed_ms": wind_speed_ms,
            "precipitation_mm": precip_mm,
            "precipitation_probability": precip_prob,
            "is_raining": is_raining,
            "is_snowing": is_snowing,
            "is_severe": is_severe,
            "weather_severity": severity,
            "observed_at": obs.observed_at.isoformat(),
        }
        _current_weather = weather_dict
        logger.debug(
            f"Weather: code={weather_code}, temp={temperature_c}°C, "
            f"precip={precip_mm}mm, severity={severity:.2f}"
        )
        return weather_dict

    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Weather parse error: {e} | data={data}")
        return _current_weather


def get_current_weather() -> Optional[dict]:
    """Return the cached weather observation without hitting the DB."""
    return _current_weather


def get_weather_feature() -> float:
    """
    Return just the severity float (0.0–1.0) for the LSTM feature vector.
    Returns 0.0 (clear) if no weather data available yet.
    """
    if _current_weather:
        return _current_weather.get("weather_severity", 0.0)
    return 0.0
