"""
app/main.py
────────────
FastAPI application entry point.

Startup sequence:
  1. Init DB (create tables)
  2. Load GTFS static data (if not already in DB)
  3. Geocode buildings
  4. Fetch initial weather
  5. Load LSTM model (if exists)
  6. Start background scheduler
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.core.logging import setup_logging, logger
from app.core.scheduler import scheduler, setup_scheduler
from app.api import buses, predictions, admin, simulator, travel_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    setup_logging(settings.debug)
    logger.info("─── IU Transit Tracker starting up ───")

    # 1. Init database schema
    await init_db()
    logger.info("Database schema ready")

    async with AsyncSessionLocal() as db:
        # 2. Load GTFS static (skip if already loaded)
        try:
            from app.services.gtfs_static import load_gtfs_static
            counts = await load_gtfs_static(db)
            await db.commit()
            logger.info(f"GTFS static: {counts}")
        except Exception as e:
            logger.error(f"GTFS static load failed: {e}")

        # 3. Geocode buildings (idempotent)
        try:
            from app.services.geocoder import geocode_all_buildings
            bld_count = await geocode_all_buildings(db)
            await db.commit()
            logger.info(f"Buildings geocoded: {bld_count}")
        except Exception as e:
            logger.error(f"Building geocoding failed: {e}")

        # 4. Fetch initial weather
        try:
            from app.services.weather import fetch_weather
            weather = await fetch_weather(db)
            await db.commit()
            if weather:
                logger.info(f"Initial weather: {weather.get('weather_code')} severity={weather.get('weather_severity', 0):.2f}")
        except Exception as e:
            logger.warning(f"Initial weather fetch failed: {e}")

        # 5. Try to compute release events if schedule data exists
        try:
            from sqlalchemy import text
            r = await db.execute(text("SELECT COUNT(*) FROM class_sections"))
            section_count = r.scalar() or 0
            if section_count > 0:
                from app.services.class_schedule import compute_release_events
                event_count = await compute_release_events(db)
                await db.commit()
                logger.info(f"Student release events computed: {event_count}")
            else:
                logger.info("No class schedule loaded yet. Upload via POST /api/admin/load-schedule")
        except Exception as e:
            logger.warning(f"Release event computation failed: {e}")

    # 6. Load LSTM model
    from app.ml.lstm_model import predictor
    predictor.load()

    # 7. Start background scheduler
    setup_scheduler(app)
    scheduler.start()
    logger.info("Background scheduler started")
    logger.info(f"─── Server ready at http://{settings.host}:{settings.port} ───")

    yield  # Application runs here

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped. Goodbye.")


app = FastAPI(
    title="IU Transit Tracker",
    description=(
        "Real-time Bloomington Transit bus tracking with LSTM-based delay prediction "
        "accounting for IU class schedules and weather conditions."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow the React dev server and production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(buses.router)
app.include_router(predictions.router)
app.include_router(admin.router)
app.include_router(simulator.router)
app.include_router(travel_agent.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "iu-transit-tracker"}


@app.get("/api/config/public")
async def public_config():
    """Serve public config values to the frontend (Mapbox token etc.)."""
    return {
        "mapbox_token": settings.mapbox_token,
        "bloomington_lat": settings.bloomington_lat,
        "bloomington_lng": settings.bloomington_lng,
    }
