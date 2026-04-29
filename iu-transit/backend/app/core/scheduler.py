"""
app/core/scheduler.py
──────────────────────
APScheduler configuration for all background polling jobs.

Job schedule:
  Every 15s  — poll GTFS-RT vehicle positions + trip updates
  Every 5min — fetch weather
  Every 1hr  — recompute student release events
  Every 4hrs — retrain LSTM if sufficient data exists
  Daily      — reload GTFS static feed (picks up schedule changes)
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.core.config import settings
from app.core.logging import logger

scheduler = AsyncIOScheduler(timezone="America/Indiana/Indianapolis")


def setup_scheduler(app) -> None:
    """Register all background jobs. Called from app startup."""

    @scheduler.scheduled_job(
        IntervalTrigger(seconds=settings.gtfs_rt_poll_interval),
        id="gtfs_rt_poll",
        max_instances=1,
        misfire_grace_time=5,
    )
    async def poll_realtime():
        """Poll GTFS-RT vehicle positions and trip updates."""
        from app.core.database import AsyncSessionLocal
        from app.services.gtfs_realtime import poll_vehicle_positions, poll_trip_updates

        async with AsyncSessionLocal() as db:
            try:
                vehicles = await poll_vehicle_positions(db)
                updates = await poll_trip_updates(db)
                await db.commit()
            except Exception as e:
                logger.error(f"GTFS-RT poll error: {e}")
                await db.rollback()

    @scheduler.scheduled_job(
        IntervalTrigger(seconds=settings.weather_poll_interval),
        id="weather_poll",
        max_instances=1,
        misfire_grace_time=30,
    )
    async def poll_weather():
        """Fetch current weather conditions."""
        from app.core.database import AsyncSessionLocal
        from app.services.weather import fetch_weather

        async with AsyncSessionLocal() as db:
            try:
                weather = await fetch_weather(db)
                await db.commit()
                if weather:
                    logger.debug(f"Weather updated: severity={weather.get('weather_severity', 0):.2f}")
            except Exception as e:
                logger.error(f"Weather poll error: {e}")
                await db.rollback()

    @scheduler.scheduled_job(
        IntervalTrigger(seconds=settings.class_schedule_refresh_interval),
        id="class_release_compute",
        max_instances=1,
        misfire_grace_time=120,
    )
    async def recompute_release_events():
        """Recompute student release events for all stops."""
        from app.core.database import AsyncSessionLocal
        from app.services.class_schedule import compute_release_events

        async with AsyncSessionLocal() as db:
            try:
                count = await compute_release_events(db)
                await db.commit()
                logger.info(f"Recomputed {count} student release events")
            except Exception as e:
                logger.error(f"Release event compute error: {e}")
                await db.rollback()

    @scheduler.scheduled_job(
        IntervalTrigger(hours=4),
        id="lstm_retrain",
        max_instances=1,
        misfire_grace_time=300,
    )
    async def retrain_lstm():
        """Retrain LSTM if sufficient data has accumulated."""
        from app.core.database import AsyncSessionLocal
        from app.ml.trainer import train_model

        async with AsyncSessionLocal() as db:
            try:
                path = await train_model(db)
                await db.commit()
                if path:
                    logger.info(f"LSTM retrain complete: {path}")
                else:
                    logger.info("LSTM retrain skipped (insufficient data)")
            except Exception as e:
                logger.error(f"LSTM retrain error: {e}")
                await db.rollback()

    @scheduler.scheduled_job(
        CronTrigger(hour=3, minute=0),  # 3 AM daily
        id="gtfs_static_reload",
        max_instances=1,
    )
    async def reload_gtfs_static():
        """Reload GTFS static feed nightly."""
        from app.core.database import AsyncSessionLocal
        from app.services.gtfs_static import load_gtfs_static

        async with AsyncSessionLocal() as db:
            try:
                counts = await load_gtfs_static(db, force=True)
                await db.commit()
                logger.info(f"Nightly GTFS static reload: {counts}")
            except Exception as e:
                logger.error(f"GTFS static reload error: {e}")
                await db.rollback()
