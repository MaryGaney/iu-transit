"""
app/ml/feature_builder.py
──────────────────────────
Assembles the LSTM feature sequence for a given (route, stop) pair.
Safe to import without PyTorch — lstm_model.py has TORCH_AVAILABLE guards.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.core.config import settings
from app.core.logging import logger
from app.models.gtfs import TripUpdate
from app.services.weather import get_weather_feature
from app.services.class_schedule import get_students_releasing
from app.ml.lstm_model import build_feature_vector, predictor, TORCH_AVAILABLE


async def build_sequence_for_stop(
    db: AsyncSession,
    route_id: str,
    stop_id: str,
    now: Optional[datetime] = None,
) -> list[list[float]]:
    """
    Build a feature sequence for LSTM inference on a specific route/stop.
    Returns seq_len feature vectors (most recent last).
    """
    if now is None:
        now = datetime.utcnow()

    seq_len = settings.lstm_sequence_length

    result = await db.execute(
        select(TripUpdate)
        .where(
            TripUpdate.route_id == route_id,
            TripUpdate.stop_id == stop_id,
            TripUpdate.arrival_delay_seconds.is_not(None),
        )
        .order_by(desc(TripUpdate.timestamp))
        .limit(seq_len)
    )
    records = list(reversed(result.scalars().all()))

    weather_severity = get_weather_feature()
    students_count   = await get_students_releasing(db, stop_id, now)
    students_norm    = min(students_count / predictor.max_students, 1.0)

    if not records:
        fv = build_feature_vector(
            hour=now.hour,
            minute=now.minute,
            day_of_week=now.weekday(),
            current_delay_seconds=0.0,
            weather_severity=weather_severity,
            students_releasing_norm=students_norm,
            delay_mean=predictor.delay_mean,
            delay_std=predictor.delay_std,
        )
        return [fv] * seq_len

    sequence = []
    for record in records:
        ts = record.timestamp
        fv = build_feature_vector(
            hour=ts.hour,
            minute=ts.minute,
            day_of_week=ts.weekday(),
            current_delay_seconds=record.arrival_delay_seconds,
            weather_severity=weather_severity,
            students_releasing_norm=students_norm,
            delay_mean=predictor.delay_mean,
            delay_std=predictor.delay_std,
        )
        sequence.append(fv)

    while len(sequence) < seq_len:
        sequence = [sequence[0]] + sequence

    return sequence[-seq_len:]


async def run_inference_for_stop(
    db: AsyncSession,
    route_id: str,
    stop_id: str,
    trip_id: Optional[str] = None,
    scheduled_arrival: Optional[datetime] = None,
) -> dict:
    """Full pipeline: build features → run predictor → persist prediction."""
    from app.models.predictions import DelayPrediction
    import json

    now = datetime.utcnow()

    sequence         = await build_sequence_for_stop(db, route_id, stop_id, now)
    weather_severity = get_weather_feature()
    students_count   = await get_students_releasing(db, stop_id, now)

    result = predictor.predict(
        sequence=sequence,
        students_releasing=students_count,
        weather_severity=weather_severity,
    )

    # Persist prediction — best effort, non-fatal
    try:
        prediction = DelayPrediction(
            route_id=route_id,
            stop_id=stop_id,
            trip_id=trip_id,
            predicted_at=now,
            predicted_delay_seconds=result["predicted_delay_seconds"],
            predicted_delay_lower=result["predicted_delay_lower"],
            predicted_delay_upper=result["predicted_delay_upper"],
            confidence=result["confidence"],
            scheduled_arrival=scheduled_arrival,
            delay_reason=result["delay_reason"],
            feature_vector_json=json.dumps(sequence[-1]),
            model_used=result["model_used"],
        )
        db.add(prediction)
        await db.flush()
    except Exception as e:
        logger.warning(f"Could not persist prediction (non-fatal): {e}")

    return result
