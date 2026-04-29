"""
app/api/predictions.py
───────────────────────
Endpoints for LSTM-based delay predictions.

GET /api/predictions/stop/{stop_id}          — prediction for all routes at a stop
GET /api/predictions/route/{route_id}        — predictions for all stops on a route
GET /api/predictions/stop/{stop_id}/explain  — full explanation with features
GET /api/predictions/status                  — model status (loaded / fallback)
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.predictions import DelayPrediction
from app.models.gtfs import TripUpdate, Stop, Route
from app.ml.feature_builder import run_inference_for_stop
from app.ml.lstm_model import predictor
from app.services.weather import get_current_weather
from app.services.class_schedule import get_students_releasing
from app.core.logging import logger

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/status")
async def get_model_status():
    """Returns whether the LSTM model is loaded or we're using the heuristic fallback."""
    weather = get_current_weather()
    return {
        "model_loaded": predictor.is_loaded,
        "mode": "lstm" if predictor.is_loaded else "heuristic_fallback",
        "delay_mean_seconds": round(predictor.delay_mean, 1),
        "delay_std_seconds": round(predictor.delay_std, 1),
        "weather": weather,
        "note": (
            "LSTM model active — predictions improving with each data point."
            if predictor.is_loaded
            else "Collecting training data. Using weather + class-release heuristic until ~100 samples."
        ),
    }


@router.get("/stop/{stop_id}")
async def predict_for_stop(
    stop_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Run LSTM inference for all active routes serving this stop.
    Returns one prediction per route currently operating at the stop.
    """
    now = datetime.utcnow()

    # Find routes with recent activity at this stop (last 2 hours)
    since = now - timedelta(hours=2)
    result = await db.execute(
        select(TripUpdate.route_id)
        .where(
            TripUpdate.stop_id == stop_id,
            TripUpdate.timestamp >= since,
        )
        .distinct()
    )
    route_ids = [r for (r,) in result.all()]

    if not route_ids:
        # No recent data — return a generic prediction
        students = await get_students_releasing(db, stop_id, now)
        weather = get_current_weather()
        from app.ml.lstm_model import predictor as p
        result_dict = p.predict(
            sequence=[],
            students_releasing=students,
            weather_severity=(weather or {}).get("weather_severity", 0.0),
        )
        return {
            "stop_id": stop_id,
            "predicted_at": now.isoformat(),
            "predictions": [{"route_id": "ALL", **result_dict}],
            "note": "No recent route activity at this stop.",
        }

    predictions = []
    for route_id in route_ids[:5]:  # cap at 5 routes for performance
        pred = await run_inference_for_stop(db, route_id, stop_id)
        predictions.append({"route_id": route_id, **pred})

    return {
        "stop_id": stop_id,
        "predicted_at": now.isoformat(),
        "predictions": predictions,
    }


@router.get("/route/{route_id}")
async def predict_for_route(
    route_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Run inference for all stops currently showing delay on a route.
    Returns a list of (stop_id, prediction) sorted by predicted delay descending.
    """
    now = datetime.utcnow()
    since = now - timedelta(hours=1)

    # Find stops on this route with recent updates
    result = await db.execute(
        select(TripUpdate.stop_id)
        .where(
            TripUpdate.route_id == route_id,
            TripUpdate.timestamp >= since,
        )
        .distinct()
        .limit(20)
    )
    stop_ids = [r for (r,) in result.all()]

    predictions = []
    for stop_id in stop_ids:
        pred = await run_inference_for_stop(db, route_id, stop_id)
        predictions.append({
            "stop_id": stop_id,
            **pred,
        })

    # Sort by predicted delay descending so worst delays are shown first
    predictions.sort(key=lambda x: x["predicted_delay_seconds"], reverse=True)

    return {
        "route_id": route_id,
        "predicted_at": now.isoformat(),
        "stop_predictions": predictions,
    }


@router.get("/stop/{stop_id}/explain")
async def explain_prediction(
    stop_id: str,
    route_id: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Detailed explanation of what is driving the prediction at this stop.
    Returns the raw feature values alongside the prediction for debugging / UI.
    """
    now = datetime.utcnow()
    weather = get_current_weather()
    students = await get_students_releasing(db, stop_id, now)

    # Get most recent delay at this stop
    q = select(TripUpdate).where(
        TripUpdate.stop_id == stop_id,
        TripUpdate.arrival_delay_seconds.is_not(None),
    ).order_by(desc(TripUpdate.timestamp)).limit(1)
    if route_id:
        q = q.where(TripUpdate.route_id == route_id)
    result = await db.execute(q)
    latest = result.scalar_one_or_none()

    current_delay = latest.arrival_delay_seconds if latest else 0.0

    return {
        "stop_id": stop_id,
        "route_id": route_id,
        "explained_at": now.isoformat(),
        "signals": {
            "current_delay_seconds": current_delay,
            "time": {
                "hour": now.hour,
                "minute": now.minute,
                "day_of_week": now.weekday(),
                "day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][now.weekday()],
            },
            "weather": weather or {"weather_severity": 0, "is_raining": False, "is_snowing": False},
            "students_releasing_soon": {
                "count": students,
                "normalised": round(min(students / predictor.max_students, 1.0), 3),
                "radius_m": 400,
                "lookahead_min": 15,
            },
        },
        "model_status": {
            "loaded": predictor.is_loaded,
            "mode": "lstm" if predictor.is_loaded else "heuristic",
        },
    }


@router.get("/history/{stop_id}")
async def get_prediction_history(
    stop_id: str,
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
):
    """Historical predictions vs actuals for a stop (for model evaluation UI)."""
    since = datetime.utcnow() - timedelta(hours=hours)
    result = await db.execute(
        select(DelayPrediction)
        .where(
            DelayPrediction.stop_id == stop_id,
            DelayPrediction.predicted_at >= since,
        )
        .order_by(desc(DelayPrediction.predicted_at))
        .limit(100)
    )
    preds = result.scalars().all()

    return {
        "stop_id": stop_id,
        "predictions": [
            {
                "predicted_at": p.predicted_at.isoformat(),
                "route_id": p.route_id,
                "predicted_delay_seconds": p.predicted_delay_seconds,
                "actual_delay_seconds": p.actual_delay_seconds,
                "confidence": p.confidence,
                "delay_reason": p.delay_reason,
                "model_used": p.model_used,
            }
            for p in preds
        ],
    }
