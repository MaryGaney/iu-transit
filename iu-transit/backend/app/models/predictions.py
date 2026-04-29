"""
app/models/predictions.py
─────────────────────────
ORM model for storing LSTM predictions and the feature vectors that generated them.
Storing predictions lets us:
  1. Compute prediction error as more data comes in (model evaluation)
  2. Serve cached predictions without running inference on every API request
  3. Build a labelled dataset for future retraining

Feature vector layout (9 features, order must match LSTM input):
  Index  Name                        Range / notes
  ─────  ──────────────────────────  ──────────────────────────────────────
  0      hour_sin                    sin(2π * hour / 24)
  1      hour_cos                    cos(2π * hour / 24)
  2      minute_sin                  sin(2π * minute / 60)
  3      minute_cos                  cos(2π * minute / 60)
  4      day_of_week_sin             sin(2π * dow / 7)
  5      day_of_week_cos             cos(2π * dow / 7)
  6      current_delay_seconds       z-scored; raw = seconds late (+ = late)
  7      weather_severity            0.0 – 1.0
  8      students_releasing_norm     normalised count of students releasing
                                     within 400m of this stop in next 15 min
"""

from datetime import datetime
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Index, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class DelayPrediction(Base):
    """
    One row per LSTM inference call.
    Created every poll cycle for each active (route, stop) pair.
    """
    __tablename__ = "delay_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    route_id: Mapped[str] = mapped_column(String(64), index=True)
    stop_id: Mapped[str] = mapped_column(String(64), index=True)
    trip_id: Mapped[str] = mapped_column(String(128), nullable=True)

    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Prediction output
    predicted_delay_seconds: Mapped[float] = mapped_column(Float)
    predicted_delay_lower: Mapped[float] = mapped_column(Float)   # 10th percentile
    predicted_delay_upper: Mapped[float] = mapped_column(Float)   # 90th percentile
    confidence: Mapped[float] = mapped_column(Float)              # 0-1

    # Scheduled ETA (from GTFS stop_times)
    scheduled_arrival: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Actual arrival (filled in retrospectively by the reconciler job)
    actual_arrival: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    actual_delay_seconds: Mapped[float] = mapped_column(Float, nullable=True)

    # Dominant reason for the prediction (for UI badge)
    # One of: "on_time", "class_release", "weather", "cascading", "unknown"
    delay_reason: Mapped[str] = mapped_column(String(32), default="unknown")

    # Serialised feature vector as JSON string (for debugging / retraining)
    feature_vector_json: Mapped[str] = mapped_column(Text, nullable=True)

    # Was the model loaded and used, or was this a fallback to schedule?
    model_used: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_pred_route_stop_ts", "route_id", "stop_id", "predicted_at"),
    )


class ModelTrainingRun(Base):
    """Tracks LSTM training runs for model versioning."""
    __tablename__ = "model_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    epochs: Mapped[int] = mapped_column(Integer)
    train_samples: Mapped[int] = mapped_column(Integer)
    val_mae_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    model_path: Mapped[str] = mapped_column(String(256), nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
