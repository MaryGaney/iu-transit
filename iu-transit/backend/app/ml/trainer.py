"""
app/ml/trainer.py
──────────────────
Builds training data from the accumulated DB logs and trains the LSTM.

Training data construction:
  For each (route_id, stop_id) pair with sufficient history:
    1. Query trip_updates ordered by timestamp
    2. Build sliding windows of seq_len timesteps
    3. Join each timestep with weather and student release features
    4. Target = arrival_delay_seconds at the NEXT timestep

Minimum data requirement: 100 samples (roughly 25 minutes of data per route).
Re-training is triggered by the scheduler every 4 hours once baseline data exists.
"""

import os
import json
import asyncio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

from app.core.config import settings
from app.core.logging import logger
from app.models.gtfs import TripUpdate, VehiclePosition
from app.models.weather import WeatherObservation
from app.models.schedule import StudentReleaseEvent
from app.models.predictions import ModelTrainingRun
from app.ml.lstm_model import DelayLSTM, DelayDataset, build_feature_vector, predictor


MIN_SAMPLES = 100       # don't train on fewer than this
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.85


async def build_training_dataset(db: AsyncSession) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Query the DB and construct (X, y) arrays for LSTM training.

    Returns:
        X: (N, seq_len, 9) float32
        y: (N, 1) float32
        meta: dict with normalisation stats
    """
    seq_len = settings.lstm_sequence_length
    logger.info("Building training dataset from DB...")

    # Pull all trip_update records ordered by route/stop/time
    result = await db.execute(
        text("""
            SELECT tu.route_id, tu.stop_id, tu.arrival_delay_seconds,
                   tu.timestamp,
                   COALESCE(wo.weather_severity, 0.0) as weather_severity,
                   COALESCE(sre.students_releasing, 0) as students_releasing
            FROM trip_updates tu
            LEFT JOIN weather_observations wo ON (
                wo.id = (
                    SELECT id FROM weather_observations
                    WHERE observed_at <= tu.timestamp
                    ORDER BY observed_at DESC LIMIT 1
                )
            )
            LEFT JOIN student_release_events sre ON (
                sre.stop_id = tu.stop_id
                AND sre.day_of_week = CAST(strftime('%w', tu.timestamp) AS INT)
                AND time(sre.window_start) <= time(tu.timestamp)
                AND time(sre.window_end) > time(tu.timestamp)
            )
            WHERE tu.arrival_delay_seconds IS NOT NULL
            ORDER BY tu.route_id, tu.stop_id, tu.timestamp
        """)
    )
    rows = result.fetchall()

    if len(rows) < MIN_SAMPLES + seq_len:
        logger.warning(f"Only {len(rows)} training rows — need {MIN_SAMPLES + seq_len}. Skipping training.")
        return None, None, {}

    # Compute normalisation stats
    delays = [r.arrival_delay_seconds for r in rows]
    delay_mean = float(np.mean(delays))
    delay_std = float(np.std(delays)) or 60.0
    max_students = max(r.students_releasing for r in rows) or 500.0

    logger.info(f"Training data: {len(rows)} rows | delay mean={delay_mean:.1f}s std={delay_std:.1f}s")

    # Build sliding windows
    sequences = []
    targets = []

    # Group by route+stop to build meaningful sequences
    from itertools import groupby
    key_fn = lambda r: (r.route_id, r.stop_id)
    sorted_rows = sorted(rows, key=key_fn)

    for (route_id, stop_id), group in groupby(sorted_rows, key=key_fn):
        group_rows = list(group)
        if len(group_rows) < seq_len + 1:
            continue

        for i in range(len(group_rows) - seq_len):
            window = group_rows[i:i + seq_len]
            target_row = group_rows[i + seq_len]

            feature_seq = []
            for r in window:
                ts = r.timestamp
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                fv = build_feature_vector(
                    hour=ts.hour,
                    minute=ts.minute,
                    day_of_week=ts.weekday(),
                    current_delay_seconds=r.arrival_delay_seconds,
                    weather_severity=r.weather_severity,
                    students_releasing_norm=min(r.students_releasing / max_students, 1.0),
                    delay_mean=delay_mean,
                    delay_std=delay_std,
                )
                feature_seq.append(fv)

            # Target: next delay, normalised
            target_delay_norm = (target_row.arrival_delay_seconds - delay_mean) / delay_std
            sequences.append(feature_seq)
            targets.append([target_delay_norm])

    if len(sequences) < MIN_SAMPLES:
        logger.warning(f"Only {len(sequences)} sequences after windowing — need {MIN_SAMPLES}.")
        return None, None, {}

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    meta = {
        "delay_mean": delay_mean,
        "delay_std": delay_std,
        "max_students": max_students,
        "n_samples": len(sequences),
    }

    logger.info(f"Dataset: X={X.shape}, y={y.shape}")
    return X, y, meta


async def train_model(db: AsyncSession) -> Optional[str]:
    """
    Full training run. Saves model to disk and updates the predictor singleton.
    Returns path to saved model, or None if training was skipped.
    """
    run = ModelTrainingRun(started_at=datetime.utcnow(), epochs=EPOCHS, train_samples=0)
    db.add(run)
    await db.flush()

    X, y, meta = await build_training_dataset(db)
    if X is None:
        run.notes = "Insufficient data"
        await db.flush()
        return None

    run.train_samples = meta["n_samples"]

    # Dataset split
    dataset = DelayDataset(X, y)
    n_train = int(len(dataset) * TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = DelayLSTM(
        input_size=settings.lstm_input_features,
        hidden_size=settings.lstm_hidden_size,
        num_layers=settings.lstm_num_layers,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None

    logger.info(f"Training LSTM: {n_train} train / {n_val} val samples, {EPOCHS} epochs")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            mean, variance = model(xb)
            # Gaussian negative log-likelihood loss (trains both mean and variance)
            loss = gaussian_nll_loss(mean, yb, variance)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                mean, variance = model(xb)
                val_loss += gaussian_nll_loss(mean, yb, variance).item()
                # MAE in original seconds
                mae_norm = torch.abs(mean - yb).mean().item()
                val_mae += mae_norm * meta["delay_std"]

        val_mae /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | val_loss={val_loss:.4f} | val_MAE={val_mae:.1f}s")

    # Save best model
    os.makedirs("./data", exist_ok=True)
    model_path = settings.lstm_model_path
    checkpoint = {
        "model_state_dict": best_state,
        "delay_mean": meta["delay_mean"],
        "delay_std": meta["delay_std"],
        "max_students": meta["max_students"],
        "trained_at": datetime.utcnow().isoformat(),
        "val_mae_seconds": val_mae,
    }
    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path} | val MAE={val_mae:.1f}s")

    # Reload predictor singleton
    predictor.load()

    run.finished_at = datetime.utcnow()
    run.val_mae_seconds = val_mae
    run.model_path = model_path
    await db.flush()

    return model_path


def gaussian_nll_loss(
    mean: torch.Tensor, target: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    Trains the model to predict both the mean AND its uncertainty.
    Better than MSE because the model learns to be confident when it should be
    and uncertain when it shouldn't (e.g. class-release days it hasn't seen).
    """
    return torch.mean(
        0.5 * torch.log(variance) + 0.5 * ((target - mean) ** 2) / variance
    )
