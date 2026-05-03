"""
app/ml/trainer.py
──────────────────
Builds training data from the accumulated DB logs and trains the LSTM.

PyTorch is optional — if not installed, train_model() returns None immediately
with a clear log message. Install torch and redeploy to enable training.
"""

import os
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional
from itertools import groupby

# ── Optional PyTorch import ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    random_split = None

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import logger
from app.models.predictions import ModelTrainingRun

# Import lazily from lstm_model to avoid circular issues
# (lstm_model also has the TORCH_AVAILABLE guard)
from app.ml.lstm_model import (
    DelayLSTM, DelayDataset, build_feature_vector,
    predictor, TORCH_AVAILABLE as LSTM_TORCH_AVAILABLE
)


MIN_SAMPLES    = 100
EPOCHS         = 40
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
TRAIN_SPLIT    = 0.85


async def build_training_dataset(db: AsyncSession) -> tuple:
    """
    Query the DB and construct (X, y) arrays for LSTM training.
    Returns (None, None, {}) if insufficient data or torch unavailable.
    """
    if not TORCH_AVAILABLE:
        logger.info("PyTorch not available — skipping dataset build.")
        return None, None, {}

    seq_len = settings.lstm_sequence_length
    logger.info("Building training dataset from DB...")

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
        logger.warning(
            f"Only {len(rows)} training rows — need {MIN_SAMPLES + seq_len}. "
            "Skipping training."
        )
        return None, None, {}

    delays      = [r.arrival_delay_seconds for r in rows]
    delay_mean  = float(np.mean(delays))
    delay_std   = float(np.std(delays)) or 60.0
    max_students = max(r.students_releasing for r in rows) or 500.0

    logger.info(
        f"Training data: {len(rows)} rows | "
        f"delay mean={delay_mean:.1f}s std={delay_std:.1f}s"
    )

    sequences = []
    targets   = []

    key_fn      = lambda r: (r.route_id, r.stop_id)
    sorted_rows = sorted(rows, key=key_fn)

    for (route_id, stop_id), group in groupby(sorted_rows, key=key_fn):
        group_rows = list(group)
        if len(group_rows) < seq_len + 1:
            continue

        for i in range(len(group_rows) - seq_len):
            window      = group_rows[i:i + seq_len]
            target_row  = group_rows[i + seq_len]

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
                    students_releasing_norm=min(
                        r.students_releasing / max_students, 1.0
                    ),
                    delay_mean=delay_mean,
                    delay_std=delay_std,
                )
                feature_seq.append(fv)

            target_delay_norm = (
                target_row.arrival_delay_seconds - delay_mean
            ) / delay_std
            sequences.append(feature_seq)
            targets.append([target_delay_norm])

    if len(sequences) < MIN_SAMPLES:
        logger.warning(
            f"Only {len(sequences)} sequences after windowing — need {MIN_SAMPLES}."
        )
        return None, None, {}

    X    = np.array(sequences, dtype=np.float32)
    y    = np.array(targets,   dtype=np.float32)
    meta = {
        "delay_mean":  delay_mean,
        "delay_std":   delay_std,
        "max_students": max_students,
        "n_samples":   len(sequences),
    }

    logger.info(f"Dataset ready: X={X.shape}, y={y.shape}")
    return X, y, meta


async def train_model(db: AsyncSession) -> Optional[str]:
    """
    Full training run.
    Returns path to saved model, or None if training was skipped.
    """
    if not TORCH_AVAILABLE:
        logger.info(
            "PyTorch not installed — cannot train LSTM. "
            "Add torch to requirements.txt and redeploy to enable training."
        )
        return None

    run = ModelTrainingRun(
        started_at=datetime.utcnow(),
        epochs=EPOCHS,
        train_samples=0,
    )
    db.add(run)
    await db.flush()

    X, y, meta = await build_training_dataset(db)
    if X is None:
        run.notes = "Insufficient data"
        await db.flush()
        return None

    run.train_samples = meta["n_samples"]

    dataset  = DelayDataset(X, y)
    n_train  = int(len(dataset) * TRAIN_SPLIT)
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

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
    best_state    = None
    val_mae       = 0.0

    logger.info(
        f"Training LSTM: {n_train} train / {n_val} val samples, "
        f"{EPOCHS} epochs"
    )

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            mean, variance = model(xb)
            loss = _gaussian_nll_loss(mean, yb, variance)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                mean, variance = model(xb)
                val_loss += _gaussian_nll_loss(mean, yb, variance).item()
                val_mae  += torch.abs(mean - yb).mean().item() * meta["delay_std"]

        val_mae /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"val_loss={val_loss:.4f} | val_MAE={val_mae:.1f}s"
            )

    os.makedirs("./data", exist_ok=True)
    model_path = settings.lstm_model_path
    torch.save(
        {
            "model_state_dict": best_state,
            "delay_mean":        meta["delay_mean"],
            "delay_std":         meta["delay_std"],
            "max_students":      meta["max_students"],
            "trained_at":        datetime.utcnow().isoformat(),
            "val_mae_seconds":   val_mae,
        },
        model_path,
    )
    logger.info(f"Model saved to {model_path} | val MAE={val_mae:.1f}s")

    predictor.load()

    run.finished_at      = datetime.utcnow()
    run.val_mae_seconds  = val_mae
    run.model_path       = model_path
    await db.flush()

    return model_path


def _gaussian_nll_loss(
    mean: "torch.Tensor",
    target: "torch.Tensor",
    variance: "torch.Tensor",
) -> "torch.Tensor":
    """Gaussian negative log-likelihood loss."""
    return torch.mean(
        0.5 * torch.log(variance) + 0.5 * ((target - mean) ** 2) / variance
    )
