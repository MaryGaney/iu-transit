"""
app/ml/lstm_model.py
─────────────────────
PyTorch LSTM for bus delay prediction.

Architecture rationale:
──────────────────────
We use an LSTM (Long Short-Term Memory) over a Transformer because:
  1. Our sequences are SHORT (10 timesteps × 15 seconds = 2.5 minutes of history)
     — Transformers only outperform LSTMs on long sequences (100s+ steps)
  2. Our dataset starts SMALL — LSTMs generalise from <1000 examples; Transformers
     need orders of magnitude more data to avoid overfitting
  3. The temporal ordering is STRICTLY CAUSAL — each timestep depends on the
     previous delay propagating forward on a route. LSTMs model this naturally
     through their hidden state; Transformers require positional encodings
  4. Inference latency: LSTM forward pass is microseconds per sample; for a
     system polling every 15 seconds with 20-40 active vehicles, this is ideal
  5. No need for a pre-trained foundation model — the problem is too narrow
     and domain-specific; starting from scratch is appropriate

Future path: Once we have 30+ days of data (~200k samples), we can experiment
with a TinyTransformer or N-BEATS variant if LSTM error plateaus.

Feature vector (9 features per timestep):
  [hour_sin, hour_cos, minute_sin, minute_cos,       ← cyclical time encoding
   dow_sin, dow_cos,                                  ← cyclical day encoding
   current_delay_seconds_norm,                        ← z-scored
   weather_severity,                                  ← 0-1
   students_releasing_norm]                           ← 0-1 normalised
"""

import os
import math
import numpy as np
from typing import Optional
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from app.core.config import settings
from app.core.logging import logger


class DelayLSTM(nn.Module):
    """
    LSTM delay predictor.

    Input:  (batch, seq_len, input_size)   where input_size = 9
    Output: (batch, 1)                     predicted delay in seconds (normalised)
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output head: hidden → 1 value (predicted delay)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

        # Uncertainty head: predict variance for confidence intervals
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),   # ensures positive variance
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            mean: (batch, 1) — predicted delay (normalised)
            variance: (batch, 1) — predicted uncertainty
        """
        lstm_out, _ = self.lstm(x)
        # Use the LAST timestep's hidden state for prediction
        last_hidden = lstm_out[:, -1, :]
        mean = self.head(last_hidden)
        variance = self.variance_head(last_hidden)
        return mean, variance


class DelayDataset(Dataset):
    """
    Sliding-window dataset built from the vehicle_positions / trip_updates table.
    Each sample is a sequence of seq_len feature vectors with a target delay.
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: (N, seq_len, 9)
            targets:   (N, 1)
        """
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_feature_vector(
    hour: int,
    minute: int,
    day_of_week: int,
    current_delay_seconds: float,
    weather_severity: float,
    students_releasing_norm: float,
    delay_mean: float = 0.0,
    delay_std: float = 60.0,
) -> list[float]:
    """
    Build the 9-element feature vector for a single timestep.
    Time features are cyclically encoded to avoid artificial discontinuities
    (e.g. hour 23 → hour 0 should be close together, not far apart).
    """
    # Cyclical time encoding
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    minute_sin = math.sin(2 * math.pi * minute / 60)
    minute_cos = math.cos(2 * math.pi * minute / 60)
    dow_sin = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * day_of_week / 7)

    # Z-score normalise the delay
    delay_norm = (current_delay_seconds - delay_mean) / max(delay_std, 1.0)
    # Clip to ±3 sigma
    delay_norm = max(-3.0, min(3.0, delay_norm))

    return [
        hour_sin, hour_cos,
        minute_sin, minute_cos,
        dow_sin, dow_cos,
        delay_norm,
        weather_severity,
        students_releasing_norm,
    ]


class LSTMPredictor:
    """
    Wrapper around DelayLSTM for inference.
    Handles model loading, scaler persistence, and prediction with confidence.
    """

    def __init__(self):
        self.model: Optional[DelayLSTM] = None
        self.is_loaded = False
        self.delay_mean = 0.0
        self.delay_std = 60.0
        self.max_students = 500.0  # for normalising student counts
        self.device = torch.device("cpu")  # CPU is fine for this scale

    def load(self) -> bool:
        """Load model weights from disk. Returns True if successful."""
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not installed — using heuristic fallback.")
            return False
        model_path = settings.lstm_model_path
        if not os.path.exists(model_path):
            logger.info(f"No LSTM model found at {model_path}. Will use schedule fallback.")
            return False
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = DelayLSTM(
                input_size=settings.lstm_input_features,
                hidden_size=settings.lstm_hidden_size,
                num_layers=settings.lstm_num_layers,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.delay_mean = checkpoint.get("delay_mean", 0.0)
            self.delay_std = checkpoint.get("delay_std", 60.0)
            self.max_students = checkpoint.get("max_students", 500.0)
            self.is_loaded = True
            logger.info(f"LSTM model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False

    def predict(
        self,
        sequence: list[list[float]],
        students_releasing: int = 0,
        weather_severity: float = 0.0,
    ) -> dict:
        """
        Run inference on a sequence of feature vectors.

        Args:
            sequence: list of seq_len feature vectors (each length 9)
            students_releasing: count for the current stop/time
            weather_severity: 0-1

        Returns dict with:
            predicted_delay_seconds, lower, upper, confidence, delay_reason
        """
        if not self.is_loaded or self.model is None:
            return self._fallback_prediction(students_releasing, weather_severity)

        # Pad or truncate sequence to expected length
        seq_len = settings.lstm_sequence_length
        while len(sequence) < seq_len:
            sequence = [sequence[0]] + sequence  # repeat first frame
        sequence = sequence[-seq_len:]

        x = torch.tensor([sequence], dtype=torch.float32)  # (1, seq_len, 9)

        with torch.no_grad():
            mean, variance = self.model(x)

        # Denormalise
        delay_seconds = float(mean[0, 0]) * self.delay_std + self.delay_mean
        std_seconds = float(torch.sqrt(variance[0, 0])) * self.delay_std

        lower = delay_seconds - 1.28 * std_seconds   # ~10th percentile
        upper = delay_seconds + 1.28 * std_seconds   # ~90th percentile
        confidence = float(torch.exp(-variance[0, 0]).clamp(0.1, 0.99))

        delay_reason = self._classify_reason(
            delay_seconds, students_releasing, weather_severity
        )

        return {
            "predicted_delay_seconds": round(delay_seconds, 1),
            "predicted_delay_lower": round(lower, 1),
            "predicted_delay_upper": round(upper, 1),
            "confidence": round(confidence, 3),
            "delay_reason": delay_reason,
            "model_used": True,
        }

    def _fallback_prediction(
        self, students_releasing: int, weather_severity: float
    ) -> dict:
        """
        Heuristic fallback when no model is trained yet.
        Based purely on the two signal features — better than showing '0 delay'.
        """
        base_delay = 0.0
        reason = "on_time"

        students_norm = min(students_releasing / self.max_students, 1.0)
        if students_norm > 0.3:
            base_delay += students_norm * 120  # up to 2 min for class release
            reason = "class_release"

        if weather_severity > 0.3:
            base_delay += weather_severity * 90  # up to 90s for severe weather
            reason = "weather" if students_norm <= 0.3 else reason

        return {
            "predicted_delay_seconds": round(base_delay, 1),
            "predicted_delay_lower": round(base_delay * 0.5, 1),
            "predicted_delay_upper": round(base_delay * 1.5 + 30, 1),
            "confidence": 0.4,
            "delay_reason": reason,
            "model_used": False,
        }

    @staticmethod
    def _classify_reason(
        delay_seconds: float, students_releasing: int, weather_severity: float
    ) -> str:
        """Classify the dominant reason for a delay, for UI display."""
        if delay_seconds < 30:
            return "on_time"
        students_norm = min(students_releasing / 500.0, 1.0)
        if students_norm > 0.2 and weather_severity < 0.3:
            return "class_release"
        if weather_severity > 0.3 and students_norm < 0.2:
            return "weather"
        if students_norm > 0.2 and weather_severity > 0.3:
            return "cascading"
        return "unknown"


# Singleton predictor instance
predictor = LSTMPredictor()
