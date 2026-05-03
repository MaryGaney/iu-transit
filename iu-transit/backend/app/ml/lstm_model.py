"""
app/ml/lstm_model.py
─────────────────────
PyTorch LSTM for bus delay prediction.
PyTorch is optional — if not installed the module imports cleanly and
LSTMPredictor falls back to the heuristic predictor automatically.

Architecture rationale:
──────────────────────
We use an LSTM (Long Short-Term Memory) over a Transformer because:
  1. Our sequences are SHORT (10 timesteps × 15 seconds = 2.5 minutes of history)
  2. Our dataset starts SMALL — LSTMs generalise from <1000 examples
  3. The temporal ordering is STRICTLY CAUSAL
  4. Inference latency: LSTM forward pass is microseconds per sample

Feature vector (9 features per timestep):
  [hour_sin, hour_cos, minute_sin, minute_cos,
   dow_sin, dow_cos,
   current_delay_seconds_norm,
   weather_severity,
   students_releasing_norm]
"""

import os
import math
import numpy as np
from typing import Optional

# ── Optional PyTorch import ───────────────────────────────────────────────────
# torch is excluded from production requirements.txt to keep Railway builds fast.
# The app runs fully on the heuristic fallback until a .pt model is available.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Dataset = object      # placeholder so type hints don't explode
    DataLoader = None

from app.core.config import settings
from app.core.logging import logger


# ── PyTorch model classes — only defined when torch is available ───────────────
if TORCH_AVAILABLE:

    class DelayLSTM(nn.Module):
        """
        LSTM delay predictor.
        Input:  (batch, seq_len, input_size)   where input_size = 9
        Output: (batch, 1)                     predicted delay (normalised)
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

            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

            self.variance_head = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus(),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            mean = self.head(last_hidden)
            variance = self.variance_head(last_hidden)
            return mean, variance

    class DelayDataset(Dataset):
        """Sliding-window dataset for LSTM training."""

        def __init__(self, sequences: np.ndarray, targets: np.ndarray):
            self.X = torch.tensor(sequences, dtype=torch.float32)
            self.y = torch.tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

else:
    # Stub classes so trainer.py imports don't crash when torch is absent
    class DelayLSTM:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed. Cannot instantiate DelayLSTM.")

    class DelayDataset:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed. Cannot instantiate DelayDataset.")


# ── Feature builder (no torch dependency) ────────────────────────────────────

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
    Pure Python/math — no torch dependency.
    """
    hour_sin   = math.sin(2 * math.pi * hour / 24)
    hour_cos   = math.cos(2 * math.pi * hour / 24)
    minute_sin = math.sin(2 * math.pi * minute / 60)
    minute_cos = math.cos(2 * math.pi * minute / 60)
    dow_sin    = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos    = math.cos(2 * math.pi * day_of_week / 7)

    delay_norm = (current_delay_seconds - delay_mean) / max(delay_std, 1.0)
    delay_norm = max(-3.0, min(3.0, delay_norm))

    return [
        hour_sin, hour_cos,
        minute_sin, minute_cos,
        dow_sin, dow_cos,
        delay_norm,
        weather_severity,
        students_releasing_norm,
    ]


# ── Predictor singleton ───────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Wrapper around DelayLSTM for inference.
    Falls back to heuristic predictions when torch is unavailable or
    no model file has been trained yet.
    """

    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.delay_mean = 0.0
        self.delay_std = 60.0
        self.max_students = 500.0
        # Only create a device if torch is available
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None

    def load(self) -> bool:
        """Load model weights from disk. Returns True if successful."""
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not installed — LSTM unavailable, using heuristic fallback.")
            return False

        model_path = settings.lstm_model_path
        if not os.path.exists(model_path):
            logger.info(f"No LSTM model at {model_path} — using heuristic fallback.")
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
            self.delay_mean   = checkpoint.get("delay_mean", 0.0)
            self.delay_std    = checkpoint.get("delay_std", 60.0)
            self.max_students = checkpoint.get("max_students", 500.0)
            self.is_loaded    = True
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
        Run inference. Falls back to heuristic if model not loaded.
        """
        if not self.is_loaded or self.model is None or not TORCH_AVAILABLE:
            return self._fallback_prediction(students_releasing, weather_severity)

        seq_len = settings.lstm_sequence_length
        while len(sequence) < seq_len:
            sequence = [sequence[0]] + sequence
        sequence = sequence[-seq_len:]

        x = torch.tensor([sequence], dtype=torch.float32)

        with torch.no_grad():
            mean, variance = self.model(x)

        delay_seconds = float(mean[0, 0]) * self.delay_std + self.delay_mean
        std_seconds   = float(torch.sqrt(variance[0, 0])) * self.delay_std
        lower         = delay_seconds - 1.28 * std_seconds
        upper         = delay_seconds + 1.28 * std_seconds
        confidence    = float(torch.exp(-variance[0, 0]).clamp(0.1, 0.99))

        return {
            "predicted_delay_seconds": round(delay_seconds, 1),
            "predicted_delay_lower":   round(lower, 1),
            "predicted_delay_upper":   round(upper, 1),
            "confidence":              round(confidence, 3),
            "delay_reason":            self._classify_reason(delay_seconds, students_releasing, weather_severity),
            "model_used":              True,
        }

    def _fallback_prediction(
        self, students_releasing: int, weather_severity: float
    ) -> dict:
        """Heuristic fallback — weather + class-schedule math."""
        base_delay = 0.0
        reason     = "on_time"

        students_norm = min(students_releasing / self.max_students, 1.0)
        if students_norm > 0.3:
            base_delay += students_norm * 120
            reason      = "class_release"

        if weather_severity > 0.3:
            base_delay += weather_severity * 90
            if students_norm <= 0.3:
                reason = "weather"

        return {
            "predicted_delay_seconds": round(base_delay, 1),
            "predicted_delay_lower":   round(base_delay * 0.5, 1),
            "predicted_delay_upper":   round(base_delay * 1.5 + 30, 1),
            "confidence":              0.4,
            "delay_reason":            reason,
            "model_used":              False,
        }

    @staticmethod
    def _classify_reason(
        delay_seconds: float, students_releasing: int, weather_severity: float
    ) -> str:
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


# Singleton used everywhere else in the app
predictor = LSTMPredictor()
