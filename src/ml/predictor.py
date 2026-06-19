"""
============================================
SignBridge - Real-Time Predictor Module
============================================
Loads the trained Random Forest model and
predicts signs from live landmark data.

Uses a "rolling window" to smooth predictions:
  Instead of showing every raw prediction
  (which flickers), it looks at the last 15
  predictions and picks the most common one.

Think of it like autocorrect - it waits to
see a consistent pattern before committing.
============================================
"""

import os
import numpy as np
import joblib
from collections import Counter

from src.utils.config import (
    MODEL_PATH,
    PREDICTION_WINDOW,
    MIN_PREDICTION_CONFIDENCE,
)


class Predictor:
    """
    Real-time sign predictor with smoothing.
    
    Usage:
        predictor = Predictor()
        sign, confidence = predictor.predict(features)
        # sign = "A" (or None if not confident enough)
        # confidence = 0.96
    """

    def __init__(self):
        # Load the trained model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}\n"
                "Train the model first: python -m src.ml.trainer"
            )

        self.model = joblib.load(MODEL_PATH)
        print(f"  Model loaded from {MODEL_PATH}")

        # Rolling window for smoothing
        # Stores the last N predictions
        self.prediction_history = []

    def predict(self, features):
        """
        Predict a sign from normalized landmark features.

        Args:
            features: list of 63 normalized floats
                      from FeatureExtractor.normalize()

        Returns:
            (sign, confidence) tuple:
                sign: str like "A", "HELLO", or None if not confident
                confidence: float between 0.0 and 1.0
        """
        if features is None:
            return None, 0.0

        # ── Get raw prediction from the model ──
        # Reshape to 2D array (model expects a "batch" of samples)
        features_array = np.array(features).reshape(1, -1)

        # predict_proba gives probability for EACH class
        # e.g., [0.01, 0.02, ..., 0.96, ...] (one per sign)
        probabilities = self.model.predict_proba(features_array)[0]

        # The predicted class is the one with highest probability
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index]

        # Get the actual label name (e.g., "A", "HELLO")
        predicted_sign = self.model.classes_[predicted_index]

        # ── Apply smoothing with rolling window ──
        # Add this prediction to our history
        self.prediction_history.append(predicted_sign)

        # Keep only the last N predictions
        if len(self.prediction_history) > PREDICTION_WINDOW:
            self.prediction_history = self.prediction_history[-PREDICTION_WINDOW:]

        # Find the most common prediction in the window
        # Counter counts occurrences: {"A": 12, "B": 2, "C": 1}
        counter = Counter(self.prediction_history)
        stable_sign, stable_count = counter.most_common(1)[0]

        # Calculate stability (what fraction of the window agrees)
        stability = stable_count / len(self.prediction_history)

        # ── Apply confidence threshold ──
        # Only return a prediction if both:
        #   1. Model confidence is high enough
        #   2. Prediction is stable across frames
        if confidence >= MIN_PREDICTION_CONFIDENCE and stability >= 0.5:
            return stable_sign, confidence

        return None, confidence

    def reset(self):
        """Clear prediction history (e.g., when switching modes)."""
        self.prediction_history = []
