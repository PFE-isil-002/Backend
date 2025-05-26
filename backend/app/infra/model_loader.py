import os
import joblib
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
from ..domain.entities import ModelType
from ..core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.scaler = None
        self.expected_feature_count = None

    def _get_model_dir(self) -> Path:
        """Return the absolute path to the model directory."""
        return settings.BASE_DIR / "model" / "les_models"

    def load(self, model_type: ModelType) -> None:
        """Load model and scaler from disk."""
        model_dir = self._get_model_dir()
        model_path = model_dir / f"{model_type.value}_model.pkl"
        scaler_path = model_dir / f"{model_type.value}_scaler.pkl"

        logger.info(f"Loading model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded = joblib.load(model_path)
        self.model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        self.model_type = model_type

        if hasattr(self.model, "n_features_in_"):
            self.expected_feature_count = self.model.n_features_in_

        if scaler_path.exists():
            logger.info(f"Loading scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            if isinstance(self.scaler, dict) and "scaler" in self.scaler:
                self.scaler = self.scaler["scaler"]

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        if self.expected_feature_count and features.shape[1] < self.expected_feature_count:
            padding = np.zeros((features.shape[0], self.expected_feature_count - features.shape[1]))
            features = np.hstack((features, padding))
        return features

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("No model loaded.")

        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if self.scaler:
            features = self.scaler.transform(features)
        elif self.expected_feature_count and features.shape[1] != self.expected_feature_count:
            features = self._transform_features(features)

        is_anomaly = False
        confidence = 0.0
        detection_type = "normal"

        try:
            prediction = self.model.predict(features)[0]
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(features)[0]
                confidence = float(np.max(probas))
            else:
                confidence = 1.0
            is_anomaly = prediction != 0

            if is_anomaly and hasattr(self.model, "classes_"):
                class_idx = int(np.argmax(probas))
                label = str(self.model.classes_[class_idx]).lower()
                if "mitm" in label:
                    detection_type = "mitm"
                elif "outsider" in label:
                    detection_type = "outsider"
                elif "adversarial" in label:
                    detection_type = "adversarial"
                else:
                    detection_type = "unknown"

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "is_anomaly": False,
                "confidence": 0.0,
                "detection_type": "error",
                "details": {"error": str(e)}
            }

        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "detection_type": detection_type if is_anomaly else "normal",
            "details": {"model_type": str(self.model_type)}
        }
