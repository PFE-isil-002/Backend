import os
import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import tensorflow as tf

from ..core.config import settings
from ..domain.entities import ModelType

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages ML models for UAV authentication"""
    
    def __init__(self):
        self.model = None
        self.model_type: Optional[ModelType] = None
        self.model_metrics: Dict[str, Any] = {}
        self.scaler = None
    
    async def load_model(self, model_type: ModelType) -> None:
        """Load the specified model from disk"""
        self.model_type = model_type
        
        # Get model path based on type
        model_path = self._get_model_path(model_type)
        scaler_path = self._get_scaler_path(model_type)
        
        try:
            # Load model based on type
            if model_type in [ModelType.LSTM, ModelType.RNN]:
                # For deep learning models with TensorFlow
                self.model = tf.keras.models.load_model(model_path)
            else:
                # For sklearn models
                self.model = joblib.load(model_path)
            
            # Load scaler if exists
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
                
            # Load model metrics if exists
            metrics_path = self._get_metrics_path(model_type)
            if metrics_path.exists():
                self.model_metrics = joblib.load(metrics_path)
            else:
                self.model_metrics = {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None
                }
                
            logger.info(f"Successfully loaded {model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            raise
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        # Apply scaler if available
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        try:
            # Handle different model types
            if self.model_type in [ModelType.LSTM, ModelType.RNN]:
                # Reshape for RNN/LSTM if needed
                if len(features.shape) == 2 and self.model.input_shape[1] != features.shape[0]:
                    # Reshape to match the expected input shape of the model
                    time_steps = self.model.input_shape[1]
                    n_features = features.shape[1]
                    features = features.reshape(1, time_steps, n_features)
                    
                # Get prediction
                raw_prediction = self.model.predict(features)
                
                # Convert to anomaly classification
                if isinstance(raw_prediction, np.ndarray) and raw_prediction.ndim > 1:
                    confidence = float(np.max(raw_prediction))
                    is_anomaly = bool(np.argmax(raw_prediction) != 0)  # Assuming 0 is normal
                else:
                    confidence = float(raw_prediction[0])
                    is_anomaly = bool(confidence > 0.5)
                    
            elif self.model_type in [ModelType.KNN, ModelType.SVM, ModelType.RANDOM_FOREST]:
                # Classification models
                prediction = self.model.predict(features)[0]
                
                # Get probabilities if available
                if hasattr(self.model, "predict_proba"):
                    probas = self.model.predict_proba(features)[0]
                    confidence = float(np.max(probas))
                else:
                    confidence = 1.0  # Default confidence if not available
                    
                is_anomaly = bool(prediction != 0)  # Assuming 0 is normal class
                
            elif self.model_type == ModelType.LOGISTIC_REGRESSION:
                # Logistic regression
                prediction = self.model.predict(features)[0]
                probas = self.model.predict_proba(features)[0]
                confidence = float(np.max(probas))
                is_anomaly = bool(prediction != 0)  # Assuming 0 is normal class
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Determine detection type
            detection_type = None
            if is_anomaly:
                # This is simplified - in a real system you'd have more sophisticated logic
                # to determine if it's MITM or outsider based on prediction class or other factors
                if self.model_type in [ModelType.LSTM, ModelType.RNN, ModelType.LOGISTIC_REGRESSION]:
                    # For multi-class models that can distinguish between attack types
                    if hasattr(self.model, "classes_"):
                        class_idx = np.argmax(raw_prediction)
                        classes = self.model.classes_
                        if class_idx == 1 or (isinstance(classes[class_idx], str) and "mitm" in classes[class_idx].lower()):
                            detection_type = "mitm"
                        elif class_idx == 2 or (isinstance(classes[class_idx], str) and "outsider" in classes[class_idx].lower()):
                            detection_type = "outsider"
                else:
                    # Binary models just detect anomaly but not type
                    detection_type = "unknown"
            
            return {
                "is_anomaly": is_anomaly,
                "confidence": confidence,
                "detection_type": detection_type,
                "details": {
                    "model_type": self.model_type
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "is_anomaly": False,
                "confidence": 0.0,
                "detection_type": None,
                "details": {
                    "error": str(e)
                }
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the currently loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_type": self.model_type,
            **self.model_metrics
        }
    
    def _get_model_path(self, model_type: ModelType) -> Path:
        """Get the path to the model file"""
        filename = f"{model_type.value}_model"
        
        # Check for different extensions
        for ext in [".pkl", ".joblib", ".h5", ".keras", ""]:
            path = settings.MODEL_DIR / f"{filename}{ext}"
            if path.exists():
                return path
        
        # Default path if no file exists yet
        return settings.MODEL_DIR / f"{filename}.pkl"
    
    def _get_scaler_path(self, model_type: ModelType) -> Path:
        """Get the path to the scaler file"""
        return settings.MODEL_DIR / f"{model_type.value}_scaler.pkl"
    
    def _get_metrics_path(self, model_type: ModelType) -> Path:
        """Get the path to the metrics file"""
        return settings.MODEL_DIR / f"{model_type.value}_metrics.pkl"