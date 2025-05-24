import os
import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
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
        self.feature_names = None
        self.expected_feature_count = None
    
    async def load_model(self, model_type: ModelType) -> None:
        """Load the specified model from disk"""
        self.model_type = model_type
        
        # Get model path based on type
        model_path = self._get_model_path(model_type)
        scaler_path = self._get_scaler_path(model_type)
        
        logger.info(f"Attempting to load {model_type} model from {model_path}")
        
        try:
            # Load model based on type
            if model_type in [ModelType.LSTM, ModelType.RNN]:
                # For deep learning models with TensorFlow
                self.model = tf.keras.models.load_model(model_path)
            else:
                # For sklearn models
                loaded_object = joblib.load(model_path)
                
                # Debug information
                logger.info(f"Loaded object type: {type(loaded_object)}")
                
                # Check if the loaded object is a dictionary (which might happen if it was saved with metadata)
                if isinstance(loaded_object, dict) and "model" in loaded_object:
                    self.model = loaded_object["model"]
                    logger.info(f"Extracted model from dictionary, model type: {type(self.model)}")
                else:
                    self.model = loaded_object
            
            # Verify the model has a predict method
            if not hasattr(self.model, "predict"):
                raise AttributeError(f"The loaded model does not have a 'predict' method. Type: {type(self.model)}")
            
            # Try to determine expected feature count
            self._determine_feature_count()
            
            # Load scaler if exists
            if scaler_path.exists():
                logger.info(f"Loading scaler from {scaler_path}")
                scaler_object = joblib.load(scaler_path)
                
                # Check if the scaler is also stored in a dictionary
                if isinstance(scaler_object, dict) and "scaler" in scaler_object:
                    self.scaler = scaler_object["scaler"]
                else:
                    self.scaler = scaler_object
                
                # Try to extract feature names from scaler if possible
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_names = self.scaler.feature_names_in_
                    logger.info(f"Extracted feature names from scaler: {len(self.feature_names)} features")
            else:
                self.scaler = None
                logger.info(f"No scaler found at {scaler_path}")
                
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
    
    def _determine_feature_count(self):
        """Try to determine the number of features the model expects"""
        try:
            if hasattr(self.model, '_n_features'):
                self.expected_feature_count = self.model._n_features
            elif hasattr(self.model, 'n_features_in_'):
                self.expected_feature_count = self.model.n_features_in_
            elif hasattr(self.model, 'feature_importances_'):
                self.expected_feature_count = len(self.model.feature_importances_)
            elif hasattr(self.model, 'coef_') and hasattr(self.model.coef_, 'shape'):
                self.expected_feature_count = self.model.coef_.shape[1]
            
            if self.expected_feature_count:
                logger.info(f"Model expects {self.expected_feature_count} features")
            else:
                logger.warning("Could not determine expected feature count")
        except Exception as e:
            logger.warning(f"Error determining feature count: {str(e)}")
            self.expected_feature_count = None
    
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform input features to match what the model expects"""
        original_feature_count = features.shape[1] if len(features.shape) > 1 else 1
        logger.info(f"Original features shape: {features.shape}")
        
        # Check if we need to do feature expansion
        if self.expected_feature_count and original_feature_count < self.expected_feature_count:
            logger.info(f"Need to expand features from {original_feature_count} to {self.expected_feature_count}")
            
            # Method 1: Time series approach - use a sliding window to create features
            # This is common for UAV data where you might have multiple timesteps
            try:
                # Calculate how many time steps we need
                if original_feature_count > 0 and self.expected_feature_count % original_feature_count == 0:
                    time_steps = self.expected_feature_count // original_feature_count
                    logger.info(f"Using time window approach with {time_steps} time steps")
                    
                    # For testing, just duplicate the current features
                    expanded = np.tile(features, time_steps)
                    logger.info(f"Expanded features shape: {expanded.shape}")
                    return expanded
            except Exception as e:
                logger.error(f"Error in time series expansion: {str(e)}")
            
            # Method 2: If we have historical data, try to compute rolling statistics
            try:
                # For demo purposes, just padding with zeros
                # In a real system, you would use real historical data and compute meaningful features
                padding = np.zeros((1, self.expected_feature_count - original_feature_count))
                expanded = np.hstack((features, padding))
                logger.info(f"Expanded features shape with zero padding: {expanded.shape}")
                return expanded
            except Exception as e:
                logger.error(f"Error in feature padding: {str(e)}")
                
            # Method 3: Simplest approach - just pad with zeros
            try:
                # Create a new array of the right size
                result = np.zeros((1, self.expected_feature_count))
                # Copy over the original features
                result[0, :original_feature_count] = features.flatten()[:original_feature_count]
                return result
            except Exception as e:
                logger.error(f"Error in zero padding: {str(e)}")
                
        return features
    
    def _load_feature_template(self) -> Optional[np.ndarray]:
        """Try to load a feature template from training data if available"""
        try:
            possible_template_paths = [
                settings.DATASET_DIR / "vols_aplatis_BON.csv",
                settings.BASE_DIR / "model" / "les_models" / "vols_aplatis_BON.csv",
                settings.BASE_DIR / "model" / "Flitrage" / "Vol_B" / "vols_aplatis_BON.csv",
            ]
            
            for path in possible_template_paths:
                if path.exists():
                    logger.info(f"Found feature template data at {path}")
                    # Load just the header to get feature names
                    import pandas as pd
                    df = pd.read_csv(path, nrows=1)
                    return np.zeros((1, len(df.columns)))
            
            return None
        except Exception as e:
            logger.error(f"Error loading feature template: {str(e)}")
            return None
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        # Ensure features is a numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Reshape to ensure 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Log input features for debugging
        logger.info(f"Input features shape: {features.shape}")
        if features.shape[1] < 20:  # Only log if not too many features
            logger.info(f"Input features: {features}")
        
        try:
            # Apply scaler if available
            if self.scaler:
                try:
                    # Make sure feature dimensions match what the scaler expects
                    if hasattr(self.scaler, 'n_features_in_') and features.shape[1] != self.scaler.n_features_in_:
                        logger.warning(f"Feature count mismatch: scaler expects {self.scaler.n_features_in_}, got {features.shape[1]}")
                        
                        # Transform features to match expectations if possible
                        features = self._transform_features(features)
                    
                    # Now try to scale
                    features = self.scaler.transform(features)
                    logger.info(f"Features after scaling: shape={features.shape}")
                except Exception as e:
                    logger.error(f"Error applying scaler: {str(e)}")
                    # Try to proceed without scaling
            else:
                # Even if no scaler, we might need to transform features
                if self.expected_feature_count and features.shape[1] != self.expected_feature_count:
                    logger.warning(f"Feature count mismatch: model expects {self.expected_feature_count}, got {features.shape[1]}")
                    features = self._transform_features(features)
            
            logger.info(f"Making prediction with model type: {self.model_type}, model object: {type(self.model)}")
            
            # Handle different model types
            if self.model_type in [ModelType.LSTM, ModelType.RNN]:
                # Reshape for RNN/LSTM if needed
                if len(features.shape) == 2 and hasattr(self.model, 'input_shape') and self.model.input_shape[1] != features.shape[0]:
                    # Reshape to match the expected input shape of the model
                    time_steps = self.model.input_shape[1]
                    n_features = features.shape[1]
                    features = features.reshape(1, time_steps, n_features)
                    
                # Get prediction
                raw_prediction = self.model.predict(features)
                logger.info(f"Raw prediction: {raw_prediction}, type: {type(raw_prediction)}")
                
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
                logger.info(f"Prediction: {prediction}")
                
                # Get probabilities if available
                if hasattr(self.model, "predict_proba"):
                    probas = self.model.predict_proba(features)[0]
                    confidence = float(np.max(probas))
                    logger.info(f"Probabilities: {probas}, confidence: {confidence}")
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
                        raw_prediction = self.model.predict_proba(features)[0] if hasattr(self.model, "predict_proba") else None
                        if raw_prediction is not None:
                            class_idx = np.argmax(raw_prediction)
                            classes = self.model.classes_
                            if len(classes) > 1 and class_idx == 1 or (isinstance(classes[class_idx], str) and "mitm" in str(classes[class_idx]).lower()):
                                detection_type = "mitm"
                            elif len(classes) > 2 and class_idx == 2 or (isinstance(classes[class_idx], str) and "outsider" in str(classes[class_idx]).lower()):
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
        # Check both potential locations
        possible_dirs = [
            settings.MODEL_DIR,           # model/les_models
            settings.BASE_DIR / "models"  # models/
        ]
        
        filename = f"{model_type.value}_model"
        
        for directory in possible_dirs:
            logger.info(f"Looking for model files in {directory}")
            # Check for different extensions
            for ext in [".pkl", ".joblib", ".h5", ".keras", ""]:
                path = directory / f"{filename}{ext}"
                if path.exists():
                    logger.info(f"Found model at: {path}")
                    return path
                    
            # Also check for filenames without "_model" suffix
            path = directory / f"{model_type.value}.pkl"
            if path.exists():
                logger.info(f"Found model at: {path}")
                return path
                
            # Check for models with full name (e.g., "random_forest_model.pkl")
            if model_type == ModelType.RANDOM_FOREST:
                path = directory / "random_forest_model.pkl"
                if path.exists():
                    logger.info(f"Found model at: {path}")
                    return path
            elif model_type == ModelType.LOGISTIC_REGRESSION:
                path = directory / "logistic_regression_model.pkl"
                if path.exists():
                    logger.info(f"Found model at: {path}")
                    return path
            elif model_type == ModelType.KNN:
                path = directory / "knn_model.pkl"
                if path.exists():
                    logger.info(f"Found model at: {path}")
                    return path
        
        # Default path if no file exists yet
        logger.warning(f"No model file found for {model_type.value}")
        return settings.MODEL_DIR / f"{filename}.pkl"
    
    def _get_scaler_path(self, model_type: ModelType) -> Path:
        """Get the path to the scaler file"""
        # First check if there's a dedicated scaler file
        scaler_path = settings.MODEL_DIR / f"{model_type.value}_scaler.pkl"
        if scaler_path.exists():
            return scaler_path
            
        # Check for preprocessor files (which might contain the scaler)
        possible_scaler_files = [
            settings.MODEL_DIR / f"preprocessor_{model_type.value}.pkl",
            settings.MODEL_DIR / "preprocessor_RF.pkl" if model_type == ModelType.RANDOM_FOREST else None,
            settings.MODEL_DIR / "preprocessor_lr.pkl" if model_type == ModelType.LOGISTIC_REGRESSION else None,
            settings.BASE_DIR / "models" / f"preprocessor_{model_type.value}.pkl",
            settings.BASE_DIR / "models" / "preprocessor_RF.pkl" if model_type == ModelType.RANDOM_FOREST else None,
            settings.BASE_DIR / "models" / "preprocessor_lr.pkl" if model_type == ModelType.LOGISTIC_REGRESSION else None,
        ]
        
        for path in possible_scaler_files:
            if path and path.exists():
                logger.info(f"Found scaler/preprocessor at: {path}")
                return path
                
        return scaler_path  # Return default path even if it doesn't exist
    
    def _get_metrics_path(self, model_type: ModelType) -> Path:
        """Get the path to the metrics file"""
        return settings.MODEL_DIR / f"{model_type.value}_metrics.pkl"