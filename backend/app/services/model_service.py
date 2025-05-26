import logging
from typing import Dict, Any, Optional, AsyncIterator, List
import numpy as np
from datetime import datetime

from ..domain.entities import ModelType, DetectionResult, DroneData
from ..domain.usecases import ModelUseCase
from ..infra.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelService(ModelUseCase):
    """Implementation of the ModelUseCase with batch prediction support"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.current_model_type: Optional[ModelType] = None
        # Store waypoints for batch prediction
        self.waypoint_buffer: List[DroneData] = []
        self.is_collecting_waypoints = False
    
    async def load_model(self, model_type: ModelType) -> None:
        """Load the specified AI model."""
        logger.info(f"Loading model: {model_type}")
        self.model_loader.load(model_type) 
        self.current_model_type = model_type

    
    def start_waypoint_collection(self) -> None:
        """Start collecting waypoints for batch prediction"""
        self.waypoint_buffer.clear()
        self.is_collecting_waypoints = True
        logger.info("Started collecting waypoints for batch prediction")
    
    def add_waypoint(self, data: DroneData) -> None:
        """Add a waypoint to the collection buffer"""
        if self.is_collecting_waypoints:
            self.waypoint_buffer.append(data)
            logger.debug(f"Added waypoint {len(self.waypoint_buffer)}: {data}")
    
    async def predict_batch(self) -> DetectionResult:
        """Make prediction on all collected waypoints"""
        if not self.current_model_type:
            raise ValueError("No model loaded. Call load_model first.")
        
        if not self.waypoint_buffer:
            logger.warning("No waypoints collected for batch prediction")
            return DetectionResult(
                is_anomaly=False,
                confidence=0.0,
                detection_type=None,
                details={"error": "No waypoints to analyze"}
            )
        
        logger.info(f"Making batch prediction on {len(self.waypoint_buffer)} waypoints")
        
        # Convert all waypoints to a feature matrix
        feature_matrix = []
        for waypoint in self.waypoint_buffer:
            features = waypoint.to_feature_vector()
            feature_matrix.append(features)
        
        # Convert to numpy array
        features = np.array(feature_matrix)
        logger.info(f"Feature matrix shape: {features.shape}")
        
        # Option 1: Flatten all features into one vector (concatenate all waypoints)
        flattened_features = features.flatten().reshape(1, -1)
        
        # Option 2: Use statistical aggregation (mean, std, min, max across waypoints)
        aggregated_features = self._aggregate_waypoint_features(features)
        
        # Option 3: Use the last waypoint (destination behavior)
        final_waypoint_features = features[-1].reshape(1, -1)
        
        # Choose which approach to use (you can make this configurable)
        prediction_features = aggregated_features  # Using aggregated approach
        
        # Get prediction
        result = self.model_loader.predict(prediction_features)
        
        # Stop collecting waypoints
        self.is_collecting_waypoints = False
        
        return DetectionResult(
            is_anomaly=result["is_anomaly"],
            confidence=result["confidence"],
            detection_type=result["detection_type"],
            details={
                **result["details"],
                "waypoints_analyzed": len(self.waypoint_buffer),
                "prediction_method": "aggregated_features"
            }
        )
    
    def _aggregate_waypoint_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Aggregate features across all waypoints using statistical measures"""
        try:
            # Calculate statistical features across waypoints
            mean_features = np.mean(feature_matrix, axis=0)
            std_features = np.std(feature_matrix, axis=0)
            min_features = np.min(feature_matrix, axis=0)
            max_features = np.max(feature_matrix, axis=0)
            
            # Calculate additional trajectory-based features
            if len(feature_matrix) > 1:
                # Rate of change features
                diff_features = np.diff(feature_matrix, axis=0)
                mean_velocity = np.mean(diff_features, axis=0)
                max_acceleration = np.max(np.abs(diff_features), axis=0)
            else:
                mean_velocity = np.zeros_like(mean_features)
                max_acceleration = np.zeros_like(mean_features)
            
            # Combine all aggregated features
            aggregated = np.concatenate([
                mean_features,
                std_features,
                min_features,
                max_features,
                mean_velocity,
                max_acceleration
            ])
            
            logger.info(f"Aggregated features shape: {aggregated.shape}")
            return aggregated.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error aggregating features: {str(e)}")
            # Fallback to simple mean
            return np.mean(feature_matrix, axis=0).reshape(1, -1)
    
    async def predict(self, data: DroneData) -> Optional[DetectionResult]:
        """Add waypoint to buffer or make real-time prediction"""
        if self.is_collecting_waypoints:
            # Add to buffer for batch prediction
            self.add_waypoint(data)
            return None  # No prediction yet
        else:
            # Make immediate prediction (original behavior)
            if not self.current_model_type:
                raise ValueError("No model loaded. Call load_model first.")
            
            features = np.array(data.to_feature_vector())
            result = self.model_loader.predict(features)
            
            return DetectionResult(
                is_anomaly=result["is_anomaly"],
                confidence=result["confidence"],
                detection_type=result["detection_type"],
                details=result["details"]
            )
    
    def get_collected_waypoints_count(self) -> int:
        """Get the number of currently collected waypoints"""
        return len(self.waypoint_buffer)
    
    def clear_waypoint_buffer(self) -> None:
        """Clear the waypoint buffer"""
        self.waypoint_buffer.clear()
        self.is_collecting_waypoints = False
        logger.info("Cleared waypoint buffer")
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics about the current model"""
        if not self.current_model_type:
            return {"error": "No model loaded"}
        
        metrics = self.model_loader.get_metrics()
        metrics.update({
            "waypoints_collected": len(self.waypoint_buffer),
            "is_collecting": self.is_collecting_waypoints
        })
        
        return metrics