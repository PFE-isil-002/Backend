import logging
from typing import Dict, Any, Optional, AsyncIterator, List
import numpy as np
from datetime import datetime

from ..domain.entities import ModelType, DetectionResult, DroneData
from ..domain.usecases import ModelUseCase
from ..infra.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelService(ModelUseCase):
    """Implementation of the ModelUseCase"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.current_model_type: Optional[ModelType] = None
    
    async def load_model(self, model_type: ModelType) -> None:
        """Load the specified AI model"""
        logger.info(f"Loading model: {model_type}")
        await self.model_loader.load_model(model_type)
        self.current_model_type = model_type
    
    async def predict(self, data: DroneData) -> DetectionResult:
        """Make predictions using the loaded model"""
        if not self.current_model_type:
            raise ValueError("No model loaded. Call load_model first.")
        
        # Convert drone data to features
        features = np.array(data.to_feature_vector())
        
        # Get prediction
        result = self.model_loader.predict(features)
        
        return DetectionResult(
            is_anomaly=result["is_anomaly"],
            confidence=result["confidence"],
            detection_type=result["detection_type"],
            details=result["details"]
        )
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics about the current model"""
        if not self.current_model_type:
            return {"error": "No model loaded"}
        
        return self.model_loader.get_metrics()