from typing import List, Dict, Any, Optional, Callable, AsyncIterator, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

from ..domain.entities import (
    ModelType, SimulationType, DroneData, DetectionResult, DronePosition
)


class SimulationUseCase(ABC):
    """Abstract simulation use case"""
    
    @abstractmethod
    async def start_simulation(
        self, 
        simulation_type: SimulationType, 
        duration: int = 300,
        step: float = 0.1
    ) -> AsyncIterator[DroneData]:
        """Start a drone simulation and yield drone data"""
        pass
    
    @abstractmethod
    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        pass
    
    @abstractmethod
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get the current status of simulation"""
        pass


class ModelUseCase(ABC):
    """Abstract model use case"""
    
    @abstractmethod
    async def load_model(self, model_type: ModelType) -> None:
        """Load a specific AI model"""
        pass
    
    @abstractmethod
    async def predict(self, data: DroneData) -> DetectionResult:
        """Make predictions using the loaded model"""
        pass
    
    @abstractmethod
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics about the current model"""
        pass


class RealTimeMonitoringUseCase:
    """Handles real-time monitoring of drone data and detection results"""
    
    def __init__(
        self, 
        simulation_use_case: SimulationUseCase,
        model_use_case: ModelUseCase
    ):
        self.simulation_use_case = simulation_use_case
        self.model_use_case = model_use_case
        self._running = False
        self._callbacks: List[Callable[[DroneData, DetectionResult], None]] = []
    
    def register_callback(self, callback: Callable[[DroneData, DetectionResult], None]) -> None:
        """Register a callback for real-time updates"""
        self._callbacks.append(callback)
    
    async def start_monitoring(
        self, 
        simulation_type: SimulationType,
        model_type: ModelType,
        duration: int = 300,
        step: float = 0.1
    ) -> AsyncIterator[Tuple[DroneData, DetectionResult]]:
        """Start monitoring and yield drone data with detection results"""
        await self.model_use_case.load_model(model_type)
        self._running = True
        
        async for drone_data in self.simulation_use_case.start_simulation(
            simulation_type=simulation_type,
            duration=duration,
            step=step
        ):
            if not self._running:
                break
                
            # Get model prediction
            detection_result = await self.model_use_case.predict(drone_data)
            
            # Notify callbacks
            for callback in self._callbacks:
                callback(drone_data, detection_result)
                
            yield drone_data, detection_result
            
            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.01)
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring process"""
        self._running = False
        await self.simulation_use_case.stop_simulation()