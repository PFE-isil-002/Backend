import logging
import asyncio
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta

from ..domain.entities import SimulationType, DroneData
from ..domain.usecases import SimulationUseCase
from ..infra.airsim_adapter import AirSimAdapter

logger = logging.getLogger(__name__)


class SimulationService(SimulationUseCase):
    """Implementation of the SimulationUseCase"""
    
    def __init__(self):
        self.airsim_adapter = AirSimAdapter()
        self.current_simulation_type: Optional[SimulationType] = None
        self.start_time: Optional[datetime] = None
        self.duration: int = 300  # default duration in seconds
    
    async def start_simulation(
        self, 
        simulation_type: SimulationType, 
        duration: int = 300,
        step: float = 0.1
    ) -> AsyncIterator[DroneData]:
        """Start a drone simulation and yield drone data"""
        logger.info(f"Starting simulation: {simulation_type}, duration: {duration}s, step: {step}s")
        
        # Set current simulation parameters
        self.current_simulation_type = simulation_type
        self.start_time = datetime.now()
        self.duration = duration
        
        # Start simulation and yield drone data
        async for drone_data in self.airsim_adapter.start_simulation(
            simulation_type=simulation_type,
            duration=duration,
            step=step
        ):
            yield drone_data
            
            # Check if simulation should end
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed >= duration:
                logger.info("Simulation duration reached, stopping")
                await self.stop_simulation()
                break
    
    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        logger.info("Stopping simulation")
        await self.airsim_adapter.stop_simulation()
        self.current_simulation_type = None
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get the current status of simulation"""
        running = self.current_simulation_type is not None
        
        status = {
            "running": running,
            "simulation_type": self.current_simulation_type.value if running else None,
            "elapsed_time": 0.0,
            "remaining_time": 0.0
        }
        
        if running and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            status["elapsed_time"] = elapsed
            status["remaining_time"] = max(0, self.duration - elapsed)
            
        return status