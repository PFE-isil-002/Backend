import logging
import asyncio
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta
import time

from ..domain.entities import SimulationType, DroneData
from ..domain.usecases import SimulationUseCase
from ..infra.airsim_adapter import AirSimAdapter

# Configure module logger
logger = logging.getLogger("app.simulation_service")


class SimulationService(SimulationUseCase):
    """Implementation of the SimulationUseCase"""
    
    def __init__(self):
        self.airsim_adapter = AirSimAdapter()
        self.current_simulation_type: Optional[SimulationType] = None
        self.start_time: Optional[datetime] = None
        self.duration: int = 300  # default duration in seconds
        logger.info("SimulationService initialized")
    
    async def start_simulation(
        self, 
        simulation_type: SimulationType, 
        duration: int = 300,
        step: float = 0.1
    ) -> AsyncIterator[DroneData]:
        """Start a drone simulation and yield drone data"""
        logger.info(f"Starting simulation: {simulation_type.value}, duration: {duration}s, step: {step}s")
        
        # Set current simulation parameters
        self.current_simulation_type = simulation_type
        self.start_time = datetime.now()
        self.duration = duration
        
        data_count = 0
        start_time = time.time()
        
        try:
            # Start simulation and yield drone data
            logger.debug("Initializing AirSim adapter simulation")
            async for drone_data in self.airsim_adapter.start_simulation(
                simulation_type=simulation_type,
                duration=duration,
                step=step
            ):
                data_count += 1
                yield drone_data
                
                # Log progress at intervals
                if data_count % 20 == 0:  # Log every 20th data point
                    elapsed = time.time() - start_time
                    remaining = max(0, duration - elapsed)
                    logger.info(f"Simulation progress: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining, {data_count} points processed")
                
                # Check if simulation should end
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed >= duration:
                    logger.info(f"Simulation duration reached ({duration}s), stopping")
                    await self.stop_simulation()
                    break
                    
        except asyncio.CancelledError:
            logger.info("Simulation cancelled")
            await self.stop_simulation()
            raise
            
        except Exception as e:
            logger.exception(f"Error in simulation: {str(e)}")
            await self.stop_simulation()
            raise
            
        finally:
            # Log simulation statistics
            total_elapsed = time.time() - start_time
            logger.info(f"Simulation finished: {data_count} data points in {total_elapsed:.2f}s ({data_count/total_elapsed:.2f} points/sec)")
    
    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        if self.current_simulation_type:
            logger.info(f"Stopping simulation type: {self.current_simulation_type.value}")
        else:
            logger.info("No active simulation to stop")
            
        await self.airsim_adapter.stop_simulation()
        
        # Calculate duration if simulation was running
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Simulation ran for {elapsed:.2f} seconds")
            
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
            
            logger.debug(f"Simulation status: elapsed={elapsed:.2f}s, remaining={status['remaining_time']:.2f}s")
            
        return status