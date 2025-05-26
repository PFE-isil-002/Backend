import logging
import asyncio
import math
from typing import Dict, Any, Optional, AsyncIterator, List, Tuple
from datetime import datetime, timedelta
import time

from ..domain.entities import SimulationType, DroneData
from ..domain.usecases import SimulationUseCase
from ..infra.airsim_adapter import AirSimAdapter

logger = logging.getLogger("app.simulation_service")


class SimulationService(SimulationUseCase):
    """Implementation of the SimulationUseCase with custom flight path support"""

    def __init__(self):
        self.airsim_adapter = AirSimAdapter()
        self.current_simulation_type: Optional[SimulationType] = None
        self.start_time: Optional[datetime] = None
        self.duration: int = 300
        self.flight_params: Dict[str, Any] = {}
        logger.info("SimulationService initialized")

    def generate_flight_path(
        self,
        start_point: Dict[str, float],
        end_point: Dict[str, float],
        waypoints: Optional[List[Dict[str, float]]] = None,
        path_type: str = "direct"
    ) -> List[Tuple[float, float, float]]:
        """Generate a flight path between start and end points"""
        if waypoints:
            path = [(start_point['x'], start_point['y'], start_point['z'])]
            for wp in waypoints:
                path.append((wp['x'], wp['y'], wp['z']))
            path.append((end_point['x'], end_point['y'], end_point['z']))
            logger.info(f"Generated path with {len(waypoints)} custom waypoints")
            return path

        if path_type == "direct":
            num_points = 5
            path = []
            for i in range(num_points + 1):
                t = i / num_points
                x = start_point['x'] + t * (end_point['x'] - start_point['x'])
                y = start_point['y'] + t * (end_point['y'] - start_point['y'])
                z = start_point['z'] + t * (end_point['z'] - start_point['z'])
                path.append((x, y, z))
            logger.info(f"Generated direct path with {len(path)} points")
            return path

        elif path_type == "curved":
            num_points = 10
            path = []
            mid_x = (start_point['x'] + end_point['x']) / 2
            mid_y = (start_point['y'] + end_point['y']) / 2
            dx = end_point['x'] - start_point['x']
            dy = end_point['y'] - start_point['y']
            length = math.sqrt(dx**2 + dy**2)

            if length > 0:
                offset_x = -dy / length * length * 0.2
                offset_y = dx / length * length * 0.2
                mid_x += offset_x
                mid_y += offset_y

            for i in range(num_points + 1):
                t = i / num_points
                x = (1 - t)**2 * start_point['x'] + 2 * (1 - t) * t * mid_x + t**2 * end_point['x']
                y = (1 - t)**2 * start_point['y'] + 2 * (1 - t) * t * mid_y + t**2 * end_point['y']
                z = start_point['z'] + t * (end_point['z'] - start_point['z'])
                path.append((x, y, z))

            logger.info(f"Generated curved path with {len(path)} points")
            return path

        elif path_type == "zigzag":
            num_segments = 6
            path = [(start_point['x'], start_point['y'], start_point['z'])]
            for i in range(1, num_segments):
                t = i / num_segments
                x = start_point['x'] + t * (end_point['x'] - start_point['x'])
                y = start_point['y'] + t * (end_point['y'] - start_point['y'])
                y += 3 * math.sin(t * math.pi * 4)
                z = start_point['z'] + t * (end_point['z'] - start_point['z'])
                path.append((x, y, z))
            path.append((end_point['x'], end_point['y'], end_point['z']))
            logger.info(f"Generated zigzag path with {len(path)} points")
            return path

        return self.generate_flight_path(start_point, end_point, path_type="direct")

    async def start_simulation_with_path(
        self,
        simulation_type: SimulationType,
        duration: int = 300,
        step: float = 0.1,
        start_point: Optional[Dict[str, float]] = None,
        end_point: Optional[Dict[str, float]] = None,
        waypoints: Optional[List[Dict[str, float]]] = None,
        velocity: float = 5.0,
        path_type: str = "direct"
    ) -> AsyncIterator[DroneData]:
        """Start a drone simulation with custom flight path"""
        self.flight_params = {
            "start_point": start_point,
            "end_point": end_point,
            "waypoints": waypoints,
            "velocity": velocity,
            "path_type": path_type
        }

        if start_point and end_point:
            logger.info(f"Starting custom path simulation: {simulation_type.value}")
            logger.info(
                f"Path: ({start_point['x']:.1f}, {start_point['y']:.1f}) → ({end_point['x']:.1f}, {end_point['y']:.1f})")
            logger.info(f"Parameters: duration={duration}s, step={step}s, velocity={velocity}m/s")
            flight_path = self.generate_flight_path(start_point, end_point, waypoints, path_type)

            async for drone_data in self.airsim_adapter.start_simulation_with_path(
                simulation_type=simulation_type,
                duration=duration,
                step=step,
                velocity=velocity,
                start_point=start_point,
                end_point=end_point,
                waypoints=waypoints,
                flight_path=flight_path
            ):
                yield drone_data
        else:
            logger.info(f"Starting default simulation: {simulation_type.value}")
            async for drone_data in self.airsim_adapter.start_simulation(
                simulation_type=simulation_type,
                duration=duration,
                step=step,
                velocity=velocity,
                start_point=start_point,
                end_point=end_point,
                waypoints=waypoints
            ):
                yield drone_data

    async def start_simulation(
        self,
        simulation_type: SimulationType,
        duration: int = 300,
        step: float = 0.1,
        velocity: float = 5.0,
        start_point: Optional[Dict[str, float]] = None,
        end_point: Optional[Dict[str, float]] = None,
        waypoints: Optional[List[Dict[str, float]]] = None
    ) -> AsyncIterator[DroneData]:
        """Unified simulation entrypoint with support for custom path"""
        # The logic here is fine, it will call start_simulation_with_path if points are provided
        # or start_simulation (default) if not. The AirSimAdapter now handles None for points.
        if start_point and end_point:
            async for drone_data in self.start_simulation_with_path(
                simulation_type=simulation_type,
                duration=duration,
                step=step,
                start_point=start_point,
                end_point=end_point,
                waypoints=waypoints,
                velocity=velocity
            ):
                yield drone_data
        else:
            async for drone_data in self.airsim_adapter.start_simulation(
                simulation_type=simulation_type,
                duration=duration,
                step=step,
                velocity=velocity,
                start_point=start_point,
                end_point=end_point,
                waypoints=waypoints
            ):
                yield drone_data

    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        if self.current_simulation_type:
            logger.info(f"Stopping simulation type: {self.current_simulation_type.value}")
        else:
            logger.info("No active simulation to stop")

        await self.airsim_adapter.stop_simulation()

        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Simulation ran for {elapsed:.2f} seconds")

        self.current_simulation_type = None
        self.flight_params = {}

    def get_simulation_status(self) -> Dict[str, Any]:
        """Get the current status of simulation"""
        running = self.current_simulation_type is not None

        status = {
            "running": running,
            "simulation_type": self.current_simulation_type.value if running else None,
            "elapsed_time": 0.0,
            "remaining_time": 0.0,
            "flight_params": self.flight_params if running else {}
        }

        if running and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            status["elapsed_time"] = elapsed
            status["remaining_time"] = max(0, self.duration - elapsed)
            logger.debug(f"Simulation status: elapsed={elapsed:.2f}s, remaining={status['remaining_time']:.2f}")
        return status
