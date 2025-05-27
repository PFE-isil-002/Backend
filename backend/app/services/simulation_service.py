import logging
import asyncio
import math
from typing import Dict, Any, Optional, AsyncIterator, List, Tuple
from datetime import datetime, timedelta
import time

from ..domain.entities import AuthenticationRequest, AuthenticationResponse, OutsiderDroneData, SimulationType, DroneData
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
        self.ai_model = None
        self.airsim_adapter.set_authentication_callback(self._authenticate_outsider_drone)
        logger.info("SimulationService initialized")

    def set_ai_model(self, ai_model):
        """Inject AI model for outsider detection"""
        self.ai_model = ai_model
        logger.info("AI model set for outsider authentication")

    async def _authenticate_outsider_drone(
        self,
        auth_request: AuthenticationRequest,
        outsider_data: OutsiderDroneData
    ) -> AuthenticationResponse:
        """Authenticate outsider drone using AI model"""

        logger.info(" AUTHENTICATING OUTSIDER DRONE")
        logger.info(f"   Analyzing flight history of {len(auth_request.flight_history)} waypoints")

        # Yield pending status
        if self.callback_for_outsider_status:
            await self.callback_for_outsider_status({
                "status": "pending",
                "drone_id": auth_request.drone_id,
                "outsider_telemetry": outsider_data.to_dict() # Assuming to_dict() method exists
            })

        try:
            if self.ai_model is None:
                logger.warning("  No AI model available - defaulting to OUTSIDER classification")
                is_outsider = True
                confidence = 0.95
            else:
                # Convert outsider data to feature vector for AI model
                feature_vector = outsider_data.to_feature_vector()
                logger.info(f"   Feature vector size: {len(feature_vector)}")

                # Use AI model to predict if this is an outsider
                # This assumes your AI model has a predict method that returns (prediction, confidence)
                prediction_result = await self._run_ai_prediction(feature_vector)
                is_outsider = prediction_result.get('is_outsider', True)
                confidence = prediction_result.get('confidence', 0.95)

                logger.info(f" AI Model Prediction:")
                logger.info(f"   - Classification: {'OUTSIDER' if is_outsider else 'LEGITIMATE'}")
                logger.info(f"   - Confidence: {confidence:.3f}")

            # Create response
            auth_response = AuthenticationResponse(
                drone_id=auth_request.drone_id,
                is_authenticated=not is_outsider,
                is_outsider=is_outsider,
                confidence=confidence,
                response_timestamp=datetime.utcnow(),
                communication_blocked=is_outsider  # Always block if classified as outsider
            )

            # Log the decision
            if is_outsider:
                logger.error(f" OUTSIDER DETECTED - Communication BLOCKED")
                logger.error(f"   Drone {auth_request.drone_id} flagged as OUTSIDER with {confidence:.1%} confidence")
                if self.callback_for_outsider_status:
                    await self.callback_for_outsider_status({
                        "status": "blocked",
                        "drone_id": auth_request.drone_id,
                        "confidence": confidence,
                        "outsider_telemetry": outsider_data.to_dict()
                    })
            else:
                logger.info(f" LEGITIMATE DRONE - Communication ALLOWED")
                logger.info(f"   Drone {auth_request.drone_id} authenticated with {confidence:.1%} confidence")
                if self.callback_for_outsider_status:
                    await self.callback_for_outsider_status({
                        "status": "authenticated",
                        "drone_id": auth_request.drone_id,
                        "confidence": confidence,
                        "outsider_telemetry": outsider_data.to_dict()
                    })

            return auth_response

        except Exception as e:
            logger.error(f" Authentication error: {str(e)}")
            # Default to blocking on error
            if self.callback_for_outsider_status:
                await self.callback_for_outsider_status({
                    "status": "error",
                    "drone_id": auth_request.drone_id,
                    "message": str(e),
                    "outsider_telemetry": outsider_data.to_dict()
                })
            return AuthenticationResponse(
                drone_id=auth_request.drone_id,
                is_authenticated=False,
                is_outsider=True,
                confidence=1.0,
                response_timestamp=datetime.utcnow(),
                communication_blocked=True
            )

    async def _run_ai_prediction(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Run AI model prediction on feature vector"""
        try:
            if hasattr(self.ai_model, 'predict_async'):
                # If model supports async prediction
                result = await self.ai_model.predict_async(feature_vector)
            elif hasattr(self.ai_model, 'predict'):
                # Synchronous prediction
                result = self.ai_model.predict([feature_vector])  # Most models expect 2D array
                if hasattr(result, '__len__') and len(result) > 0:
                    result = result[0]  # Get first result
            else:
                raise AttributeError("AI model does not have predict method")

            # Parse result based on your model's output format
            # This is a generic implementation - adjust based on your specific AI model
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'item'):  # numpy scalar
                # Assume binary classification where > 0.5 means outsider
                confidence = float(result.item())
                is_outsider = confidence > 0.5
                return {'is_outsider': is_outsider, 'confidence': confidence}
            else:
                # Default interpretation
                confidence = float(result) if isinstance(result, (int, float)) else 0.95
                is_outsider = confidence > 0.5
                return {'is_outsider': is_outsider, 'confidence': confidence}

        except Exception as e:
            logger.error(f"AI model prediction failed: {str(e)}")
            # Default to outsider with high confidence on error
            return {'is_outsider': True, 'confidence': 0.95}

    def get_outsider_status(self) -> Dict[str, Any]:
        """Get current outsider drone status"""
        if hasattr(self.airsim_adapter, 'outsider_drone_active'):
            return {
                "outsider_active": self.airsim_adapter.outsider_drone_active,
                "authentication_attempted": self.airsim_adapter.outsider_authentication_attempted,
                "communication_blocked": self.airsim_adapter.outsider_communication_blocked,
                "appearance_time": self.airsim_adapter.outsider_appearance_time,
                "outsider_data": self.airsim_adapter.outsider_drone_data
            }
        return {"outsider_active": False}

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
        self.current_simulation_type = simulation_type # Set current simulation type

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
        self.current_simulation_type = simulation_type # Set current simulation type
        self.start_time = datetime.utcnow() # Record start time
        self.duration = duration # Store duration

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
        """Get the current status of simulation with outsider information"""
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

        # Add outsider status for outsider simulations
        if running and self.current_simulation_type == SimulationType.OUTSIDER:
            status["outsider_status"] = self.get_outsider_status()

        return status

    def set_outsider_status_callback(self, callback):
        """Set a callback function to send outsider drone status to the client."""
        self.callback_for_outsider_status = callback