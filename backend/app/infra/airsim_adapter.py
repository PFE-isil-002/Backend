import asyncio
import subprocess
import logging
from typing import AsyncIterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import math
import time
from datetime import datetime, timedelta

from ..domain.entities import AuthenticationRequest, DroneData, OutsiderDroneData, SimulationType, DronePosition
from ..core.config import settings

logger = logging.getLogger(__name__)

class AirSimAdapter:
    """Adapter for AirSim drone simulation"""

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.simulation_active = False
        self.outsider_drone_active = False
        self.outsider_authentication_attempted = False
        self.outsider_communication_blocked = False
        self.outsider_appearance_time = None
        self.outsider_drone_data = None
        self.authentication_callback = None
        
    def set_authentication_callback(self, callback):
        """Set callback function for AI model authentication"""
        self.authentication_callback = callback

    def _generate_outsider_flight_history(self, current_time: float, outsider_current_pos: Dict[str, float], outsider_start_pos: Dict[str, float]) -> List[DronePosition]:
            """Generate suspicious flight history for outsider drone up to its current position"""
            history = []
    
            # Generate 15-20 historical positions showing erratic/suspicious behavior
            num_history_points = 18
            base_time = current_time - (num_history_points * 2.0)  # 2 seconds apart
    
            for i in range(num_history_points):
                timestamp = datetime.utcnow() - timedelta(seconds=(num_history_points - i) * 2)
        
                # Progress from start to current position (not just start position)
                progress = i / num_history_points
        
                # Calculate base trajectory from start to current position
                base_x = outsider_start_pos['x'] + progress * (outsider_current_pos['x'] - outsider_start_pos['x'])
                base_y = outsider_start_pos['y'] + progress * (outsider_current_pos['y'] - outsider_start_pos['y'])
                base_z = outsider_start_pos['z'] + progress * (outsider_current_pos['z'] - outsider_start_pos['z'])
        
                # Add suspicious behavior patterns along the trajectory
                if progress < 0.3:
                    # Phase 1: Erratic movement near start position
                    erratic_radius = 25.0 * (1 - progress)
                    angle = progress * math.pi * 8  # Multiple circles
                    x = base_x + erratic_radius * math.cos(angle)
                    y = base_y + erratic_radius * math.sin(angle)
                    z = base_z + 5.0 * math.sin(progress * math.pi * 6)
            
                elif progress < 0.7:
                    # Phase 2: Rapid approach with sudden direction changes
                    approach_progress = (progress - 0.3) / 0.4
            
                    # Add zigzag pattern while moving toward current position
                    if int(approach_progress * 10) % 2 == 0:
                        x = base_x + 15.0 * math.sin(approach_progress * math.pi * 4)
                        y = base_y + 12.0 * math.cos(approach_progress * math.pi * 3)
                        z = base_z
                    else:
                        x = base_x - 10.0 * math.sin(approach_progress * math.pi * 5)
                        y = base_y - 8.0 * math.cos(approach_progress * math.pi * 4)
                        z = base_z # This was missing in the original code
                
                else:
                    # Phase 3: Final approach to current position - more direct but still suspicious
                    final_progress = (progress - 0.7) / 0.3
            
                    # Move closer to actual current position with small evasive maneuvers
                    x = base_x + 3.0 * math.sin(final_progress * math.pi * 12)
                    y = base_y + 2.0 * math.cos(final_progress * math.pi * 10)
                    z = base_z
        
                history.append(DronePosition(x=x, y=y, z=z, timestamp=timestamp))
    
            # Ensure the last position in history is very close to the current position
            # This represents the most recent known position before authentication request
            recent_timestamp = datetime.utcnow() - timedelta(seconds=1)
            recent_pos = DronePosition(
                x=outsider_current_pos['x'] + 1.0 * math.sin(current_time),  # Small deviation
                y=outsider_current_pos['y'] + 0.8 * math.cos(current_time),
                z=outsider_current_pos['z'] + 0.5 * math.sin(current_time * 2),
                timestamp=recent_timestamp
            )
            history.append(recent_pos)
    
            logger.info(f"Generated outsider flight history with {len(history)} suspicious waypoints ending near current position")
            logger.info(f"History start: ({history[0].x:.2f}, {history[0].y:.2f}, {history[0].z:.2f})")
            logger.info(f"History end: ({history[-1].x:.2f}, {history[-1].y:.2f}, {history[-1].z:.2f})")
            logger.info(f"Current pos: ({outsider_current_pos['x']:.2f}, {outsider_current_pos['y']:.2f}, {outsider_current_pos['z']:.2f})")
    
            return history

    def _should_trigger_outsider_authentication(self, current_time: float, total_duration: float) -> bool:
        """Determine if outsider drone should appear and request authentication"""
        if self.outsider_authentication_attempted:
            return False
            
        # Trigger between 30% and 60% of flight time, randomly
        min_trigger_time = total_duration * 0.3
        max_trigger_time = total_duration * 0.6
        
        # Use a random seed based on current time for consistency
        random_seed = int(current_time * 1000) % 1000
        trigger_time = min_trigger_time + (random_seed / 1000.0) * (max_trigger_time - min_trigger_time)
        
        return current_time >= trigger_time

    async def _handle_outsider_authentication(
        self, 
        current_time: float, 
        main_drone_pos: Tuple[float, float, float],
        outsider_pos: Tuple[float, float, float],
        outsider_start_pos: Dict[str, float]
    ) -> Optional[OutsiderDroneData]:
        """Handle outsider drone authentication process"""
        
        if not self.outsider_authentication_attempted:
            self.outsider_authentication_attempted = True
            self.outsider_appearance_time = current_time
            
            # Convert outsider_pos tuple to a dictionary for _generate_outsider_flight_history
            outsider_current_pos_dict = {"x": outsider_pos[0], "y": outsider_pos[1], "z": outsider_pos[2]}

            # Generate outsider drone data with suspicious flight history
            flight_history = self._generate_outsider_flight_history(current_time, outsider_current_pos_dict, outsider_start_pos)
            
            outsider_data = OutsiderDroneData(
                drone_id=f"OUTSIDER_{int(current_time * 100)}",
                position=DronePosition(x=outsider_pos[0], y=outsider_pos[1], z=outsider_pos[2], timestamp=datetime.utcnow()),
                velocity={"x": 8.5, "y": -6.2, "z": 1.1},
                acceleration={"x": 2.1, "y": -1.8, "z": 0.5},
                orientation={"pitch": 12.0, "roll": -8.0, "yaw": 245.0},
                angular_velocity={"x": 0.3, "y": -0.2, "z": 0.8},
                battery=85.0,
                signal_strength=-72.0,
                packet_loss=0.15,
                latency=0.08,
                flight_history=flight_history,
                authentication_timestamp=datetime.utcnow()
            )
            
            # Create authentication request
            auth_request = AuthenticationRequest(
                drone_id=outsider_data.drone_id,
                flight_history=flight_history,
                request_timestamp=datetime.utcnow(),
                source_position=outsider_data.position
            )
            
            logger.warning("="*80)
            logger.warning(" OUTSIDER DRONE DETECTED!")
            logger.warning(f"   Drone ID: {outsider_data.drone_id}")
            logger.warning(f"   Position: ({outsider_pos[0]:.2f}, {outsider_pos[1]:.2f}, {outsider_pos[2]:.2f})")
            logger.warning(f"   Distance from main drone: {math.dist(main_drone_pos, outsider_pos):.2f} units")
            logger.warning("="*80)
            
            logger.info(" AUTHENTICATION REQUEST INITIATED")
            logger.info(f" Request Details:")
            logger.info(f"   - Drone ID: {auth_request.drone_id}")
            logger.info(f"   - Flight History Points: {len(auth_request.flight_history)}")
            logger.info(f"   - Request Time: {auth_request.request_timestamp}")
            
            # Send to AI model for authentication if callback is available
            if self.authentication_callback:
                try:
                    logger.info(" Sending authentication request to AI model...")
                    auth_response = await self.authentication_callback(auth_request, outsider_data)
                    
                    logger.info(" AUTHENTICATION RESPONSE RECEIVED")
                    logger.info(f"   - Is Authenticated: {auth_response.is_authenticated}")
                    logger.info(f"   - Is Outsider: {auth_response.is_outsider}")
                    logger.info(f"   - Confidence: {auth_response.confidence:.2f}")
                    logger.info(f"   - Communication Blocked: {auth_response.communication_blocked}")
                    
                    if auth_response.is_outsider:
                        self.outsider_communication_blocked = True
                        logger.error(" COMMUNICATION BLOCKED - OUTSIDER DRONE DETECTED!")
                        logger.error("   AI Model classified this drone as an OUTSIDER")
                        logger.error(f"   Detection confidence: {auth_response.confidence:.2f}")
                    else:
                        logger.info(" Authentication successful - Communication allowed")
                        
                except Exception as e:
                    logger.error(f" Authentication failed: {str(e)}")
                    # Default to blocking communication on error
                    self.outsider_communication_blocked = True
                    logger.error(" COMMUNICATION BLOCKED - Authentication system error")
            else:
                logger.warning("  No authentication callback available - Blocking by default")
                self.outsider_communication_blocked = True
            
            self.outsider_drone_data = outsider_data
            self.outsider_drone_active = True
            
            return outsider_data
        
        return self.outsider_drone_data

    async def start_simulation(
        self,
        simulation_type: SimulationType,
        duration: float,
        step: float,
        velocity: float = 5.0,
        start_point: Optional[Dict[str, float]] = None,
        end_point: Optional[Dict[str, float]] = None,
        waypoints: Optional[list] = None
    ) -> AsyncIterator[DroneData]:
        """Start the simulation and yield drone data"""

        logger.info(f"Starting simulation: {simulation_type}, duration: {duration}s, step: {step}s")

        effective_start_point = start_point or {"x": 0.0, "y": 0.0, "z": -5.0}
        effective_end_point   = end_point   or {"x": 50.0, "y": 25.0, "z": -5.0}
        effective_waypoints   = waypoints   or []

        try:
            async for data in self.start_simulation_with_path(
                simulation_type=simulation_type,
                duration=duration,
                step=step,
                velocity=velocity,
                start_point=effective_start_point,
                end_point=effective_end_point,
                waypoints=effective_waypoints
            ):
                yield data
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            raise
        finally:
            await self.stop_simulation()

    async def start_simulation_with_path(
        self,
        simulation_type: SimulationType,
        duration: float,
        step: float,
        velocity: float,
        start_point: Optional[Dict[str, float]] = None,
        end_point: Optional[Dict[str, float]] = None,
        waypoints: Optional[list] = None,
        flight_path: Optional[List[Tuple[float, float, float]]] = None
    ) -> AsyncIterator[DroneData]:
        """Start simulation with specific path parameters"""

        start_point = start_point or {"x": 0.0, "y": 0.0, "z": -5.0}
        end_point   = end_point   or {"x": 50.0, "y": 25.0, "z": -5.0}
        waypoints   = waypoints   or []

        try:
            if not flight_path:
                flight_path = self._generate_flight_path_with_21_waypoints(start_point, end_point, waypoints)
            logger.info(f"Created flight path with exactly {len(flight_path)} waypoints")

            async for data in self._simulate_flight_data(
                simulation_type=simulation_type,
                flight_path=flight_path,
                duration=duration,
                step=step,
                velocity=velocity,
                start_point=start_point,
                end_point=end_point
            ):
                yield data
        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}")
            raise

    def _generate_outsider_start_position(self, main_start: Dict[str, float], main_end: Dict[str, float]) -> Dict[str, float]:
        """Generate a starting position for the outsider drone that's significantly different from main drone"""
        
        # Calculate the vector from start to end for the main drone
        dx = main_end['x'] - main_start['x']
        dy = main_end['y'] - main_start['y']
        perp_distance = 140.0  
        
        # Calculate perpendicular vector
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            perp_x = -dy / length * perp_distance
            perp_y = dx / length * perp_distance
        else:
            perp_x = perp_distance
            perp_y = 0
        
        # Start the outsider drone at the midpoint of main path plus perpendicular offset
        mid_x = (main_start['x'] + main_end['x']) / 2
        mid_y = (main_start['y'] + main_end['y']) / 2
        
        outsider_start = {
            'x': mid_x + perp_x,
            'y': mid_y + perp_y,
            'z': main_start['z'] - 8.0  # Start at different altitude
        }
        
        logger.info(f"Outsider drone starting at: ({outsider_start['x']:.2f}, {outsider_start['y']:.2f}, {outsider_start['z']:.2f})")
        return outsider_start

    def _generate_outsider_convergence_path(
        self, 
        outsider_start: Dict[str, float], 
        main_path: List[Tuple[float, float, float]],
        convergence_point_ratio: float = 0.6
    ) -> List[Tuple[float, float, float]]:
        """Generate flight path for outsider drone that converges with main drone"""
        
        # Find convergence point on main path (60% through the flight by default)
        convergence_index = int(len(main_path) * convergence_point_ratio)
        convergence_point = main_path[convergence_index]
        
        # Generate path from outsider start to convergence point
        outsider_path = []
        
        # First phase: approach convergence point (curved path for realism)
        approach_points = 12  # Points to reach convergence
        
        start_tuple = (outsider_start['x'], outsider_start['y'], outsider_start['z'])
        
        for i in range(approach_points):
            progress = i / (approach_points - 1)
            
            # Create a curved approach rather than straight line
            curve_factor = math.sin(progress * math.pi) * 0.3  # Adds curvature
            
            # Linear interpolation with curve
            x = start_tuple[0] + progress * (convergence_point[0] - start_tuple[0])
            y = start_tuple[1] + progress * (convergence_point[1] - start_tuple[1])
            z = start_tuple[2] + progress * (convergence_point[2] - start_tuple[2])
            
            # Add curve deviation
            perp_offset = curve_factor * 15.0
            x += perp_offset * math.cos(progress * math.pi)
            y += perp_offset * math.sin(progress * math.pi)
            
            outsider_path.append((x, y, z))
        
        # Second phase: follow main path from convergence point to end
        remaining_main_path = main_path[convergence_index:]
        outsider_path.extend(remaining_main_path)
        
        # Ensure exactly 21 waypoints
        if len(outsider_path) > 21:
            # Sample 21 points evenly
            indices = [int(i * (len(outsider_path) - 1) / 20) for i in range(21)]
            outsider_path = [outsider_path[i] for i in indices]
        elif len(outsider_path) < 21:
            # Interpolate more points
            while len(outsider_path) < 21:
                new_path = [outsider_path[0]]
                for i in range(len(outsider_path) - 1):
                    new_path.append(outsider_path[i + 1])
                    if len(new_path) < 21:
                        # Add midpoint
                        mid_x = (outsider_path[i][0] + outsider_path[i + 1][0]) / 2
                        mid_y = (outsider_path[i][1] + outsider_path[i + 1][1]) / 2
                        mid_z = (outsider_path[i][2] + outsider_path[i + 1][2]) / 2
                        new_path.insert(-1, (mid_x, mid_y, mid_z))
                outsider_path = new_path[:21]
        
        logger.info(f"Generated outsider convergence path with {len(outsider_path)} waypoints")
        logger.info(f"Convergence occurs at waypoint {min(approach_points, 21)} - point ({convergence_point[0]:.2f}, {convergence_point[1]:.2f}, {convergence_point[2]:.2f})")
        
        return outsider_path

    def _generate_flight_path_with_21_waypoints(
        self,
        start_point: Dict[str, float],
        end_point: Dict[str, float],
        waypoints: list
    ) -> List[Tuple[float, float, float]]:
        """Generate flight path with exactly 21 waypoints (including start and end)"""
        
        # Always start with the start point
        path = [(start_point['x'], start_point['y'], start_point['z'])]
        
        # Add provided waypoints if any (up to 19 to leave room for interpolation)
        provided_waypoints = []
        for wp in waypoints[:19]:  # Limit to 19 to ensure we can fit start and end
            if isinstance(wp, dict):
                provided_waypoints.append((wp['x'], wp['y'], wp['z']))
        
        # If we have provided waypoints, add them
        if provided_waypoints:
            path.extend(provided_waypoints)
        
        # Calculate how many more waypoints we need to reach exactly 21
        current_count = len(path)  # This includes start point and any provided waypoints
        needed_waypoints = 21 - current_count - 1  # -1 because we'll add end point
        
        # Generate intermediate waypoints between last current point and end point
        if needed_waypoints > 0:
            last_point = path[-1]
            end_tuple = (end_point['x'], end_point['y'], end_point['z'])
            
            # Generate evenly spaced waypoints between last point and end
            for i in range(1, needed_waypoints + 1):
                fraction = i / (needed_waypoints + 1)
                x = last_point[0] + fraction * (end_tuple[0] - last_point[0])
                y = last_point[1] + fraction * (end_tuple[1] - last_point[1])
                z = last_point[2] + fraction * (end_tuple[2] - last_point[2])
                path.append((x, y, z))
        
        # Always end with the end point
        path.append((end_point['x'], end_point['y'], end_point['z']))
        
        # Ensure we have exactly 21 waypoints
        if len(path) != 21:
            logger.warning(f"Path has {len(path)} waypoints, adjusting to exactly 21")
            if len(path) > 21:
                # Keep start, end, and evenly distribute the middle points
                start = path[0]
                end = path[-1]
                # Take 19 evenly spaced points from the middle
                middle_indices = [int(i * (len(path) - 2) / 18) + 1 for i in range(19)]
                path = [start] + [path[i] for i in middle_indices] + [end]
            elif len(path) < 21:
                # Interpolate more points
                while len(path) < 21:
                    new_path = [path[0]]
                    for i in range(len(path) - 1):
                        new_path.append(path[i + 1])
                        if len(new_path) < 21:
                            # Add midpoint
                            mid_x = (path[i][0] + path[i + 1][0]) / 2
                            mid_y = (path[i][1] + path[i + 1][1]) / 2
                            mid_z = (path[i][2] + path[i + 1][2]) / 2
                            new_path.insert(-1, (mid_x, mid_y, mid_z))
                    path = new_path[:21]  # Ensure exactly 21
        
        logger.info(f"Generated flight path with exactly {len(path)} waypoints")
        return path

    def _generate_flight_path(
        self,
        start_point: Dict[str, float],
        end_point: Dict[str, float],
        waypoints: list
    ) -> List[Tuple[float, float, float]]:
        """Legacy method - now calls the 21-waypoint version"""
        return self._generate_flight_path_with_21_waypoints(start_point, end_point, waypoints)

    async def _simulate_flight_data(
        self,
        simulation_type: SimulationType,
        flight_path: List[Tuple[float, float, float]],
        duration: float,
        step: float,
        velocity: float,
        start_point: Dict[str, float],
        end_point: Dict[str, float]
    ) -> AsyncIterator[DroneData]:
        """Simulate drone flight data along the path with enhanced MITM hijack logic and outsider drone"""

        self.simulation_active = True
        current_time = 0.0

        # MITM attack state - Enhanced for more noticeable deviation
        mitm_active = False
        hijack_start = None
        hijack_duration = 25.0  
        orig_pos = None
        hijack_target = None  

        # Outsider drone state
        outsider_path = None
        outsider_start_pos = None
        
        if simulation_type == SimulationType.OUTSIDER:
            outsider_start_pos = self._generate_outsider_start_position(start_point, end_point)
            outsider_path = self._generate_outsider_convergence_path(outsider_start_pos, flight_path)

        # Calculate total distance and estimated flight time
        total_dist = sum(
            math.dist(flight_path[i], flight_path[i+1])
            for i in range(len(flight_path)-1)
        )
        
        # Much slower effective velocity
        effective_velocity = velocity * 0.4  
        estimated_flight_time = total_dist / effective_velocity if effective_velocity > 0 else duration
        
        logger.info(f"Total distance: {total_dist:.2f}, Estimated flight time: {estimated_flight_time:.2f}s")

        try:
            while self.simulation_active:
                # Calculate current position along path
                dist_cov = current_time * effective_velocity
                
                x = y = z = 0.0
                reached_end = False

                if total_dist > 0:
                    if dist_cov >= total_dist:
                        # Reached the end point
                        x, y, z = flight_path[-1]
                        reached_end = True
                        logger.info(f"Reached endpoint at time {current_time:.2f}s")
                    else:
                        # Find current position along path
                        cum = 0.0
                        for idx in range(len(flight_path)-1):
                            p1, p2 = flight_path[idx], flight_path[idx+1]
                            seg = math.dist(p1, p2)
                            if dist_cov <= cum + seg:
                                frac = (dist_cov - cum) / seg if seg > 0 else 0
                                x = p1[0] + frac * (p2[0] - p1[0])
                                y = p1[1] + frac * (p2[1] - p1[1])
                                z = p1[2] + frac * (p2[2] - p1[2])
                                break
                            cum += seg
                        else:
                            x, y, z = flight_path[-1]
                            reached_end = True
                elif flight_path:
                    x, y, z = flight_path[0]

                norm_x, norm_y, norm_z = x, y, z

                # Handle outsider drone simulation
                if simulation_type == SimulationType.OUTSIDER and outsider_path:
            
                    outsider_total_dist = sum(
                    math.dist(outsider_path[i], outsider_path[i+1])
                    for i in range(len(outsider_path)-1)
                     )
            
                    outsider_dist_cov = current_time * effective_velocity * 1.1  
            
                    if outsider_total_dist > 0 and outsider_dist_cov < outsider_total_dist:
                     
                        cum = 0.0
                        for idx in range(len(outsider_path)-1):
                            p1, p2 = outsider_path[idx], outsider_path[idx+1]
                            seg = math.dist(p1, p2)
                            if outsider_dist_cov <= cum + seg:
                                frac = (outsider_dist_cov - cum) / seg if seg > 0 else 0
                                outsider_x = p1[0] + frac * (p2[0] - p1[0])
                                outsider_y = p1[1] + frac * (p2[1] - p1[1])
                                outsider_z = p1[2] + frac * (p2[2] - p1[2])
                                break
                            cum += seg
                        else:
                            outsider_x, outsider_y, outsider_z = outsider_path[-1]
                    else:
                        outsider_x, outsider_y, outsider_z = outsider_path[-1] if outsider_path else (x, y, z)
            
                    
                    if self._should_trigger_outsider_authentication(current_time, duration):
                        outsider_drone_data = await self._handle_outsider_authentication(
                            current_time, 
                            (x, y, z), 
                            (outsider_x, outsider_y, outsider_z),
                            outsider_start_pos
                        )
            
                    
                    distance_to_outsider = math.sqrt(
                    (x - outsider_x)**2 + (y - outsider_y)**2 + (z - outsider_z)**2
                    )
            
                    
                    if distance_to_outsider < 20.0:  
                        interference_factor = (20.0 - distance_to_outsider) / 20.0
                
                        
                        if self.outsider_drone_active:
                            logger.info(f" OUTSIDER INTERFERENCE: distance={distance_to_outsider:.2f}, factor={interference_factor:.2f}")
                            if self.outsider_communication_blocked:
                                logger.warning(" Communication with outsider BLOCKED")
                            else:
                                logger.info(" Communication with outsider ACTIVE")
                
                        
                        x += interference_factor * 8.0 * math.sin(current_time * 3.0)
                        y += interference_factor * 6.0 * math.cos(current_time * 2.5)
                        z += interference_factor * 3.0 * math.sin(current_time * 4.0)
                
                        
                        phase_shift = current_time * 2.0
                        x += interference_factor * 5.0 * math.cos(phase_shift)
                        y += interference_factor * 5.0 * math.sin(phase_shift)
                
               
                elif simulation_type == SimulationType.MITM and not reached_end:
                    prog = dist_cov / total_dist if total_dist > 0 else current_time / duration
                    
                    
                    if 0.25 <= prog <= 0.7 and not mitm_active and hijack_start is None:
                        mitm_active = True
                        hijack_start = current_time
                        orig_pos = (x, y, z)
                        
                        # Create a dramatic deviation target - move significantly off path
                        deviation_distance = 30.0  # Much larger deviation
                        angle = math.radians(45 + (current_time * 30) % 90)  # Varying angle
                        hijack_target = (
                            x + deviation_distance * math.cos(angle),
                            y + deviation_distance * math.sin(angle),
                            z
                        )
                        
                        logger.warning(f"MITM ATTACK INITIATED at {current_time:.2f}s")
                        logger.warning(f"Original position: ({x:.2f}, {y:.2f}, {z:.2f})")
                        logger.warning(f"Hijack target: ({hijack_target[0]:.2f}, {hijack_target[1]:.2f}, {hijack_target[2]:.2f})")
                    
                    if mitm_active:
                        elapsed = current_time - hijack_start
                        
                        if elapsed < hijack_duration:
                            # Phase 1: Move to hijack target (first 30% of hijack time)
                            if elapsed < hijack_duration * 0.3:
                                transition_progress = elapsed / (hijack_duration * 0.3)
                                # Smooth transition to hijack target
                                smooth_factor = 1 - math.exp(-transition_progress * 4)
                                x = orig_pos[0] + (hijack_target[0] - orig_pos[0]) * smooth_factor
                                y = orig_pos[1] + (hijack_target[1] - orig_pos[1]) * smooth_factor
                                z = orig_pos[2] + (hijack_target[2] - orig_pos[2]) * smooth_factor
                                
                                logger.debug(f"HIJACK PHASE 1 - Moving to target, progress: {transition_progress:.2f}")
                            
                            # Phase 2: Erratic behavior around hijack target (middle 40% of hijack time)
                            elif elapsed < hijack_duration * 0.7:
                                phase_elapsed = elapsed - (hijack_duration * 0.3)
                                phase_duration = hijack_duration * 0.4
                                
                                # Erratic circular and random movements around hijack target
                                erratic_radius = 15.0
                                circle_freq = 2.0
                                random_freq = 5.0
                                
                                circle_x = erratic_radius * math.cos(phase_elapsed * circle_freq)
                                circle_y = erratic_radius * math.sin(phase_elapsed * circle_freq)
                                
                                # Add random jitter
                                jitter_x = 8.0 * math.sin(phase_elapsed * random_freq * 1.3) * math.cos(phase_elapsed * 2.7)
                                jitter_y = 8.0 * math.cos(phase_elapsed * random_freq * 1.7) * math.sin(phase_elapsed * 3.1)
                                jitter_z = 5.0 * math.sin(phase_elapsed * random_freq * 2.3)
                                
                                x = hijack_target[0] + circle_x + jitter_x
                                y = hijack_target[1] + circle_y + jitter_y
                                z = hijack_target[2] + jitter_z
                                
                                logger.debug(f"HIJACK PHASE 2 - Erratic behavior at target")
                            
                            # Phase 3: Gradual return to normal path (last 30% of hijack time)
                            else:
                                recovery_elapsed = elapsed - (hijack_duration * 0.7)
                                recovery_duration = hijack_duration * 0.3
                                recovery_progress = recovery_elapsed / recovery_duration
                                
                                # Smooth transition back to normal path
                                smooth_recovery = 1 - math.exp(-recovery_progress * 3)
                                
                                current_hijack_x = hijack_target[0]
                                current_hijack_y = hijack_target[1]
                                current_hijack_z = hijack_target[2]
                                
                                x = current_hijack_x + (norm_x - current_hijack_x) * smooth_recovery
                                y = current_hijack_y + (norm_y - current_hijack_y) * smooth_recovery
                                z = current_hijack_z + (norm_z - current_hijack_z) * smooth_recovery
                                
                                logger.info(f"HIJACK PHASE 3 - Recovery progress: {recovery_progress:.2f}")
                        else:
                            # Hijack complete, return to normal flight
                            mitm_active = False
                            hijack_start = None
                            x, y, z = norm_x, norm_y, norm_z
                            logger.info("MITM ATTACK COMPLETE - Returned to normal flight path")
                
                # Add subtle noise when not being hijacked or interfered with
                if not (simulation_type == SimulationType.MITM and mitm_active) and simulation_type != SimulationType.OUTSIDER:
                    nf = self._get_noise_factor(simulation_type)
                    x += nf * math.sin(current_time * 2) * 0.3
                    y += nf * math.cos(current_time * 1.5) * 0.2

                pos = DronePosition(x=x, y=y, z=z, timestamp=datetime.utcnow())
                
                # Enhanced velocity/acceleration/orientation/signal data
                if simulation_type == SimulationType.MITM and mitm_active:
                    elapsed = current_time - hijack_start
                    
                    # More dramatic velocity changes during hijack
                    vmul = 3.0 + 2.0 * math.sin(elapsed * 4.0)
                    velocity_data = {
                        "x": vmul * velocity * math.cos(elapsed * 2.0),
                        "y": vmul * velocity * math.sin(elapsed * 2.5),
                        "z": velocity * 0.8 * math.sin(elapsed * 3.0)
                    }
                    
                    # More erratic acceleration
                    accel_data = {
                        "x": 5.0 * math.sin(elapsed * 3.0),
                        "y": 5.0 * math.cos(elapsed * 2.8),
                        "z": 3.0 * math.sin(elapsed * 4.0)
                    }
                    
                    # More dramatic orientation changes
                    orient = {
                        "roll": 45 * math.sin(elapsed * 2.2),
                        "pitch": 35 * math.cos(elapsed * 1.8),
                        "yaw": (elapsed * 90) % 360
                    }
                    
                    # Worse signal quality during attack
                    sig = -85 - abs(math.sin(elapsed * 2)) * 25
                    loss = 0.4 + abs(math.cos(elapsed * 1.5)) * 0.5
                    lat = 0.8 + abs(math.sin(elapsed * 2.2)) * 1.2
                    
                elif simulation_type == SimulationType.OUTSIDER:
                    # Enhanced telemetry during outsider interference
                    nf = self._get_noise_factor(simulation_type)
                    
                    # More erratic behavior when outsider is present
                    outsider_influence = 1.0
                    if outsider_path:
                        # Calculate current outsider position for influence
                        outsider_dist_cov = current_time * effective_velocity * 1.1
                        outsider_total_dist = sum(math.dist(outsider_path[i], outsider_path[i+1]) for i in range(len(outsider_path)-1))
                        if outsider_total_dist > 0 and outsider_dist_cov < outsider_total_dist:
                            # Find distance to outsider for influence calculation
                            distance_to_outsider = 15.0  # Approximate for telemetry effects
                            if distance_to_outsider < 20.0:
                                outsider_influence = 2.0 + (20.0 - distance_to_outsider) / 10.0
                    
                    velocity_data = {
                        "x": effective_velocity * math.cos(current_time * 0.1) * outsider_influence,
                        "y": effective_velocity * math.sin(current_time * 0.1) * outsider_influence,
                        "z": 0.3 * math.sin(current_time * 0.5) * outsider_influence
                    }
                    accel_data = {
                        "x": nf * outsider_influence * math.sin(current_time * 1.5),
                        "y": nf * outsider_influence * math.cos(current_time * 1.8),
                        "z": nf * 0.5 * math.sin(current_time * 2.2)
                    }
                    orient = {
                        "roll": 10 * math.sin(current_time * 0.8) * outsider_influence,
                        "pitch": 8 * math.cos(current_time * 0.6) * outsider_influence,
                        "yaw": (current_time * 15 * outsider_influence) % 360
                    }
                    sig = -60 - nf * 15 - (outsider_influence - 1) * 10
                    loss = nf * 0.1 + (outsider_influence - 1) * 0.05
                    lat = nf * 0.02 + (outsider_influence - 1) * 0.01
                    
                else:
                    nf = self._get_noise_factor(simulation_type)
                    velocity_data = {
                        "x": effective_velocity * math.cos(current_time * 0.1),
                        "y": effective_velocity * math.sin(current_time * 0.1),
                        "z": 0.1 * math.sin(current_time * 0.5)
                    }
                    accel_data = {
                        "x": 0.1 * math.sin(current_time),
                        "y": 0.1 * math.cos(current_time),
                        "z": 0.05 * math.sin(current_time * 2)
                    }
                    orient = {
                        "roll": 5 * math.sin(current_time * 0.3),
                        "pitch": 3 * math.cos(current_time * 0.4),
                        "yaw": current_time * 10 % 360
                    }
                    sig = -50 - nf * 10
                    loss = nf * 0.05
                    lat = nf * 0.01

                data = DroneData(
                    position=pos,
                    velocity=velocity_data,
                    acceleration=accel_data,
                    orientation=orient,
                    angular_velocity={
                        "x": 0.1 * math.sin(current_time),
                        "y": 0.1 * math.cos(current_time),
                        "z": 0.2 * math.sin(current_time * 0.5)
                    },
                    battery=max(20, 100 - (current_time / max(duration, estimated_flight_time)) * 80),
                    signal_strength=sig,
                    packet_loss=loss,
                    latency=lat,
                )
                yield data
                
                # Stop simulation if we've reached the endpoint
                if reached_end:
                    logger.info("Simulation stopped - reached endpoint")
                    break
                
                current_time += step
                
                
                await asyncio.sleep(step * 0.3)  
                
        except Exception as e:
            logger.error(f"Error in flight simulation: {str(e)}")
            raise
        finally:
            self.simulation_active = False

    def _get_noise_factor(self, simulation_type: SimulationType) -> float:
        """Get noise factor based on simulation type"""
        return {
            SimulationType.NORMAL: 0.1,
            SimulationType.MITM: 0.3,
            SimulationType.OUTSIDER: 1.2,
        }.get(simulation_type, 0.1)

    def _get_flight_mode(self, simulation_type: SimulationType) -> str:
        """Get flight mode based on simulation type"""
        return {
            SimulationType.NORMAL: "AUTO",
            SimulationType.MITM: "MANUAL",
            SimulationType.OUTSIDER: "FAILSAFE",
        }.get(simulation_type, "AUTO")

    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        if self.current_process:
            try:
                logger.info("Stopping simulation process")
                self.current_process.terminate()
                await asyncio.wait_for(self._wait_for_process(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Process didn't terminate gracefully, killing it")
                self.current_process.kill()
            except Exception as e:
                logger.error(f"Error stopping simulation: {str(e)}")
            finally:
                self.current_process = None
        self.simulation_active = False

    async def _wait_for_process(self):
        if self.current_process:
            while self.current_process.poll() is None:
                await asyncio.sleep(0.1)

    def _get_script_path(self, simulation_type: SimulationType) -> Path:
        """Get the path to the simulation script based on type"""
        mapping = {
            SimulationType.NORMAL: "normal_flight.py",
            SimulationType.MITM: "mitm_attack.py",
            SimulationType.OUTSIDER: "outsider_attack.py",
        }
        if isinstance(simulation_type, str):
            try:
                simulation_type = SimulationType(simulation_type.lower())
            except ValueError:
                logger.error(f"Unknown type: {simulation_type}")
                simulation_type = SimulationType.NORMAL
        script = mapping.get(simulation_type, "normal_flight.py")
        path = settings.SCRIPTS_DIR / script
        if not path.exists():
            for fb in ["normal_flight.py", "simulation.py", "flight.py"]:
                fbp = settings.SCRIPTS_DIR / fb
                if fbp.exists():
                    return fbp
            raise FileNotFoundError(f"No scripts in {settings.SCRIPTS_DIR}")
        return path

    def is_simulation_active(self) -> bool:
        return self.simulation_active and bool(self.current_process and self.current_process.poll() is None)

    def get_simulation_status(self) -> Dict[str, Any]:
        return {
            "active": self.simulation_active,
            "process_running": bool(self.current_process and self.current_process.poll() is None),
            "process_id": self.current_process.pid if self.current_process else None
        }