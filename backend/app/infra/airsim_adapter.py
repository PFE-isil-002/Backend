import asyncio
import sys
import importlib.util
import logging
from typing import AsyncIterator, Dict, Any, Optional, List
from pathlib import Path
import subprocess
import json
from datetime import datetime

from ..core.config import settings
from ..domain.entities import DroneData, DronePosition, SimulationType

logger = logging.getLogger(__name__)


class AirSimAdapter:
    """Adapter for interacting with drone simulation scripts"""
    
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self._running = False
    
    async def start_simulation(
        self, 
        simulation_type: SimulationType,
        duration: int = 300, 
        step: float = 0.1
    ) -> AsyncIterator[DroneData]:
        """Start a simulation based on the type and yield drone data"""
        
        # Stop any running simulations
        await self.stop_simulation()
        
        # Determine which script to run
        script_path = self._get_script_path(simulation_type)
        
        if not script_path.exists():
            raise FileNotFoundError(f"Simulation script not found at {script_path}")
        
        # Start the simulation process
        cmd = [
            sys.executable, 
            str(script_path),
            "--duration", str(duration),
            "--step", str(step)
        ]
        
        self._running = True
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Process output
        if self.process.stdout:
            while self._running:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    # Parse drone data from the script's output
                    data = json.loads(line.decode('utf-8').strip())
                    drone_data = self._parse_drone_data(data)
                    yield drone_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse simulation output: {line}")
                except Exception as e:
                    logger.error(f"Error processing simulation data: {str(e)}")
        
        # Wait for process to complete
        if self.process:
            await self.process.wait()
    
    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        self._running = False
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping simulation: {str(e)}")
    
    def _get_script_path(self, simulation_type: SimulationType) -> Path:
        """Get the path to the simulation script based on the type"""
        if simulation_type == SimulationType.NORMAL:
            return settings.NORMAL_FLIGHT_SCRIPT
        elif simulation_type == SimulationType.MITM:
            return settings.MITM_FLIGHT_SCRIPT
        elif simulation_type == SimulationType.OUTSIDER:
            return settings.OUTSIDER_DRONE_SCRIPT
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
    
    def _parse_drone_data(self, data: Dict[str, Any]) -> DroneData:
        """Parse the simulation output into DroneData object"""
        position = DronePosition(
            x=float(data.get("position", {}).get("x", 0)),
            y=float(data.get("position", {}).get("y", 0)),
            z=float(data.get("position", {}).get("z", 0)),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )
        
        return DroneData(
            position=position,
            velocity=data.get("velocity", {"x": 0, "y": 0, "z": 0}),
            orientation=data.get("orientation", {"pitch": 0, "roll": 0, "yaw": 0}),
            battery=float(data.get("battery", 100.0)),
            signal_strength=float(data.get("signal_strength", 100.0)),
            packet_loss=float(data.get("packet_loss", 0.0)),
            latency=float(data.get("latency", 0.0))
        )