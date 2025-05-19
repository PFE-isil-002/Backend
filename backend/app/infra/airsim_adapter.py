import asyncio
import sys
import importlib.util
import logging
from typing import AsyncIterator, Dict, Any, Optional, List
from pathlib import Path
import subprocess
import json
from datetime import datetime
import time

from ..core.config import settings
from ..domain.entities import DroneData, DronePosition, SimulationType

# Configure module logger
logger = logging.getLogger("app.airsim_adapter")


class AirSimAdapter:
    """Adapter for interacting with drone simulation scripts"""

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self._running = False
        logger.info("AirSimAdapter initialized")

    async def start_simulation(
        self,
        simulation_type: SimulationType,
        duration: int = 300,
        step: float = 0.1
    ) -> AsyncIterator[DroneData]:
        """Start a simulation based on the type and yield drone data"""

        await self.stop_simulation()
        script_path = self._get_script_path(simulation_type)

        if not script_path.exists():
            logger.error(f"Simulation script not found: {script_path}")
            raise FileNotFoundError(f"Simulation script not found at {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--duration", str(duration),
            "--step", str(step)
        ]

        logger.info(f"Starting simulation process: {' '.join(cmd)}")

        self._running = True
        start_time = time.time()
        data_count = 0

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.info(f"Simulation process started with PID: {self.process.pid if self.process else 'unknown'}")
            stderr_task = asyncio.create_task(self._read_stderr())

            if self.process.stdout:
                while self._running:
                    line = await self.process.stdout.readline()
                    if not line:
                        logger.info("End of simulation output stream")
                        break

                    raw_data = line.decode("utf-8", errors="ignore").strip()

                    if not raw_data:
                        continue

                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        logger.info(f"Non-JSON output: {raw_data}")
                        continue

                    try:
                        data_count += 1
                        if data_count % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = data_count / elapsed if elapsed > 0 else 0
                            logger.info(f"Processing data point {data_count} ({rate:.2f} points/sec)")

                        drone_data = self._parse_drone_data(data)
                        yield drone_data

                    except Exception as e:
                        logger.error(f"Error processing simulation data: {str(e)}")

            if self.process:
                exit_code = await self.process.wait()
                logger.info(f"Simulation process exited with code: {exit_code}")

        except asyncio.CancelledError:
            logger.info("Simulation was cancelled")
            await self.stop_simulation()
            raise
        except Exception as e:
            logger.exception(f"Error in simulation: {str(e)}")
            await self.stop_simulation()
            raise
        finally:
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()

            elapsed = time.time() - start_time
            logger.info(f"Simulation completed: {data_count} data points in {elapsed:.2f}s ({data_count/elapsed:.2f} points/sec)")

    async def _read_stderr(self) -> None:
        """Read and log stderr from the simulation process"""
        if not self.process or not self.process.stderr:
            return
        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            error_msg = line.decode('utf-8').strip()
            if error_msg:
                logger.error(f"Simulation stderr: {error_msg}")

    async def stop_simulation(self) -> None:
        """Stop the current simulation"""
        self._running = False
        if self.process and self.process.returncode is None:
            try:
                logger.info(f"Stopping simulation process (PID: {self.process.pid if self.process else 'unknown'})")
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    logger.info("Simulation process terminated gracefully")
                except asyncio.TimeoutError:
                    logger.warning("Simulation process did not terminate gracefully, killing it")
                    self.process.kill()
                    await self.process.wait()
                    logger.info("Simulation process killed")
            except Exception as e:
                logger.error(f"Error stopping simulation: {str(e)}")

    def _get_script_path(self, simulation_type: SimulationType) -> Path:
        """Get the path to the simulation script based on the type"""
        if simulation_type == SimulationType.NORMAL:
            script_path = settings.NORMAL_FLIGHT_SCRIPT
            script_name = "normal flight script"
        elif simulation_type == SimulationType.MITM:
            script_path = settings.MITM_FLIGHT_SCRIPT
            script_name = "MITM flight script"
        elif simulation_type == SimulationType.OUTSIDER:
            script_path = settings.OUTSIDER_DRONE_SCRIPT
            script_name = "outsider drone script"
        else:
            logger.error(f"Unknown simulation type: {simulation_type}")
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        logger.info(f"Using {script_name} at path: {script_path}")
        return script_path

    def _parse_drone_data(self, data: Dict[str, Any]) -> DroneData:
        """Parse the simulation output into DroneData object"""
        try:
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
        except Exception as e:
            logger.error(f"Error parsing drone data: {str(e)}")
            logger.debug(f"Raw data: {data}")
            return DroneData(
                position=DronePosition(x=0.0, y=0.0, z=0.0, timestamp=datetime.now()),
                velocity={"x": 0, "y": 0, "z": 0},
                orientation={"pitch": 0, "roll": 0, "yaw": 0},
                battery=100.0,
                signal_strength=100.0,
                packet_loss=0.0,
                latency=0.0
            )
