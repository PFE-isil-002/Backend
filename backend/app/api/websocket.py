import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..schemas.response import (
    WebSocketMessage, DroneDataMessage, DetectionResultMessage, StatusUpdateMessage, ErrorMessage
)
from ..domain.entities import DroneData, DetectionResult, ModelType, SimulationType
from ..domain.usecases import RealTimeMonitoringUseCase
from .deps import get_active_connections, get_monitoring_usecase

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages"""

    def __init__(self):
        self.active_connections = get_active_connections()
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new connection and store it"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "monitoring_task": None,
            "model_type": None,
            "simulation_type": None,
        }
        logger.info(f"New connection established: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """Remove a connection"""
        if connection_id in self.active_connections:
            logger.info(f"Connection closed: {connection_id}")
            self.active_connections.pop(connection_id, None)
    
    async def send_message(self, connection_id: str, message: WebSocketMessage) -> None:
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]["websocket"]
            try:
                await websocket.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
                self.disconnect(connection_id)
    
    async def broadcast(self, message: WebSocketMessage) -> None:
        """Send a message to all active connections"""
        for connection_id in list(self.active_connections.keys()):
            await self.send_message(connection_id, message)


manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket):
    """Handle WebSocket connections and messages"""
    connection_id = await manager.connect(websocket)
    
    try:
        while True:
            # Wait for a message from the client
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message = json.loads(data)
                
                # Process the message based on its type
                if message.get("type") == "start_simulation":
                    await handle_start_simulation(connection_id, message.get("data", {}))
                    
                elif message.get("type") == "stop_simulation":
                    await handle_stop_simulation(connection_id)
                    
                elif message.get("type") == "ping":
                    # Simple ping-pong to keep connection alive
                    await manager.send_message(
                        connection_id, 
                        WebSocketMessage(type="pong", data={"timestamp": message.get("data", {}).get("timestamp")})
                    )
                    
                else:
                    # Unknown message type
                    await manager.send_message(
                        connection_id,
                        ErrorMessage(type="error", data={"message": f"Unknown message type: {message.get('type')}"})
                    )
                    
            except json.JSONDecodeError:
                # Invalid JSON
                await manager.send_message(
                    connection_id,
                    ErrorMessage(type="error", data={"message": "Invalid JSON message"})
                )
                
            except Exception as e:
                # Other errors
                logger.exception("Error processing WebSocket message")
                await manager.send_message(
                    connection_id,
                    ErrorMessage(type="error", data={"message": f"Error: {str(e)}"})
                )
                
    except WebSocketDisconnect:
        # Handle client disconnect
        await cleanup_connection(connection_id)
        manager.disconnect(connection_id)
    
    except Exception as e:
        # Handle other errors
        logger.exception(f"WebSocket error: {str(e)}")
        await cleanup_connection(connection_id)
        manager.disconnect(connection_id)


async def handle_start_simulation(connection_id: str, data: Dict[str, Any]) -> None:
    """Handle a request to start a simulation"""
    # Stop any existing simulation for this connection
    await handle_stop_simulation(connection_id)
    
    try:
        # Get parameters
        simulation_type_str = data.get("simulation_type")
        model_type_str = data.get("model_type")
        duration = int(data.get("duration", 300))
        step = float(data.get("step", 0.1))
        
        # Validate parameters
        if not simulation_type_str or not model_type_str:
            raise ValueError("simulation_type and model_type are required")
        
        simulation_type = SimulationType(simulation_type_str)
        model_type = ModelType(model_type_str)
        
        # Get monitoring use case
        monitoring_usecase = get_monitoring_usecase()
        
        # Update connection info
        manager.active_connections[connection_id]["model_type"] = model_type
        manager.active_connections[connection_id]["simulation_type"] = simulation_type
        
        # Define callback for data updates
        def update_callback(drone_data: DroneData, detection_result: DetectionResult) -> None:
            # This runs in the monitoring task, so we need to create a task to send messages
            asyncio.create_task(send_update(connection_id, drone_data, detection_result))
        
        # Register the callback
        monitoring_usecase.register_callback(update_callback)
        
        # Create and store the monitoring task
        task = asyncio.create_task(
            run_monitoring(
                connection_id=connection_id,
                monitoring_usecase=monitoring_usecase,
                simulation_type=simulation_type,
                model_type=model_type,
                duration=duration,
                step=step
            )
        )
        
        manager.active_connections[connection_id]["monitoring_task"] = task
        
        # Send confirmation to client
        await manager.send_message(
            connection_id,
            WebSocketMessage(
                type="simulation_started",
                data={
                    "simulation_type": simulation_type.value,
                    "model_type": model_type.value,
                    "duration": duration,
                    "step": step
                }
            )
        )
        
    except ValueError as e:
        # Parameter validation error
        await manager.send_message(
            connection_id,
            ErrorMessage(type="error", data={"message": f"Invalid parameters: {str(e)}"})
        )
        
    except Exception as e:
        # Other errors
        logger.exception("Error starting simulation")
        await manager.send_message(
            connection_id,
            ErrorMessage(type="error", data={"message": f"Error starting simulation: {str(e)}"})
        )


async def handle_stop_simulation(connection_id: str) -> None:
    """Handle a request to stop a simulation"""
    await cleanup_connection(connection_id)
    
    # Send confirmation to client
    await manager.send_message(
        connection_id,
        WebSocketMessage(type="simulation_stopped", data={})
    )


async def cleanup_connection(connection_id: str) -> None:
    """Clean up tasks and resources for a connection"""
    # Cancel the monitoring task if it exists
    if connection_id in manager.active_connections:
        task = manager.active_connections[connection_id].get("monitoring_task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
        # Reset connection info
        manager.active_connections[connection_id]["monitoring_task"] = None
        manager.active_connections[connection_id]["model_type"] = None
        manager.active_connections[connection_id]["simulation_type"] = None


async def run_monitoring(
    connection_id: str,
    monitoring_usecase: RealTimeMonitoringUseCase,
    simulation_type: SimulationType,
    model_type: ModelType,
    duration: int,
    step: float
) -> None:
    """Run the monitoring process in the background"""
    try:
        async for drone_data, detection_result in monitoring_usecase.start_monitoring(
            simulation_type=simulation_type,
            model_type=model_type,
            duration=duration,
            step=step
        ):
            # The callback will handle sending updates to the client
            pass
            
    except asyncio.CancelledError:
        # Handle cancellation
        await monitoring_usecase.stop_monitoring()
        raise
        
    except Exception as e:
        # Handle other errors
        logger.exception(f"Error in monitoring task: {str(e)}")
        if connection_id in manager.active_connections:
            await manager.send_message(
                connection_id,
                ErrorMessage(type="error", data={"message": f"Monitoring error: {str(e)}"})
            )
    
    finally:
        # Ensure the monitoring is stopped
        await monitoring_usecase.stop_monitoring()
        
        # Send simulation complete message if the connection still exists
        if connection_id in manager.active_connections:
            await manager.send_message(
                connection_id,
                WebSocketMessage(type="simulation_complete", data={})
            )


async def send_update(connection_id: str, drone_data: DroneData, detection_result: DetectionResult) -> None:
    """Send updates to the client"""
    # Check if the connection still exists
    if connection_id not in manager.active_connections:
        return
    
    # Send drone data update
    await manager.send_message(
        connection_id,
        DroneDataMessage(
            type="drone_data",
            data={
                "position": {
                    "x": drone_data.position.x,
                    "y": drone_data.position.y,
                    "z": drone_data.position.z,
                    "timestamp": drone_data.position.timestamp.isoformat()
                },
                "velocity": drone_data.velocity,
                "orientation": drone_data.orientation,
                "battery": drone_data.battery,
                "signal_strength": drone_data.signal_strength,
                "packet_loss": drone_data.packet_loss,
                "latency": drone_data.latency
            }
        )
    )
    
    # Send detection result update
    await manager.send_message(
        connection_id,
        DetectionResultMessage(
            type="detection_result",
            data={
                "is_anomaly": detection_result.is_anomaly,
                "confidence": detection_result.confidence,
                "detection_type": detection_result.detection_type,
                "details": detection_result.details
            }
        )
    )