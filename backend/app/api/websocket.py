import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect
import uuid
import traceback
import time

from ..schemas.response import (
    WebSocketMessage, DroneDataMessage, DetectionResultMessage, StatusUpdateMessage, ErrorMessage
)
from ..domain.entities import DroneData, DetectionResult, ModelType, SimulationType
from ..domain.usecases import RealTimeMonitoringUseCase
from .deps import get_active_connections, get_monitoring_usecase

# Configure module logger
logger = logging.getLogger("app.websocket")


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages"""

    def __init__(self):
        self.active_connections = get_active_connections()
        logger.info("ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new connection and store it"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "monitoring_task": None,
            "model_type": None,
            "simulation_type": None,
            "connected_at": time.time(),
        }
        logger.info(f"New connection established: {connection_id}")
        logger.debug(f"Active connections: {len(self.active_connections)}")
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """Remove a connection"""
        if connection_id in self.active_connections:
            connection_data = self.active_connections[connection_id]
            duration = time.time() - connection_data.get("connected_at", time.time())
            logger.info(f"Connection closed: {connection_id} (duration: {duration:.2f}s)")
            self.active_connections.pop(connection_id, None)
            logger.debug(f"Remaining active connections: {len(self.active_connections)}")
    
    async def send_message(self, connection_id: str, message: WebSocketMessage) -> None:
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]["websocket"]
            try:
                message_dict = message.dict()
                msg_type = message_dict.get("type", "unknown")
                
                # Don't log full message content for frequent updates to avoid log spam
                if msg_type in ["drone_data", "detection_result"]:
                    logger.debug(f"Sending {msg_type} message to {connection_id}")
                else:
                    logger.info(f"Sending message to {connection_id}: {msg_type} - {message_dict}")
                
                await websocket.send_json(message_dict)
            except Exception as e:
                logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
                self.disconnect(connection_id)
    
    async def broadcast(self, message: WebSocketMessage) -> None:
        """Send a message to all active connections"""
        logger.info(f"Broadcasting message to {len(self.active_connections)} connections: {message.type}")
        for connection_id in list(self.active_connections.keys()):
            await self.send_message(connection_id, message)


manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket):
    """Handle WebSocket connections and messages"""
    connection_id = await manager.connect(websocket)
    
    try:
        logger.info(f"Starting websocket handler loop for connection: {connection_id}")
        while True:
            # Wait for a message from the client
            logger.debug(f"Waiting for message from client: {connection_id}")
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message = json.loads(data)
                msg_type = message.get("type", "unknown")
                
                logger.info(f"Received message from {connection_id}: {msg_type}")
                logger.debug(f"Message data: {message.get('data', {})}")
                
                # Process the message based on its type
                if msg_type == "start_simulation":
                    await handle_start_simulation(connection_id, message.get("data", {}))
                    
                elif msg_type == "stop_simulation":
                    await handle_stop_simulation(connection_id)
                    
                elif msg_type == "ping":
                    # Simple ping-pong to keep connection alive
                    logger.debug(f"Ping received from {connection_id}")
                    await manager.send_message(
                        connection_id, 
                        WebSocketMessage(type="pong", data={"timestamp": message.get("data", {}).get("timestamp")})
                    )
                    
                else:
                    # Unknown message type
                    logger.warning(f"Unknown message type from {connection_id}: {msg_type}")
                    await manager.send_message(
                        connection_id,
                        ErrorMessage(type="error", data={"message": f"Unknown message type: {msg_type}"})
                    )
                    
            except json.JSONDecodeError:
                # Invalid JSON
                logger.error(f"Invalid JSON message from {connection_id}: {data[:100]}...")
                await manager.send_message(
                    connection_id,
                    ErrorMessage(type="error", data={"message": "Invalid JSON message"})
                )
                
            except Exception as e:
                # Other errors
                logger.exception(f"Error processing WebSocket message from {connection_id}")
                await manager.send_message(
                    connection_id,
                    ErrorMessage(type="error", data={"message": f"Error: {str(e)}"})
                )
                
    except WebSocketDisconnect:
        # Handle client disconnect
        logger.info(f"WebSocket disconnect detected for {connection_id}")
        await cleanup_connection(connection_id)
        manager.disconnect(connection_id)
    
    except Exception as e:
        # Handle other errors
        logger.exception(f"WebSocket error for {connection_id}: {str(e)}")
        await cleanup_connection(connection_id)
        manager.disconnect(connection_id)


async def handle_start_simulation(connection_id: str, data: Dict[str, Any]) -> None:
    """Handle a request to start a simulation"""
    # Stop any existing simulation for this connection
    logger.info(f"Starting simulation for {connection_id} with data: {data}")
    await handle_stop_simulation(connection_id)
    
    try:
        # Get parameters
        simulation_type_str = data.get("simulation_type")
        model_type_str = data.get("model_type")
        duration = int(data.get("duration", 300))
        step = float(data.get("step", 0.1))
        
        # Log parameters
        logger.info(f"Simulation parameters - simulation_type: {simulation_type_str}, model_type: {model_type_str}, duration: {duration}s, step: {step}s")
        
        # Validate parameters
        if not simulation_type_str or not model_type_str:
            logger.error(f"Missing required parameters for {connection_id}")
            raise ValueError("simulation_type and model_type are required")
        
        simulation_type = SimulationType(simulation_type_str)
        model_type = ModelType(model_type_str)
        
        # Get monitoring use case
        monitoring_usecase = get_monitoring_usecase()
        logger.debug(f"Monitoring usecase retrieved for {connection_id}")
        
        # Update connection info
        manager.active_connections[connection_id]["model_type"] = model_type
        manager.active_connections[connection_id]["simulation_type"] = simulation_type
        
        # Define callback for data updates
        def update_callback(drone_data: DroneData, detection_result: DetectionResult) -> None:
            # This runs in the monitoring task, so we need to create a task to send messages
            logger.debug(f"Callback triggered with drone data and detection result for {connection_id}")
            asyncio.create_task(send_update(connection_id, drone_data, detection_result))
        
        # Register the callback
        monitoring_usecase.register_callback(update_callback)
        logger.debug(f"Callback registered for {connection_id}")
        
        # Create and store the monitoring task
        logger.info(f"Creating monitoring task for {connection_id}")
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
        logger.info(f"Simulation started for {connection_id}")
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
        logger.error(f"Invalid parameters for {connection_id}: {str(e)}")
        await manager.send_message(
            connection_id,
            ErrorMessage(type="error", data={"message": f"Invalid parameters: {str(e)}"})
        )
        
    except Exception as e:
        # Other errors
        logger.exception(f"Error starting simulation for {connection_id}")
        await manager.send_message(
            connection_id,
            ErrorMessage(type="error", data={"message": f"Error starting simulation: {str(e)}"})
        )


async def handle_stop_simulation(connection_id: str) -> None:
    """Handle a request to stop a simulation"""
    logger.info(f"Stopping simulation for {connection_id}")
    await cleanup_connection(connection_id)
    
    # Send confirmation to client
    await manager.send_message(
        connection_id,
        WebSocketMessage(type="simulation_stopped", data={})
    )
    logger.info(f"Simulation stopped for {connection_id}")


async def cleanup_connection(connection_id: str) -> None:
    """Clean up tasks and resources for a connection"""
    logger.info(f"Cleaning up connection {connection_id}")
    # Cancel the monitoring task if it exists
    if connection_id in manager.active_connections:
        task = manager.active_connections[connection_id].get("monitoring_task")
        if task and not task.done():
            logger.debug(f"Cancelling monitoring task for {connection_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug(f"Monitoring task cancelled for {connection_id}")
            except Exception as e:
                logger.error(f"Error while cancelling task for {connection_id}: {str(e)}")
            
        # Reset connection info
        logger.debug(f"Resetting connection info for {connection_id}")
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
    logger.info(f"Starting monitoring for {connection_id} - simulation: {simulation_type.value}, model: {model_type.value}")
    start_time = time.time()
    data_count = 0
    
    try:
        logger.debug(f"Initiating monitoring stream for {connection_id}")
        async for drone_data, detection_result in monitoring_usecase.start_monitoring(
            simulation_type=simulation_type,
            model_type=model_type,
            duration=duration,
            step=step
        ):
            # The callback will handle sending updates to the client
            data_count += 1
            
            # Log data at appropriate intervals to avoid excessive logging
            if data_count % 10 == 0:  # Log every 10th data point
                logger.info(f"Processed {data_count} data points for {connection_id} ({time.time() - start_time:.2f}s elapsed)")
            
            logger.debug(f"Data point {data_count}: position=({drone_data.position.x:.2f}, {drone_data.position.y:.2f}, {drone_data.position.z:.2f}), anomaly={detection_result.is_anomaly}")
            
    except asyncio.CancelledError:
        # Handle cancellation
        logger.info(f"Monitoring task cancelled for {connection_id} after {data_count} data points")
        await monitoring_usecase.stop_monitoring()
        raise
        
    except Exception as e:
        # Handle other errors
        logger.exception(f"Error in monitoring task for {connection_id}: {str(e)}")
        if connection_id in manager.active_connections:
            await manager.send_message(
                connection_id,
                ErrorMessage(type="error", data={"message": f"Monitoring error: {str(e)}"})
            )
    
    finally:
        # Ensure the monitoring is stopped
        logger.info(f"Finalizing monitoring for {connection_id} after {time.time() - start_time:.2f}s")
        await monitoring_usecase.stop_monitoring()
        
        # Send simulation complete message if the connection still exists
        if connection_id in manager.active_connections:
            logger.info(f"Simulation complete for {connection_id}, processed {data_count} data points")
            await manager.send_message(
                connection_id,
                WebSocketMessage(type="simulation_complete", data={
                    "processed_data_points": data_count,
                    "total_duration": time.time() - start_time
                })
            )


async def send_update(connection_id: str, drone_data: DroneData, detection_result: DetectionResult) -> None:
    """Send updates to the client"""
    # Check if the connection still exists
    if connection_id not in manager.active_connections:
        logger.debug(f"Connection {connection_id} no longer exists, skipping update")
        return
    
    # For anomaly detections, increase log level
    if detection_result.is_anomaly:
        logger.info(f"ANOMALY DETECTED for {connection_id} - confidence: {detection_result.confidence}, type: {detection_result.detection_type}")
    
    # Send drone data update
    try:
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
    except Exception as e:
        logger.error(f"Error sending update to {connection_id}: {str(e)}")