import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from ..domain.entities import ModelType, SimulationType
from ..services.model_service import ModelService
from ..services.simulation_service import SimulationService
from ..domain.usecases import RealTimeMonitoringUseCase as MonitoringUseCase

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and batch prediction workflow"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}
        self.model_service = ModelService()
        self.simulation_service = SimulationService()
        self.monitoring_use_case = MonitoringUseCase(
            model_use_case=self.model_service,
            simulation_use_case=self.simulation_service
        )

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        await self.send_message(client_id, "connection_established", {"client_id": client_id})

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.connection_tasks:
            task = self.connection_tasks[client_id]
            if not task.done():
                task.cancel()
            del self.connection_tasks[client_id]

        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message_type: str, data: Any):
        if client_id not in self.active_connections:
            return

        try:
            # Custom JSON serializer for dataclasses if needed, or convert before calling send_message
            # For now, we'll assume data is already pre-processed for JSON serialization
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.active_connections[client_id].send_text(json.dumps(message))
            logger.info(f"Sent message to {client_id}: {message_type}")
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {str(e)}")
            await self.disconnect(client_id)

    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        message_type = message.get("type")
        data = message.get("data", {})

        logger.info(f"Received message from {client_id}: {message_type}")

        try:
            if message_type == "start_simulation":
                await self.start_simulation_with_batch_prediction(client_id, data)
            elif message_type == "stop_simulation":
                await self.stop_simulation(client_id)
            elif message_type == "get_model_metrics":
                await self.send_model_metrics(client_id)
            elif message_type == "load_model":
                await self.load_model(client_id, data)
            else:
                await self.send_message(client_id, "error", {"message": f"Unknown message type: {message_type}"})

        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {str(e)}")
            await self.send_message(client_id, "error", {"message": str(e)})

    async def start_simulation_with_batch_prediction(self, client_id: str, data: Dict[str, Any]):
        simulation_type = data.get("simulation_type", "normal")
        model_type = data.get("model_type", "knn")
        duration = data.get("duration", 300)
        step = data.get("step", 0.1)
        velocity = data.get("velocity", 5.0)
        start_point = data.get("start_point", {"x": 0.0, "y": 0.0, "z": -5.0})
        end_point = data.get("end_point", {"x": 50.0, "y": 25.0, "z": -5.0})
        waypoints = data.get("waypoints", [])

        logger.info(f"Starting simulation for {client_id} with batch prediction")
        logger.info(f"Parameters - simulation_type: {simulation_type}, model_type: {model_type}, duration: {duration}s, step: {step}s")

        try:
            await self.model_service.load_model(ModelType(model_type))
            self.model_service.start_waypoint_collection()

            await self.send_message(client_id, "simulation_started", {
                "simulation_type": simulation_type,
                "model_type": model_type,
                "duration": duration,
                "step": step,
                "batch_prediction": True
            })

            task = asyncio.create_task(
                self.run_batch_monitoring(client_id, simulation_type, duration, step, velocity,
                                            start_point, end_point, waypoints)
            )
            self.connection_tasks[client_id] = task

        except Exception as e:
            logger.error(f"Error starting simulation for {client_id}: {str(e)}")
            await self.send_message(client_id, "error", {"message": f"Simulation error: {str(e)}"})

    async def run_batch_monitoring(
        self,
        client_id: str,
        simulation_type: str,
        duration: float,
        step: float,
        velocity: float,
        start_point: Dict,
        end_point: Dict,
        waypoints: list
    ):
        logger.info(f"Starting batch monitoring for {client_id}")
        start_time = datetime.utcnow()
        data_points_processed = 0

        try:
            try:
                sim_type = SimulationType(simulation_type.lower())
            except ValueError:
                logger.warning(f"Unknown simulation type '{simulation_type}', defaulting to 'normal'")
                sim_type = SimulationType.NORMAL

            async for drone_data in self.simulation_service.start_simulation(
                simulation_type=sim_type,
                duration=duration,
                step=step,
                velocity=velocity,
                start_point=start_point,
                end_point=end_point,
                waypoints=waypoints
            ):
                self.model_service.add_waypoint(drone_data)
                data_points_processed += 1

                if data_points_processed % 50 == 0:
                    await self.send_message(client_id, "waypoint_collected", {
                        "waypoints_collected": self.model_service.get_collected_waypoints_count(),
                        # Convert DronePosition object to a dictionary for JSON serialization
                        "current_position": {
                            "x": drone_data.position.x,
                            "y": drone_data.position.y,
                            "z": drone_data.position.z,
                            "timestamp": drone_data.position.timestamp.isoformat()
                        },
                        "timestamp": drone_data.position.timestamp.isoformat()
                    })

                if data_points_processed % 10 == 0:
                    await asyncio.sleep(0.01)

            logger.info(f"Collected {data_points_processed} waypoints for batch prediction")

            await self.send_message(client_id, "analyzing_flight", {
                "total_waypoints": data_points_processed,
                "message": "Analyzing complete flight path..."
            })

            detection_result = await self.model_service.predict_batch()

            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds()

            await self.send_message(client_id, "batch_prediction_complete", {
                "detection_result": {
                    "is_anomaly": bool(detection_result.is_anomaly), # Explicitly cast to bool
                    "confidence": detection_result.confidence,
                    "detection_type": detection_result.detection_type,
                    "details": detection_result.details
                },
                "waypoints_analyzed": data_points_processed,
                "analysis_duration": total_duration,
                "flight_summary": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_waypoints": data_points_processed,
                    "avg_waypoints_per_second": data_points_processed / total_duration if total_duration > 0 else 0
                }
            })

            await self.send_message(client_id, "simulation_complete", {
                "processed_data_points": data_points_processed,
                "total_duration": total_duration,
                "prediction_method": "batch",
                "final_result": {
                    "threat_detected": bool(detection_result.is_anomaly), # Explicitly cast to bool
                    "threat_type": detection_result.detection_type,
                    "confidence": detection_result.confidence
                }
            })

        except Exception as e:
            logger.error(f"Error in batch monitoring for {client_id}: {str(e)}")
            await self.send_message(client_id, "error", {"message": f"Monitoring error: {str(e)}"})

        finally:
            logger.info(f"Finalizing batch monitoring for {client_id}")
            self.model_service.clear_waypoint_buffer()

            if client_id in self.connection_tasks:
                del self.connection_tasks[client_id]

    async def stop_simulation(self, client_id: str):
        logger.info(f"Stopping simulation for {client_id}")

        if client_id in self.connection_tasks:
            task = self.connection_tasks[client_id]
            if not task.done():
                task.cancel()
            del self.connection_tasks[client_id]

        await self.simulation_service.stop_simulation()
        self.model_service.clear_waypoint_buffer()

        await self.send_message(client_id, "simulation_stopped", {})

    async def load_model(self, client_id: str, data: Dict[str, Any]):
        model_type = data.get("model_type")
        if not model_type:
            await self.send_message(client_id, "error", {"message": "Missing model_type"})
            return

        try:
            await self.model_service.load_model(ModelType(model_type))
            await self.send_message(client_id, "model_loaded", {
                "model_type": model_type,
                "status": "success"
            })
        except Exception as e:
            await self.send_message(client_id, "error", {"message": f"Failed to load model: {str(e)}"})

    async def send_model_metrics(self, client_id: str):
        metrics = self.model_service.get_model_metrics()
        await self.send_message(client_id, "model_metrics", metrics)


# Global WebSocket manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())

    try:
        await manager.connect(websocket, client_id)

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await manager.handle_message(client_id, message)

            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected normally")
                break
            except json.JSONDecodeError:
                await manager.send_message(client_id, "error", {"message": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"Error in websocket for {client_id}: {str(e)}")
                await manager.send_message(client_id, "error", {"message": str(e)})

    except Exception as e:
        logger.error(f"WebSocket connection error for {client_id}: {str(e)}")

    finally:
        await manager.disconnect(client_id)


# Alias for FastAPI to use in routing
handle_websocket = websocket_endpoint