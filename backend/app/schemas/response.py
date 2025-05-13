from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool = Field(True, description="Whether the operation was successful")
    message: str = Field("", description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class DronePositionResponse(BaseModel):
    """Drone position data"""
    x: float
    y: float
    z: float
    timestamp: datetime


class DroneDataResponse(BaseModel):
    """Drone telemetry data"""
    position: DronePositionResponse
    velocity: Dict[str, float]
    orientation: Dict[str, float]
    battery: float
    signal_strength: float
    packet_loss: float
    latency: float


class DetectionResultResponse(BaseModel):
    """Anomaly detection result"""
    is_anomaly: bool
    confidence: float
    detection_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SimulationStatusResponse(BaseModel):
    """Status of the current simulation"""
    running: bool
    simulation_type: Optional[str] = None
    model_type: Optional[str] = None
    elapsed_time: float = 0.0
    remaining_time: float = 0.0


class ModelMetricsResponse(BaseModel):
    """Metrics for a specific model"""
    model_type: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class WebSocketMessage(BaseModel):
    """Base message format for WebSocket communication"""
    type: str
    data: Dict[str, Any]


class DroneDataMessage(WebSocketMessage):
    """WebSocket message for drone data updates"""
    type: str = "drone_data"
    data: Dict[str, Any]


class DetectionResultMessage(WebSocketMessage):
    """WebSocket message for detection result updates"""
    type: str = "detection_result"
    data: Dict[str, Any]


class StatusUpdateMessage(WebSocketMessage):
    """WebSocket message for status updates"""
    type: str = "status_update"
    data: Dict[str, Any]


class ErrorMessage(WebSocketMessage):
    """WebSocket message for errors"""
    type: str = "error"
    data: Dict[str, Any]