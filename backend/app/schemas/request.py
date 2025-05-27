from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..domain.entities import ModelType, SimulationType


class Position(BaseModel):
    """3D Position coordinates"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(-5.0, description="Z coordinate (altitude, default -5m)")


class SimulationRequest(BaseModel):
    """Request to start a simulation"""
    simulation_type: SimulationType = Field(..., description="Type of simulation to run")
    model_type: ModelType = Field(..., description="Type of model to use for detection")
    duration: Optional[int] = Field(300, description="Duration of simulation in seconds")
    step: Optional[float] = Field(0.1, description="Time step interval in seconds")
    start_point: Optional[Position] = Field(None, description="Starting position for the drone")
    end_point: Optional[Position] = Field(None, description="Ending position for the drone")
    waypoints: Optional[List[Position]] = Field(None, description="Intermediate waypoints between start and end")
    velocity: Optional[float] = Field(5.0, description="Flight velocity in m/s", ge=1.0, le=15.0)
    
    @validator('waypoints')
    def validate_waypoints(cls, v, values):
        """Ensure waypoints are provided with start/end points"""
        if v is not None:
            start_point = values.get('start_point')
            end_point = values.get('end_point')
            if start_point is None or end_point is None:
                raise ValueError("start_point and end_point must be provided when using waypoints")
        return v
    
    @validator('end_point')
    def validate_end_point(cls, v, values):
        """Ensure end_point is provided if start_point is provided"""
        start_point = values.get('start_point')
        if start_point is not None and v is None:
            raise ValueError("end_point must be provided when start_point is specified")
        return v


class SimulationStopRequest(BaseModel):
    """Request to stop a simulation"""
    pass


class ModelMetricsRequest(BaseModel):
    """Request to get model metrics"""
    model_type: ModelType = Field(..., description="Type of model to get metrics for")


class FlightPathRequest(BaseModel):
    """Request to generate a flight path"""
    start_point: Position = Field(..., description="Starting position")
    end_point: Position = Field(..., description="Ending position")
    path_type: str = Field("direct", description="Type of path: 'direct', 'curved', 'zigzag'")
    num_waypoints: Optional[int] = Field(5, description="Number of intermediate waypoints", ge=0, le=20)


class SimulationStatus(BaseModel):
    """Status of current simulation"""
    running: bool = Field(..., description="Whether simulation is currently running")
    simulation_type: Optional[str] = Field(None, description="Current simulation type")
    elapsed_time: float = Field(0.0, description="Elapsed time in seconds")
    remaining_time: float = Field(0.0, description="Remaining time in seconds")
    current_position: Optional[Position] = Field(None, description="Current drone position")
    target_position: Optional[Position] = Field(None, description="Current target position")