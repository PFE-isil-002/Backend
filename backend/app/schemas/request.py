from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum

from ..domain.entities import ModelType, SimulationType


class SimulationRequest(BaseModel):
    """Request to start a simulation"""
    simulation_type: SimulationType = Field(..., description="Type of simulation to run")
    model_type: ModelType = Field(..., description="Type of model to use for detection")
    duration: Optional[int] = Field(300, description="Duration of simulation in seconds")
    step: Optional[float] = Field(0.1, description="Time step interval in seconds")


class SimulationStopRequest(BaseModel):
    """Request to stop a simulation"""
    pass


class ModelMetricsRequest(BaseModel):
    """Request to get model metrics"""
    model_type: ModelType = Field(..., description="Type of model to get metrics for")