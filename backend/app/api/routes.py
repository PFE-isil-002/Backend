from fastapi import APIRouter, Depends, HTTPException, status, WebSocket
from typing import List, Dict, Any

from ..schemas.request import (
    SimulationRequest, SimulationStopRequest, ModelMetricsRequest
)
from ..schemas.response import (
    APIResponse, ModelMetricsResponse, SimulationStatusResponse
)
from ..domain.entities import ModelType, SimulationType
from ..services.model_service import ModelService
from ..services.simulation_service import SimulationService
from .deps import get_model_service, get_simulation_service
from .websocket import handle_websocket

router = APIRouter()


@router.post("/simulation/start", response_model=APIResponse)
async def start_simulation(
    request: SimulationRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
):
    """
    Start a new simulation.
    
    This endpoint only initiates the simulation process.
    The actual simulation data should be consumed via WebSocket.
    """
    try:
        # Check if simulation is already running
        status = simulation_service.get_simulation_status()
        if status["running"]:
            return APIResponse(
                success=False,
                message="A simulation is already running. Stop it before starting a new one.",
                data=status
            )
        
        # Validate the model and simulation types
        try:
            SimulationType(request.simulation_type)
            ModelType(request.model_type)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid simulation or model type: {str(e)}"
            )
        
        return APIResponse(
            success=True,
            message=f"Simulation {request.simulation_type} initiated with model {request.model_type}. Connect to the WebSocket endpoint to receive real-time data.",
            data={
                "simulation_type": request.simulation_type,
                "model_type": request.model_type,
                "duration": request.duration,
                "step": request.step
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start simulation: {str(e)}"
        )


@router.post("/simulation/stop", response_model=APIResponse)
async def stop_simulation(
    request: SimulationStopRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
):
    """Stop the currently running simulation"""
    try:
        # Check if simulation is running
        status = simulation_service.get_simulation_status()
        if not status["running"]:
            return APIResponse(
                success=False,
                message="No simulation is currently running.",
                data=status
            )
        
        # Stop the simulation
        await simulation_service.stop_simulation()
        
        return APIResponse(
            success=True,
            message="Simulation stopped successfully.",
            data={}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop simulation: {str(e)}"
        )


@router.get("/simulation/status", response_model=APIResponse)
def get_simulation_status(
    simulation_service: SimulationService = Depends(get_simulation_service),
):
    """Get the status of the current simulation"""
    try:
        status = simulation_service.get_simulation_status()
        
        return APIResponse(
            success=True,
            message="Simulation status retrieved successfully.",
            data=status
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get simulation status: {str(e)}"
        )


@router.post("/model/metrics", response_model=APIResponse)
async def get_model_metrics(
    request: ModelMetricsRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Get metrics for a specific model"""
    try:
        # Load the model
        await model_service.load_model(ModelType(request.model_type))
        
        # Get metrics
        metrics = model_service.get_model_metrics()
        
        return APIResponse(
            success=True,
            message=f"Metrics for model {request.model_type} retrieved successfully.",
            data=metrics
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model type: {str(e)}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metrics: {str(e)}"
        )


@router.get("/model/types", response_model=APIResponse)
def get_model_types():
    """Get a list of available model types"""
    try:
        model_types = [m.value for m in ModelType]
        
        return APIResponse(
            success=True,
            message="Available model types retrieved successfully.",
            data={"model_types": model_types}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model types: {str(e)}"
        )


@router.get("/simulation/types", response_model=APIResponse)
def get_simulation_types():
    """Get a list of available simulation types"""
    try:
        simulation_types = [s.value for s in SimulationType]
        
        return APIResponse(
            success=True,
            message="Available simulation types retrieved successfully.",
            data={"simulation_types": simulation_types}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get simulation types: {str(e)}"
        )