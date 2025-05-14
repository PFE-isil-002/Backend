from typing import Generator, AsyncGenerator, Dict
import logging
from ..domain.usecases import RealTimeMonitoringUseCase
from ..services.model_service import ModelService
from ..services.simulation_service import SimulationService

logger = logging.getLogger(__name__)


def get_model_service() -> ModelService:
    """Dependency for model service"""
    return ModelService()


def get_simulation_service() -> SimulationService:
    """Dependency for simulation service"""
    return SimulationService()


def get_monitoring_usecase(
    simulation_service: SimulationService = get_simulation_service(),
    model_service: ModelService = get_model_service()
) -> RealTimeMonitoringUseCase:
    """Dependency for real-time monitoring use case"""
    return RealTimeMonitoringUseCase(
        simulation_use_case=simulation_service,
        model_use_case=model_service
    )


# Dictionary to store active websocket connections
active_connections: Dict[str, Dict] = {}


def get_active_connections() -> Dict[str, Dict]:
    """Get the dictionary of active connections"""
    return active_connections