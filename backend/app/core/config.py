from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "UAV Authentication System"
    
    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost", "http://localhost:3000", "http://localhost:9000"]
    
    # Simulation settings
    SIMULATION_DURATION: int = 300  # seconds
    SIMULATION_STEP: float = 0.1    # seconds
    
    # Model paths relative to BASE_DIR
    MODEL_DIR: Path = BASE_DIR / "model/les_models"
    DATASET_DIR: Path = BASE_DIR / "dataset"
    
    # Simulation script paths
    NORMAL_FLIGHT_SCRIPT: Path = BASE_DIR / "scripts" / "vol_normal.py"
    MITM_FLIGHT_SCRIPT: Path = BASE_DIR / "scripts" / "miTM.py"
    OUTSIDER_DRONE_SCRIPT: Path = BASE_DIR / "scripts" / "false_flight.py"
    
    # Available model types
    AVAILABLE_MODELS: List[str] = ["knn", "logistic_regression", "svm", "lstm", "rnn", "random_forest"]
    
    # Simulation types
    AVAILABLE_SIMULATIONS: List[str] = ["normal", "mitm", "outsider"]
    
    class Config:
        case_sensitive = True


settings = Settings()