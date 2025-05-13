from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime


class ModelType(str, Enum):
    """Available AI model types"""
    KNN = "knn"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    LSTM = "lstm"
    RNN = "rnn"
    RANDOM_FOREST = "random_forest"


class SimulationType(str, Enum):
    """Available simulation types"""
    NORMAL = "normal"
    MITM = "mitm"
    OUTSIDER = "outsider"


@dataclass
class DronePosition:
    """Represents a drone's position in 3D space"""
    x: float
    y: float 
    z: float
    timestamp: datetime


@dataclass
class DroneData:
    """Represents telemetry and features used for model prediction"""
    position: DronePosition
    velocity: Dict[str, float]
    orientation: Dict[str, float]
    battery: float
    signal_strength: float
    packet_loss: float
    latency: float
    # Add more features as needed
    
    def to_feature_vector(self) -> List[float]:
        """Convert drone data to feature vector for model input"""
        return [
            self.position.x, self.position.y, self.position.z,
            self.velocity.get("x", 0), self.velocity.get("y", 0), self.velocity.get("z", 0),
            self.orientation.get("pitch", 0), self.orientation.get("roll", 0), self.orientation.get("yaw", 0),
            self.battery, self.signal_strength, self.packet_loss, self.latency
        ]


@dataclass
class DetectionResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    confidence: float
    detection_type: Optional[str] = None  # 'mitm', 'outsider', or None if no anomaly
    details: Optional[Dict[str, Any]] = None