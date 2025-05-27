from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
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

    def to_dict(self):
        """Converts DronePosition object to a dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DroneData:
    """Represents telemetry and features used for model prediction"""
    position: DronePosition
    velocity: Dict[str, float]
    acceleration: Dict[str, float]
    orientation: Dict[str, float]
    angular_velocity: Dict[str, float]
    battery: float
    signal_strength: float
    packet_loss: float
    latency: float
    # Add more features as needed

    def to_feature_vector(self) -> List[float]:
        """Convert drone data to feature vector for model input"""
        base_features = [
            self.position.x, self.position.y, self.position.z,
            self.velocity.get("x", 0), self.velocity.get("y", 0), self.velocity.get("z", 0),
            self.acceleration.get("x", 0), self.acceleration.get("y", 0), self.acceleration.get("z", 0),
            self.orientation.get("pitch", 0), self.orientation.get("roll", 0), self.orientation.get("yaw", 0),
            self.angular_velocity.get("x", 0), self.angular_velocity.get("y", 0), self.angular_velocity.get("z", 0),
            self.battery, self.signal_strength, self.packet_loss, self.latency
        ]
        # Add flight history features (last 10 positions) - This part was not fully implemented in the original DroneData.to_feature_vector.
        # If flight_history is intended for DroneData, you'll need to add it as a field to DroneData as well.
        # For now, I'm keeping it consistent with the provided DroneData definition.
        return base_features


@dataclass
class DetectionResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    confidence: float
    detection_type: Optional[str] = None  # 'mitm', 'outsider', or None if no anomaly
    details: Optional[Dict[str, Any]] = None


@dataclass
class OutsiderDroneData:
    """Represents outsider drone data with flight history"""
    drone_id: str
    position: DronePosition
    velocity: Dict[str, float]
    acceleration: Dict[str, float]
    orientation: Dict[str, float]
    angular_velocity: Dict[str, float]
    battery: float
    signal_strength: float
    packet_loss: float
    latency: float
    flight_history: List[DronePosition]  # Historical positions
    authentication_timestamp: datetime
    
    def to_feature_vector(self) -> List[float]:
        """Convert outsider drone data to feature vector for model input"""
        base_features = [
            self.position.x, self.position.y, self.position.z,
            self.velocity.get("x", 0), self.velocity.get("y", 0), self.velocity.get("z", 0),
            self.acceleration.get("x", 0), self.acceleration.get("y", 0), self.acceleration.get("z", 0),
            self.orientation.get("pitch", 0), self.orientation.get("roll", 0), self.orientation.get("yaw", 0),
            self.angular_velocity.get("x", 0), self.angular_velocity.get("y", 0), self.angular_velocity.get("z", 0),
            self.battery, self.signal_strength, self.packet_loss, self.latency
        ]
        
        # Add flight history features (last 10 positions)
        history_features = []
        for i in range(min(10, len(self.flight_history))):
            pos = self.flight_history[-(i+1)]
            history_features.extend([pos.x, pos.y, pos.z])
        
        # Pad with zeros if less than 10 historical positions
        while len(history_features) < 30:  # 10 positions * 3 coordinates
            history_features.append(0.0)
            
        return base_features + history_features

    def to_dict(self):
        """Convert OutsiderDroneData object to a dictionary for JSON serialization."""
        return {
            "drone_id": self.drone_id,
            "position": self.position.to_dict(),  # Use to_dict for nested DronePosition
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "orientation": self.orientation,
            "angular_velocity": self.angular_velocity,
            "battery": self.battery,
            "signal_strength": self.signal_strength,
            "packet_loss": self.packet_loss,
            "latency": self.latency,
            "flight_history": [pos.to_dict() for pos in self.flight_history],  # Convert each DronePosition in history
            "authentication_timestamp": self.authentication_timestamp.isoformat()
        }


@dataclass
class AuthenticationRequest:
    """Authentication request from outsider drone"""
    drone_id: str
    flight_history: List[DronePosition]
    request_timestamp: datetime
    source_position: DronePosition


@dataclass
class AuthenticationResponse:
    """Response to authentication request"""
    drone_id: str
    is_authenticated: bool
    is_outsider: bool
    confidence: float
    response_timestamp: datetime
    communication_blocked: bool = False