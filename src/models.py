from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum

class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"

@dataclass
class HyperparamAction:
    """What the agent sends to us"""
    learning_rate: float
    batch_size: int
    weight_decay: float
    optimizer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HyperparamObservation:
    """What we send back after training one epoch"""
    epoch: int
    validation_accuracy: float
    training_loss: float
    current_learning_rate: float
    model_size_mb: float
    time_elapsed_seconds: float
    time_remaining_seconds: float
    done: bool
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HyperparamState:
    """Episode info"""
    episode_id: Optional[str]
    difficulty: str
    dataset_name: str
    total_epochs: int
    current_epoch: int
    best_accuracy: float
    total_configs_tried: int
    metadata: Dict[str, Any] = field(default_factory=dict)