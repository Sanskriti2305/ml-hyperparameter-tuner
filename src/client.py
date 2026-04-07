import httpx
from typing import Optional
from src.models import HyperparamAction, HyperparamObservation, HyperparamState

class HyperparamClient:
    """Client to talk to the environment"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.env_id: Optional[str] = None
        self.client = httpx.Client(timeout=300.0)  # 5 minute timeout
    
    def reset(self, difficulty: str = "easy") -> HyperparamObservation:
        """Start fresh training"""
        response = self.client.post(
            f"{self.base_url}/reset",
            params={"difficulty": difficulty},
            timeout=120.0  # 1 minute for reset
        )
        data = response.json()
        self.env_id = data["env_id"]
        
        obs_data = data["observation"]
        return HyperparamObservation(
            epoch=obs_data["epoch"],
            validation_accuracy=obs_data["validation_accuracy"],
            training_loss=obs_data["training_loss"],
            current_learning_rate=obs_data["current_learning_rate"],
            model_size_mb=obs_data["model_size_mb"],
            time_elapsed_seconds=obs_data["time_elapsed_seconds"],
            time_remaining_seconds=obs_data["time_remaining_seconds"],
            done=obs_data["done"],
            reward=obs_data["reward"],
            metadata=obs_data.get("metadata", {})
        )
    
    def step(self, action: HyperparamAction) -> HyperparamObservation:
        """Train with these settings"""
        if self.env_id is None:
            raise RuntimeError("Call reset() first!")
        
        response = self.client.post(
            f"{self.base_url}/step",
            params={"env_id": self.env_id},
            json={
                "learning_rate": action.learning_rate,
                "batch_size": action.batch_size,
                "weight_decay": action.weight_decay,
                "optimizer": action.optimizer,
            },
            timeout=300.0  # 5 minutes for step (training takes time!)
        )
        data = response.json()
        
        obs_data = data["observation"]
        return HyperparamObservation(
            epoch=obs_data["epoch"],
            validation_accuracy=obs_data["validation_accuracy"],
            training_loss=obs_data["training_loss"],
            current_learning_rate=obs_data["current_learning_rate"],
            model_size_mb=obs_data["model_size_mb"],
            time_elapsed_seconds=obs_data["time_elapsed_seconds"],
            time_remaining_seconds=obs_data["time_remaining_seconds"],
            done=obs_data["done"],
            reward=obs_data["reward"],
            metadata=obs_data.get("metadata", {})
        )
    
    def state(self) -> HyperparamState:
        """Get episode info"""
        if self.env_id is None:
            raise RuntimeError("Call reset() first!")
        
        response = self.client.get(
            f"{self.base_url}/state",
            params={"env_id": self.env_id},
            timeout=30.0  # 30 seconds for state
        )
        data = response.json()
        
        return HyperparamState(
            episode_id=data["episode_id"],
            difficulty=data["difficulty"],
            dataset_name=data["dataset_name"],
            total_epochs=data["total_epochs"],
            current_epoch=data["current_epoch"],
            best_accuracy=data["best_accuracy"],
            total_configs_tried=data["total_configs_tried"],
            metadata=data.get("metadata", {})
        )
    
    def close(self):
        """Close connection"""
        self.client.close()