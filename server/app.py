from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict

from server.environment import HyperparamEnvironment
from src.models import HyperparamAction

app = FastAPI(
    title="ML Hyperparameter Tuner",
    description="RL environment for tuning ML hyperparameters",
    version="1.0.0"
)

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active environments
ENVS: Dict[str, HyperparamEnvironment] = {}
ENV_COUNTER = 0

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.post("/reset")
async def reset(difficulty: str = "easy"):
    """Reset environment"""
    global ENV_COUNTER
    
    env = HyperparamEnvironment(difficulty=difficulty)
    obs = env.reset()
    
    env_id = f"env_{ENV_COUNTER}"
    ENV_COUNTER += 1
    ENVS[env_id] = env
    
    return {
        "env_id": env_id,
        "observation": {
            "epoch": obs.epoch,
            "validation_accuracy": obs.validation_accuracy,
            "training_loss": obs.training_loss,
            "current_learning_rate": obs.current_learning_rate,
            "model_size_mb": obs.model_size_mb,
            "time_elapsed_seconds": obs.time_elapsed_seconds,
            "time_remaining_seconds": obs.time_remaining_seconds,
            "done": obs.done,
            "reward": obs.reward,
            "metadata": obs.metadata,
        }
    }

@app.post("/step")
async def step(env_id: str, action: dict):
    """Step environment"""
    if env_id not in ENVS:
        return {"error": f"Unknown env_id: {env_id}"}, 400
    
    env = ENVS[env_id]
    
    hp_action = HyperparamAction(
        learning_rate=float(action["learning_rate"]),
        batch_size=int(action["batch_size"]),
        weight_decay=float(action["weight_decay"]),
        optimizer=action["optimizer"],
    )
    
    obs = env.step(hp_action)
    
    return {
        "observation": {
            "epoch": obs.epoch,
            "validation_accuracy": obs.validation_accuracy,
            "training_loss": obs.training_loss,
            "current_learning_rate": obs.current_learning_rate,
            "model_size_mb": obs.model_size_mb,
            "time_elapsed_seconds": obs.time_elapsed_seconds,
            "time_remaining_seconds": obs.time_remaining_seconds,
            "done": obs.done,
            "reward": obs.reward,
            "metadata": obs.metadata,
        }
    }

@app.get("/state")
async def get_state(env_id: str):
    """Get state"""
    if env_id not in ENVS:
        return {"error": f"Unknown env_id: {env_id}"}, 400
    
    env = ENVS[env_id]
    state = env.state
    
    return {
        "episode_id": state.episode_id,
        "difficulty": state.difficulty,
        "dataset_name": state.dataset_name,
        "total_epochs": state.total_epochs,
        "current_epoch": state.current_epoch,
        "best_accuracy": state.best_accuracy,
        "total_configs_tried": state.total_configs_tried,
        "metadata": state.metadata,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)