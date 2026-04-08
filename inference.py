#!/usr/bin/env python3
"""
Baseline inference script for ML Hyperparameter Tuner
Follows strict OpenEnv stdout format specification
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI

from src.client import HyperparamClient
from src.models import HyperparamAction

# MANDATORY environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 200

SYSTEM_PROMPT = """You are an AI agent optimizing machine learning hyperparameters.
Based on the current training state (accuracy, loss, epoch), suggest the next hyperparameters.
Reply with a JSON object containing: learning_rate, batch_size, weight_decay, optimizer.
Example: {"learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0, "optimizer": "adam"}"""


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def parse_llm_action(response_text: str) -> HyperparamAction:
    """Parse LLM JSON response to HyperparamAction"""
    try:
        import json
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            action_dict = json.loads(json_match.group())
            return HyperparamAction(
                learning_rate=float(action_dict.get("learning_rate", 0.001)),
                batch_size=int(action_dict.get("batch_size", 32)),
                weight_decay=float(action_dict.get("weight_decay", 0.0)),
                optimizer=action_dict.get("optimizer", "adam")
            )
    except Exception as e:
        pass
    
    # Fallback to defaults
    return HyperparamAction(
        learning_rate=0.001,
        batch_size=32,
        weight_decay=0.0,
        optimizer="adam"
    )


def get_llm_action(client_llm: OpenAI, obs, difficulty: str, step: int) -> tuple:
    """Get action from LLM and return (action, action_str)"""
    
    user_prompt = f"""Current state (Step {step}, {difficulty}):
- Validation Accuracy: {obs.validation_accuracy:.4f}
- Training Loss: {obs.training_loss:.4f}
- Learning Rate: {obs.current_learning_rate:.6f}
- Epoch: {obs.epoch}
- Time: {obs.time_elapsed_seconds:.1f}s elapsed

Suggest next hyperparameters as JSON."""

    try:
        response = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = response.choices[0].message.content
        action = parse_llm_action(response_text)
        action_str = f"lr={action.learning_rate:.4f},bs={action.batch_size},wd={action.weight_decay:.4f},opt={action.optimizer}"
        return action, action_str
    except Exception as e:
        # Fallback action
        action = HyperparamAction(learning_rate=0.001, batch_size=32, weight_decay=0.0, optimizer="adam")
        action_str = f"lr={action.learning_rate:.4f},bs={action.batch_size},wd={action.weight_decay:.4f},opt={action.optimizer}"
        return action, action_str


def main():
    # Validate required env vars
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set!", file=sys.stderr)
        sys.exit(1)
    
    client_llm = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
    client_env = HyperparamClient(base_url=API_BASE_URL)
    
    difficulty = os.getenv("TASK_NAME", "easy")
    
    # Emit [START]
    log_start(task=difficulty, env="ml-hyperparameter-tuner", model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        # Reset environment
        obs = client_env.reset(difficulty=difficulty)
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            
            # Get action from LLM
            action, action_str = get_llm_action(client_llm, obs, difficulty, step)
            
            # Step environment
            obs = client_env.step(action)
            
            reward = obs.reward
            done = obs.done
            error = None
            
            rewards.append(reward)
            steps_taken = step
            
            # Emit [STEP]
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            if done:
                break
        
        # Calculate score (normalized to [0, 1])
        grader_score = obs.metadata.get("grader_score", 0.0)
        score = min(max(grader_score, 0.0), 1.0)
        success = score >= 0.5
        
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        success = False
    finally:
        try:
            client_env.close()
        except:
            pass
        
        # Emit [END]
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
