#!/usr/bin/env python3
"""
Inference Script for ML Hyperparameter Tuner
Compliant with OpenEnv stdout format specification.
Runs 3 tasks: easy, medium, hard
"""

import os
import sys
import json
import re
from typing import List, Optional

from openai import OpenAI

from src.client import HyperparamClient
from src.models import HyperparamAction

# MANDATORY environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK = "ml-hyperparameter-tuner"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an AI agent optimizing machine learning hyperparameters.
Based on the current training state (accuracy, loss, epoch), suggest the next hyperparameters.
Reply with a JSON object containing exactly: learning_rate, batch_size, weight_decay, optimizer.
Example: {"learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0, "optimizer": "adam"}
Only output the JSON object, nothing else."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def clamp_score(raw: float) -> float:
    """Always return a score strictly inside (0.0, 1.0) — never exactly 0.0 or 1.0"""
    return min(0.999, max(0.001, float(raw)))


def parse_llm_action(response_text: str) -> HyperparamAction:
    try:
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            action_dict = json.loads(json_match.group())
            return HyperparamAction(
                learning_rate=float(action_dict.get("learning_rate", 0.001)),
                batch_size=int(action_dict.get("batch_size", 32)),
                weight_decay=float(action_dict.get("weight_decay", 0.0)),
                optimizer=action_dict.get("optimizer", "adam"),
            )
    except Exception:
        pass
    return HyperparamAction(
        learning_rate=0.001, batch_size=32, weight_decay=0.0, optimizer="adam"
    )


def get_llm_action(llm: OpenAI, obs, difficulty: str, step: int):
    user_prompt = f"""Current state (Step {step}, difficulty={difficulty}):
- Validation Accuracy: {obs.validation_accuracy:.4f}
- Training Loss: {obs.training_loss:.4f}
- Learning Rate: {obs.current_learning_rate:.6f}
- Epoch: {obs.epoch}
- Time elapsed: {obs.time_elapsed_seconds:.1f}s

Suggest next hyperparameters as JSON."""

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = response.choices[0].message.content or ""
        action = parse_llm_action(response_text)
    except Exception:
        action = HyperparamAction(
            learning_rate=0.001, batch_size=32, weight_decay=0.0, optimizer="adam"
        )

    action_str = (
        f"lr={action.learning_rate:.4f},"
        f"bs={action.batch_size},"
        f"wd={action.weight_decay:.4f},"
        f"opt={action.optimizer}"
    )
    return action, action_str


def run_task(llm: OpenAI, difficulty: str) -> None:
    client_env = HyperparamClient(base_url=ENV_BASE_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.001  # never 0.0 — safe default if everything fails
    success = False
    obs = None

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = client_env.reset(difficulty=difficulty)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action, action_str = get_llm_action(llm, obs, difficulty, step)

            try:
                obs = client_env.step(action)
                reward = float(obs.reward)
                done = bool(obs.done)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Extract grader_score — fallback to 0.1 so it's never exactly 0.0
        if obs is not None and obs.metadata:
            raw_score = obs.metadata.get("grader_score", 0.1)
        else:
            raw_score = 0.1

        score = clamp_score(raw_score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] task={difficulty} {str(e)}", file=sys.stderr, flush=True)
        score = 0.001  # strictly in range even on hard failure
        success = False

    finally:
        try:
            client_env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    if not API_KEY:
        print("[ERROR] HF_TOKEN (or API_KEY) is not set!", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for difficulty in TASKS:
        run_task(llm, difficulty)


if __name__ == "__main__":
    main()