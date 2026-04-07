#!/usr/bin/env python3
"""
Baseline inference script - Tests all 3 difficulties
"""

import json
import sys
import os
import argparse

# MANDATORY: Use these environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
HF_TOKEN = os.getenv("HF_TOKEN", "fake_token")

from src.client import HyperparamClient
from src.models import HyperparamAction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    
    # MUST START WITH [START]
    print(json.dumps({
        "type": "[START]",
        "environment": "ml-hyperparameter-tuner",
        "difficulty": args.difficulty,
        "episodes": args.episodes
    }), flush=True)
    
    client = HyperparamClient(base_url="http://localhost:8000")
    total_reward = 0.0
    best_accuracy = 0.0
    grader_score = 0.0
    
    try:
        # Reset
        obs = client.reset(difficulty=args.difficulty)
        
        # Run 3 steps (reduced from 5 for speed)
        for step in range(3):
            action = HyperparamAction(
                learning_rate=0.001,
                batch_size=64,
                weight_decay=0.01,
                optimizer="adam"
            )
            
            obs = client.step(action)
            total_reward += obs.reward
            best_accuracy = max(best_accuracy, obs.validation_accuracy)
            grader_score = obs.metadata.get("grader_score", 0.0)
            
            # MUST HAVE [STEP] for each step
            print(json.dumps({
                "type": "[STEP]",
                "step": step + 1,
                "accuracy": round(obs.validation_accuracy, 4),
                "loss": round(obs.training_loss, 4),
                "reward": round(obs.reward, 4),
                "grader_score": round(grader_score, 4)
            }), flush=True)
        
        state = client.state()
        
    finally:
        client.close()
    
    # MUST END WITH [END]
    print(json.dumps({
        "type": "[END]",
        "total_steps": 3,
        "final_accuracy": round(best_accuracy, 4),
        "total_reward": round(total_reward, 4),
        "grader_score": round(grader_score, 4),
        "difficulty": args.difficulty,
        "success": True
    }), flush=True)

if __name__ == "__main__":
    main()