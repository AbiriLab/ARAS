"""
This file implements a test harness for evaluating Hindsight Optimization in a shared autonomy setting.
It runs experiments using the jacoDiverseObjectEnv environment with a HindsightOptimizer that
generates synthetic user inputs and selects actions based on belief distributions over potential goals.
The script runs for a specified number of episodes, tracks metrics like success rate, steps taken,
user inputs, error actions, and amplified actions, and saves the final summary results to a JSON file.
"""

import os
import numpy as np
import random
import pybullet as pb
import pybullet_data
import time
from collections import deque
import json
import datetime

from jaco_env import jacoDiverseObjectEnv
from hindsight_optimizer import HindsightOptimizer
from config import SCENARIO, RENDER, EPISODE_NUMBER

def run_hindsight_optimization():
    """Run the hindsight optimization with synthetic user data."""
    
    np.random.seed(42)
    random.seed(42)
    
    try:
        pb.disconnect()
    except:
        pass
    
    env = jacoDiverseObjectEnv(
        actionRepeat=80, 
        renders=RENDER, 
        isDiscrete=True, 
        maxSteps=70, 
        dv=0.02,
        AutoXDistance=False, 
        AutoGrasp=True, 
        width=64, 
        height=64, 
        numObjects=3,
        numContainers=3, 
        scenario=SCENARIO
    )
    
    env.reset()
    
    optimizer = HindsightOptimizer(env)
    
    success_count = 0
    failures = 0
    episode_metrics = []
    
    save_dir = "./hindsight_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Hindsight Optimization Experiments")
    print(f"Scenario: {SCENARIO}")
    print(f"Episodes: {EPISODE_NUMBER}")
    print("-" * 50)
    
    for episode in range(EPISODE_NUMBER):
        env.reset()
        optimizer.reset()
        
        steps = 0
        total_inputs = 0
        error_actions = 0
        amplified_actions = 0
        gripper_trajectory = []
        
        done = False
        while not done:
            user_input = optimizer.generate_synthetic_user_input()

            action = optimizer.select_action(user_input)

            if action == 0 or action == 1: # left, right
                total_inputs += 1
            else:
                amplified_actions += 1
            
            gripper_pos = env._getGripper()[:2]
            gripper_trajectory.append(list(gripper_pos))
            
            _, reward, done, info = env.step(action)
            
            progress_reward, following_rew, total_reward = reward
            
            if progress_reward < 0:
                error_actions += 1
            
            steps += 1
            
            if RENDER:
                time.sleep(0.05)
        
        task_success = info.get('task_success', 0)
        if task_success > 0:
            success_count += 1
        else:
            failures += 1
        
        episode_data = {
            "episode": episode,
            "steps": steps,
            "success": bool(task_success > 0),
            "user_inputs": total_inputs / 0.05,
            "error_actions": error_actions,
            "amplified_actions": amplified_actions,
            "scenario": SCENARIO,
            "trajectory": gripper_trajectory,
            "goal_position": env._mugPos[:2].tolist() if hasattr(env, '_mugPos') else None,
            "container_position": env._containerPos[:2].tolist() if hasattr(env, '_containerPos') else None,
        }
        episode_metrics.append(episode_data)
        
        print(f"Episode {episode+1}/{EPISODE_NUMBER} - Steps: {steps}, Success: {bool(task_success)}")
        print(f"Total Inputs: {total_inputs}, Error Actions: {error_actions}, Amplified Actions: {amplified_actions}")
        print(f"Current Success Rate: {success_count/(episode+1):.2f}")
        print("-" * 50)
    
    # Calculate final metrics
    success_rate = success_count / EPISODE_NUMBER
    avg_steps = np.mean([data["steps"] for data in episode_metrics])
    avg_inputs = np.mean([data["user_inputs"] for data in episode_metrics])
    avg_errors = np.mean([data["error_actions"] for data in episode_metrics])
    avg_amplified = np.mean([data["amplified_actions"] for data in episode_metrics])
    
    # Save only the final average metrics
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_metrics = {
        "scenario": SCENARIO,
        "num_episodes": EPISODE_NUMBER,
        "success_rate": float(success_rate),
        "avg_steps": float(avg_steps),
        "avg_user_inputs": float(avg_inputs),
        "avg_error_actions": float(avg_errors),
        "avg_amplified_actions": float(avg_amplified),
        "timestamp": timestamp
    }
    
    save_path = os.path.join(save_dir, f"hindsight_summary_{SCENARIO}_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(final_metrics, f)
    
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS FOR SCENARIO: {SCENARIO}")
    print("=" * 50)
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{EPISODE_NUMBER})")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average User Inputs: {avg_inputs:.2f}")
    print(f"Average Error Actions: {avg_errors:.2f}")
    print(f"Average Apmplified Actions: {avg_amplified:.2f}")
    print("=" * 50)
    print(f"Summary results saved to: {save_path}")
    
    env.close()
    
    return success_rate, avg_steps, avg_inputs, avg_errors

if __name__ == "__main__":
    run_hindsight_optimization()
