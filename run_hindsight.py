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
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Ensure no previous connections exist
    try:
        pb.disconnect()
    except:
        pass
    
    # Create environment using parameters from config
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
    
    # Reset environment first to create objects and containers
    env.reset()
    
    # Create hindsight optimizer
    optimizer = HindsightOptimizer(env)
    
    # Metrics tracking
    success_count = 0
    failures = 0
    episode_metrics = []
    
    # Directory to save metrics
    save_dir = "./hindsight_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Hindsight Optimization Experiments")
    print(f"Scenario: {SCENARIO}")
    print(f"Episodes: {EPISODE_NUMBER}")
    print("-" * 50)
    
    # Run episodes
    for episode in range(EPISODE_NUMBER):
        # Reset environment and optimizer
        env.reset()
        optimizer.reset()
        
        # Initialize metrics for this episode
        steps = 0
        user_inputs = 0
        error_actions = 0
        gripper_trajectory = []
        
        # Run episode until termination
        done = False
        while not done:
            # Get synthetic user input
            user_input = optimizer.generate_synthetic_user_input()
            if user_input != "neutral":
                user_inputs += 1
            
            # Select action using hindsight optimization
            action = optimizer.select_action(user_input)
            
            # Get debug information
            debug_info = optimizer.debug_info()
            
            # Print user input and selected action
            print(f"Step {steps+1}: User Input: {user_input} → Action: {optimizer.action_mapping.get(action, action)}")
            
            # Print additional debug information if verbose mode is enabled
            if RENDER:  # Only print details in render mode to avoid cluttering output
                phase = debug_info.get("phase", "UNKNOWN")
                delta_y = debug_info.get("delta_y", 0)
                normalized_delta_y = debug_info.get("normalized_delta_y", 0)
                
                # Print phase-specific information
                if phase == "PICKUP":
                    top_beliefs = debug_info.get("top_object_beliefs", [])
                    target_idx = debug_info.get("target_object_idx", -1)
                    
                    # Format beliefs for display
                    belief_str = ", ".join([f"Obj{idx}: {belief:.2f}" for idx, belief in top_beliefs])
                    
                    # Show target and current positions
                    gripper_pos = debug_info.get("gripper_position", [0, 0])
                    target_pos = debug_info.get("target_object_position", [0, 0])
                    
                    print(f"  Phase: {phase}, Target Obj: {target_idx}")
                    print(f"  Gripper: ({gripper_pos[0]:.2f}, {gripper_pos[1]:.2f}), Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
                    print(f"  Delta-Y: {delta_y:.3f}, Normalized: {normalized_delta_y} → Input: {user_input}")
                    print(f"  Top Beliefs: {belief_str}")
                else:
                    top_beliefs = debug_info.get("top_container_beliefs", [])
                    target_idx = debug_info.get("target_container_idx", -1)
                    
                    # Format beliefs for display
                    belief_str = ", ".join([f"Cont{idx}: {belief:.2f}" for idx, belief in top_beliefs])
                    
                    # Show target and current positions
                    gripper_pos = debug_info.get("gripper_position", [0, 0])
                    target_pos = debug_info.get("target_container_position", [0, 0])
                    
                    print(f"  Phase: {phase}, Target Container: {target_idx}")
                    print(f"  Gripper: ({gripper_pos[0]:.2f}, {gripper_pos[1]:.2f}), Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
                    print(f"  Delta-Y: {delta_y:.3f}, Normalized: {normalized_delta_y} → Input: {user_input}")
                    print(f"  Top Beliefs: {belief_str}")
                print()  # Empty line for readability
            
            # Get current positions for trajectory tracking
            gripper_pos = env._getGripper()[:2]
            gripper_trajectory.append(list(gripper_pos))
            
            # Take action in environment
            _, reward, done, info = env.step(action)
            
            # Extract component rewards
            progress_reward, following_rew, total_reward = reward
            
            # Track error actions (when we move away from goal)
            if progress_reward < 0:
                error_actions += 1
            
            # Increment step counter
            steps += 1
            
            # Render if enabled
            if RENDER:
                time.sleep(0.05)
        
        # Check success
        task_success = info.get('task_success', 0)
        if task_success > 0:
            success_count += 1
        else:
            failures += 1
        
        # Store episode metrics
        episode_data = {
            "episode": episode,
            "steps": steps,
            "success": bool(task_success > 0),
            "user_inputs": user_inputs,
            "error_actions": error_actions,
            "scenario": SCENARIO,
            "trajectory": gripper_trajectory,
            "goal_position": env._mugPos[:2].tolist() if hasattr(env, '_mugPos') else None,
            "container_position": env._containerPos[:2].tolist() if hasattr(env, '_containerPos') else None,
        }
        episode_metrics.append(episode_data)
        
        # Print episode results
        print(f"Episode {episode+1}/{EPISODE_NUMBER} - Steps: {steps}, Success: {bool(task_success)}")
        print(f"User Inputs: {user_inputs}, Error Actions: {error_actions}")
        print(f"Current Success Rate: {success_count/(episode+1):.2f}")
        print("-" * 50)
        
        # Save metrics every 100 episodes
        if (episode + 1) % 100 == 0 or episode == EPISODE_NUMBER - 1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"hindsight_metrics_{SCENARIO}_{timestamp}.json")
            with open(save_path, "w") as f:
                json.dump(episode_metrics, f)
    
    # Final stats
    success_rate = success_count / EPISODE_NUMBER
    avg_steps = np.mean([data["steps"] for data in episode_metrics])
    avg_inputs = np.mean([data["user_inputs"] for data in episode_metrics])
    avg_errors = np.mean([data["error_actions"] for data in episode_metrics])
    
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS FOR SCENARIO: {SCENARIO}")
    print("=" * 50)
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{EPISODE_NUMBER})")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average User Inputs: {avg_inputs:.2f}")
    print(f"Average Error Actions: {avg_errors:.2f}")
    print("=" * 50)
    
    # Close environment
    env.close()
    
    return success_rate, avg_steps, avg_inputs, avg_errors

if __name__ == "__main__":
    run_hindsight_optimization()
