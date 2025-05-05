"""
This file evaluates a trained DQN model in the jacoDiverseObjectEnv robotic environment.
It loads a pre-trained policy network, runs it for a specified number of episodes across different
scenarios, and tracks performance metrics including success rate, steps taken, user inputs,
error actions, and amplified actions. The final average metrics are calculated and saved to a JSON file,
providing a quantitative assessment of the DQN model's performance in shared autonomy tasks.
"""

import random
import numpy as np
from collections import namedtuple
import collections
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from config import *
import os
import json
import pybullet as pb
import datetime

# From pybullet_envs.bullet.jaco_diverse_object_gym_env import jacoDiverseObjectEnv
from jaco_env import jacoDiverseObjectEnv
from utils import get_screen
from networks import *

# If gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get stack size from model trained in learnDQN.py from the model name 
STACK_SIZE = int(modelPath.split("ss",1)[1].split("_rb",1)[0]) #[1,4,10]

# Number of different seeds
seeds_total = 1

# Directory to save trajectory data
modelname = "ARAS" if "ARAS" in modelPath else "DQN_baseline"
save_dir = f"./{modelname}_results"
os.makedirs(save_dir, exist_ok=True)

""" Evaluation of trained DQN model on different seeds"""
for seed in range(seeds_total):

    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    PATH = modelPath

    scores_window = collections.deque(maxlen=100)  # Last 100 scores

    try:
        pb.disconnect()  # Ensure no previous connections exist
    except:
        pass

    # isTest=True -> perform grasping on test set of objects. Currently just mug.
    # Select renders=True for GUI rendering
    env = jacoDiverseObjectEnv(actionRepeat=80, renders=RENDER, isDiscrete=True, maxSteps=70, dv=0.02,
                               AutoXDistance=False, AutoGrasp=True, width=64, height=64, numObjects=3,
                               numContainers=3, scenario=SCENARIO)
    env.reset()

    init_screen, _ = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n  # Get number of actions from gym action space

    policy_net = DQN(screen_height, screen_width, n_actions, stack_size=STACK_SIZE).to(device)
    # Load trained model for the policy network
    checkpoint = torch.load(PATH, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    # Success and failures
    s=0
    f=0

    # Data for saving trajectories
    trj_data = []
    episode_metrics_data_list = []

    for i_episode in range(EPISODE_NUMBER):
        env.reset()
        state, y_relative = get_screen(env)  # Adjusted to new function
        stacked_states = collections.deque(STACK_SIZE*[state], maxlen=STACK_SIZE)
        stacked_y_relatives = collections.deque(STACK_SIZE*[y_relative], maxlen=STACK_SIZE)  # Track y_relative
        
        # Initialize variables 
        gripper_trajectory = []
        steps = 0
        total_inputs = 0
        err_actions = 0
        ampl_actions = 0

        for t in count():
            steps += 1
            stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
            stacked_y_relatives_t = torch.cat(tuple(stacked_y_relatives), dim=1)  
            
            action = policy_net(stacked_states_t, stacked_y_relatives_t).max(1)[1].view(1, 1)
            action = action.item()
            _, reward, done, _ = env.step(action)  
            progress_reward, following_rew, total_reward = reward

            if action == 0 or action == 1: # left, right
                total_inputs += 1
            else:
                ampl_actions += 1

            if progress_reward < 0:  # If the gripper is moving away from the goal
                err_actions += 1

            # Record gripper trajectory
            gripper_pos = env._getGripper()[:2]  # x, y position
            gripper_trajectory.append(gripper_pos)

            # Observe new state and y_relative
            next_state, next_y_relative = get_screen(env)
            stacked_states.append(next_state)
            stacked_y_relatives.append(next_y_relative)  # Update stacked y_relatives
            
            if done:
                break 

        if total_reward==1:
            s=s+1 
        else: 
            f=f+1


        # Save trajectory data
        # episode_trajectory_data = {
        #     "episode": i_episode,
        #     "trajectory": [list(pos) for pos in gripper_trajectory],  # Convert each position to list
        #     "goal_position": env._mugPos[:2].tolist(),  # Convert numpy array to list
        #     "bin_position": env._containerPos[:2].tolist(),  # Convert numpy array to list
        # }
        # trj_data.append(episode_trajectory_data)

        # Save Metrics data
        episode_metrics_data = {
            "episode": i_episode,
            "steps": steps,
            "success": bool(total_reward == 1),
            "total_inputs": total_inputs / 0.05,
            "error_actions": err_actions,
            "amplified_actions": ampl_actions,
        }

        episode_metrics_data_list.append(episode_metrics_data)

        # Uncomment for immediate feedback after each episode   
        print("Episode: " + str(i_episode+1))
        print("Successed: " + str(s) + "\tFailures: " + str(f) + "\t\tSuccessRate: " + str(s/(i_episode + 1)))
        print("Steps: " + str(steps) + "\tTotal Inputs: " + str(total_inputs / 0.05) + "\tError Actions: " + str(err_actions) + "\tAmplified Actions: " + str(ampl_actions))
        print("-"*50)

    # Save metrics data to file
    save_path = os.path.join(save_dir, f"metrics_data_{SCENARIO}.json")
    with open(save_path, "w") as file:
        json.dump(episode_metrics_data_list, file)

    # # Save trajectory data to file after every seed
    # save_path = os.path.join(save_dir, f"trajectory_data_{seed}_{i_episode+1}.json")

    # # Convert NumPy types to native Python types
    # for entry in trj_data:
    #     entry["trajectory"] = [[float(x), float(y)] for x, y in entry["trajectory"]]
    #     entry["goal_position"] = [float(x) for x in entry["goal_position"]]
    #     entry["bin_position"] = [float(x) for x in entry["bin_position"]]
    #     entry["steps"] = int(entry["steps"])
    #     entry["success"] = bool(entry["success"])  # Convert numpy.bool_ to Python bool

    # with open(save_path, "w") as file:
    #     json.dump(trj_data, file)

    # Feedback after each
    print("For Seed " + str(seed+1) +": \t Successed: " + str(s) + "\tFailures: " + str(f) + "\t\tSuccessRate: " + str(s/(i_episode + 1)))
    
    # Print average metrics data 
    avg_steps = np.mean([data["steps"] for data in episode_metrics_data_list])
    avg_total_inputs = np.mean([data["total_inputs"] for data in episode_metrics_data_list])
    avg_error_actions = np.mean([data["error_actions"] for data in episode_metrics_data_list])
    avg_amplified_actions = np.mean([data["amplified_actions"] for data in episode_metrics_data_list])
    avg_success_rate = np.mean([data["success"] for data in episode_metrics_data_list])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_metrics = {
        "scenario": SCENARIO,
        "model_path": modelPath,
        "num_episodes": EPISODE_NUMBER,
        "success_rate": float(avg_success_rate),
        "avg_steps": float(avg_steps),
        "avg_user_inputs": float(avg_total_inputs),
        "avg_error_actions": float(avg_error_actions),
        "avg_amplified_actions": float(avg_amplified_actions),
        "timestamp": timestamp
    }

    save_path = os.path.join(save_dir, f"{modelname}_summary_{SCENARIO}_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(final_metrics, f)

    print("\n" + "=" * 50)
    print(f"FINAL RESULTS FOR SCENARIO: {SCENARIO}")
    print("=" * 50)
    print(f"Success Rate: {avg_success_rate:.4f} ({s}/{EPISODE_NUMBER})")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average User Inputs: {avg_total_inputs:.2f}")
    print(f"Average Error Actions: {avg_error_actions:.2f}")
    print(f"Average Amplified Actions: {avg_amplified_actions:.2f}")
    print("=" * 50)
    print(f"Summary results saved to: {save_path}")

