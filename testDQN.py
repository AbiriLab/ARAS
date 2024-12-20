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

# From pybullet_envs.bullet.jaco_diverse_object_gym_env import jacoDiverseObjectEnv
from jaco_env import jacoDiverseObjectEnv
from utils import get_screen
from DQN_net import DQN

# If gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Select trained model
# modelPath = '/home/ali/Projects/RobaticRL/extended/main/phase2/models/policyDQN_phase2_bs64_ss4_rb25000_gamma0.99_decaylf20000_lr0.0001.pt'
# modelPath = "/home/ali/Projects/RobaticRL/extended/main/phase2/models/FullAuto2obj_bs64_ss4_rb30000_gamma0.99_decaylf100000.0_lr0.001.pt"

# Get stack size from model trained in learnDQN.py from the model name 
STACK_SIZE = int(modelPath.split("ss",1)[1].split("_rb",1)[0]) #[1,4,10]

# Number of different seeds
seeds_total = 5

# Directory to save trajectory data
save_dir = "./trajectory_test_data"
os.makedirs(save_dir, exist_ok=True)

""" Evaluation of trained DQN model on different seeds"""
for seed in range(seeds_total):

    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    PATH = modelPath

    # Number of trials per seed
    episode = 1000   #[100,500,1000]
    
    scores_window = collections.deque(maxlen=100)  # Last 100 scores

    try:
        pb.disconnect()  # Ensure no previous connections exist
    except:
        pass

    # isTest=True -> perform grasping on test set of objects. Currently just mug.
    # Select renders=True for GUI rendering
    env = jacoDiverseObjectEnv(actionRepeat=80, renders=True, isDiscrete=True, maxSteps=70, dv=0.02,
                            AutoXDistance=False, AutoGrasp=True, width=64, height=64, numObjects=3, numContainers=3)
    env.reset()

    init_screen, _ = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n  # Get number of actions from gym action space
    # policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    policy_net = DQN(screen_height, screen_width, n_actions, stack_size=STACK_SIZE).to(device)
    # Load trained model for the policy network
    checkpoint = torch.load(PATH, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    # Success and failures
    s=0
    f=0

    # Data for saving trajectories
    trj_data = []

    for i_episode in range(episode):
        env.reset()
        state, y_relative = get_screen(env)  # Adjusted to new function
        stacked_states = collections.deque(STACK_SIZE*[state], maxlen=STACK_SIZE)
        stacked_y_relatives = collections.deque(STACK_SIZE*[y_relative], maxlen=STACK_SIZE)  # Track y_relative
        
        # Track trajectory
        gripper_trajectory = []

        steps = 0
        for t in count():
            steps += 1
            stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
            stacked_y_relatives_t = torch.cat(tuple(stacked_y_relatives), dim=1)  
            
            # Select and perform an action
            action = policy_net(stacked_states_t, stacked_y_relatives_t).max(1)[1].view(1, 1)
            _, reward, done, _ = env.step(action.item())
            # print(action)
            # Record gripper trajectory
            gripper_pos = env._getGripper()[:2]  # x, y position
            gripper_trajectory.append(gripper_pos)

            # Observe new state and y_relative
            next_state, next_y_relative = get_screen(env)
            stacked_states.append(next_state)
            stacked_y_relatives.append(next_y_relative)  # Update stacked y_relatives
            
            if done:
                break

        print("# of Steps:", steps)    

        if reward==1:
            s=s+1
            
        else: 
            f=f+1


        # Save episode data
        episode_data = {
            "episode": i_episode,
            "trajectory": [list(pos) for pos in gripper_trajectory],  # Convert each position to list
            "goal_position": env._mugPos[:2].tolist(),  # Convert numpy array to list
            "bin_position": env._containerPos[:2].tolist(),  # Convert numpy array to list
            "steps": steps,  # Number of steps
            "success": bool(reward == 1)  # Success if reward indicates task completion
        }
        trj_data.append(episode_data)

        # Uncomment for immediate feedback after each episode   
        print("Successed: " + str(s) + "\tFailures: " + str(f) + "\t\tSuccessRate: " + str(s/(i_episode + 1)))

    # Save trajectory data to file after every seed
    save_path = os.path.join(save_dir, f"trajectory_data_{seed}_{i_episode+1}.json")

    # Convert NumPy types to native Python types
    for entry in trj_data:
        entry["trajectory"] = [[float(x), float(y)] for x, y in entry["trajectory"]]
        entry["goal_position"] = [float(x) for x in entry["goal_position"]]
        entry["bin_position"] = [float(x) for x in entry["bin_position"]]
        entry["steps"] = int(entry["steps"])
        entry["success"] = bool(entry["success"])  # Convert numpy.bool_ to Python bool

    with open(save_path, "w") as file:
        json.dump(trj_data, file)


    # Feedback after each
    print("For Seed " + str(seed+1) +": \t Successed: " + str(s) + "\tFailures: " + str(f) + "\t\tSuccessRate: " + str(s/(i_episode + 1)))
