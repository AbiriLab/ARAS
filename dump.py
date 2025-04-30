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

env = jacoDiverseObjectEnv(actionRepeat=80, renders=RENDER, isDiscrete=True, maxSteps=70, dv=0.02,
                            AutoXDistance=False, AutoGrasp=True, width=64, height=64, numObjects=3,
                            numContainers=3, scenario=SCENARIO)
env.reset()

init_screen, _ = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n  # Get number of actions from gym action space

# Setup model architecture
policy_net = DQN(screen_height, screen_width, env.action_space.n, stack_size=4).to(device)
target_net = DQN(screen_height, screen_width, env.action_space.n, stack_size=4).to(device)

# Load checkpoint
checkpoint_path = "/home/tnlab/Projects/github/ARAS/models/DQN_baseline_bs64_ss4_rb30000_gamma0.3_decaylf5000_lr1e-05.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
target_net.load_state_dict(checkpoint['target_net_state_dict'])

# Optional: also restore optimizer if needed
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer.load_state_dict(checkpoint['optimizer_policy_net_state_dict'])

# Add noise to weights (e.g., 1% noise)
with torch.no_grad():
    for param in policy_net.parameters():
        param.add_(0.008 * torch.randn_like(param))

# Update target_net with noisy policy_net
target_net.load_state_dict(policy_net.state_dict())

# Save the modified model using the same checkpoint structure
torch.save({
    'policy_net_state_dict': policy_net.state_dict(),
    'target_net_state_dict': target_net.state_dict(),
    'optimizer_policy_net_state_dict': optimizer.state_dict()
}, "/home/tnlab/Projects/github/ARAS/models/DQN_baseline_v2_bs64_ss4_rb30000_gamma0.3_decaylf5000_lr1e-05.pt")

print("Noisy model saved.")
