"""
Baseline DQN code from: 
    https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl.ipynb
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://source.coderefinery.org/JosteinDanielsen/msc-thesis/-/tree/master/kuka_eye_dqn

"""

import random
import numpy as np
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from env_extended import jacoDiverseObjectEnv
from utils import DQN, ReplayMemory, get_screen
import pybullet as pb
from config import *
import os

# Set seed fixed for reproducibility, also try different seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Load env
env = jacoDiverseObjectEnv(actionRepeat=80, renders=False, isDiscrete=True, maxSteps=30, dv=0.02,
                           AutoXDistance=True, AutoGrasp=True, width=64, height=64, numObjects=3)

env.cid = pb.connect(pb.DIRECT)

# Choose system (CPU/GPU), depending if Nvidia Cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



def select_action(state, relative_position, i_episode):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - (i_episode / EPS_DECAY_LAST_FRAME))
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state, relative_position).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def log(m):
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    mess = "" + str(ct) + " - " + str(m)
    LOGFILE_POINTER.write(mess + "\n")
    LOGFILE_POINTER.flush()
    if LOG_ON_SCREEN:
      print(mess)
    return ts

'''
Training loop
'''
# Update network
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            [s[0] for s in batch.next_state if s is not None])), 
                                  device=device, dtype=torch.bool)
    
    # Assuming your Transition namedtuple or equivalent structure has 'next_state' as a tuple of (image, y_relative)
    non_final_next_states = [s for s in batch.next_state if s is not None and s[0] is not None]

    # Separate the image and y_relative components for non-final next states
    non_final_next_states_images = torch.cat([s[0] for s in non_final_next_states])
    non_final_next_states_y_relative = torch.cat([s[1] for s in non_final_next_states])


    # Similarly, separate the image and y_relative components for the current state batch
    state_batch_images = torch.cat([s[0] for s in batch.state])
    state_batch_y_relative = torch.cat([s[1] for s in batch.state])

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Adjust the network forward pass to include y_relative
    state_action_values = policy_net(state_batch_images, state_batch_y_relative).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if len(non_final_next_states_images) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states_images, non_final_next_states_y_relative).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()




'''
Main training loop
'''
              
writer = SummaryWriter()
total_rewards = []
ten_rewards = 0
best_mean_reward = None
start_time = timeit.default_timer()

env.reset()


'''
Training
*Instantiate DQN.
*Epsilon greedy action selection with epsilon decay:
probability of choosing a random action will start at EPS_START and 
will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
'''

# Get screen size so that we can initialize layers correctly based on shape
# returned from pybullet (128, 128, 3).
init_screen, _ = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

eps_threshold = 0
STACK_SIZE = 0
memory = None
policy_net = None
target_net = None

LOGFILE=""
LOGFILE_POINTER = None
LOG_ON_SCREEN = False


if __name__ == "__main__":
    import datetime
    import optparse
    ts_start = datetime.datetime.now()
    parser = optparse.OptionParser()

    parser.add_option('-l', '--logfile',
                    action="store", 
                    dest="logfile_name",
                    help="Name of logfile", 
                    default="")
    parser.add_option('-d', '--detail_level',
                    action="store", 
                    dest="detail_level",
                    help="Level of detail (a,e,p)", 
                    default="a")
    parser.add_option('-a', '--logging on both screen and logfile (default)',
                    action="store", 
                    dest="both",
                    help="Logging on screen and logfile", 
                    default=True)

    options, args = parser.parse_args()

    LOG_ON_SCREEN = True if options.both == True else False

    # Init replay buffer, policy net and target net
    STACK_SIZE = Stack_Size
    memory = ReplayMemory(REPLAY_BUFFER_SIZE, Transition)
    policy_net = DQN(screen_height, screen_width, n_actions, stack_size=STACK_SIZE).to(device)
    target_net = DQN(screen_height, screen_width, n_actions, stack_size=Stack_Size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # Use ADAM as optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)


    # Assume your pre-trained model's file path
    PRETRAINED_MODEL_PATH = '/home/ali/Projects/RobaticRL/extended/main/phase2/models/FullAuto2obj_bs64_ss4_rb30000_gamma0.99_decaylf100000.0_lr0.001.pt'

    # Check if the pretrained model file exists and load it
    if os.path.isfile(PRETRAINED_MODEL_PATH):
        print("Loading pre-trained model from", PRETRAINED_MODEL_PATH)
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        
        # Optionally load the optimizer state as well
        optimizer.load_state_dict(checkpoint['optimizer_policy_net_state_dict'])
        
        print("Pre-trained model loaded successfully!")
    else:
        print("Pre-trained model not found. Starting training from scratch.")

    # Now proceed with your training loop


    # Name of saved .log file and .pt model
    filename = "bs"+str(BATCH_SIZE)+"_ss" + str(STACK_SIZE) + "_rb" + str(REPLAY_BUFFER_SIZE)+"_gamma"+str(GAMMA)+"_decaylf"+str(EPS_DECAY_LAST_FRAME)+"_lr"+str(LEARNING_RATE)

    if options.logfile_name == "":
        LOGFILE = f"logs/{MODEL_NAME}_{filename}.log"
        print("Logging progress to: " + LOGFILE)
        print("Detail level of logging is set to \'" + options.detail_level + "' - ", end = "")
    if (options.detail_level == 'p'):
        print("Print progress in terms of increased mean rewards")
    elif (options.detail_level == 'e'):
        print("Print progress in terms of timestamped Epoc's ")
    else:
        print("Print all logging info")

    # Path of saved model
    PATH = f"models/{MODEL_NAME}_{filename}.pt" 
    if LOGFILE_POINTER == None:
        LOGFILE_POINTER = open(LOGFILE, "w")

    old_reward_ts = log("Starting learning loop ...\n==================================================================")
    log("Using Framebuffer of: " + str(STACK_SIZE) + " frames\t and \tReplayBufferSize: " +  str(REPLAY_BUFFER_SIZE))
    log("Loglevel is set to: " + str(options.detail_level))

    old_epoch_ts = -1

  # Begin training
    for i_episode in range(num_episodes):
        # Print Epochs
        if i_episode % 10 == 0:
            ct = datetime.datetime.now()
            new_epoch_ts = ct.timestamp()
            if old_epoch_ts > 0:
                diff = new_epoch_ts - old_epoch_ts
            else:
                diff = ""
            if options.detail_level != 'p':
                old_epoch_ts = log("Epoc #\t" + str(i_episode) + "\t " + str(diff))

        # Initialize the environment and state
        env.reset()
        state, y_relative = get_screen(env)  # Adjusted to new function

        stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
        stacked_y_relative = collections.deque(STACK_SIZE * [y_relative], maxlen=STACK_SIZE)  # Track y_relative
  
        for t in count():
            # Prepare the inputs for the network
            stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
            y_relative_t = torch.cat(tuple(stacked_y_relative), dim=1)
            # Select and perform an action
            action = select_action(stacked_states_t, y_relative_t, i_episode)  
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state and relative position
            next_state, next_y_relative = get_screen(env)

            if not done:
                next_stacked_states = stacked_states.copy()
                next_stacked_states.append(next_state)
                next_stacked_states_t =  torch.cat(tuple(next_stacked_states), dim=1)

                next_stacked_y_relative = stacked_y_relative.copy()
                next_stacked_y_relative.append(next_y_relative)
                next_stacked_y_relative_t =  torch.cat(tuple(next_stacked_y_relative), dim=1)
            else:
                next_stacked_states = None
                next_stacked_y_relative = None

            # Store the transition in memory (assuming memory can handle the new structure)
            memory.push((stacked_states_t, y_relative_t), action, 
                        (next_stacked_states_t, next_stacked_y_relative_t), reward)

            # Move to the next state
            stacked_states = next_stacked_states
            stacked_y_relative = next_stacked_y_relative

            # Perform one step of the optimization (on the target network)
            optimize_model()

            if done:
                reward = reward.cpu().numpy().item()
                ten_rewards += reward
                total_rewards.append(reward)
                mean_reward = np.mean(total_rewards[-100:])
                writer.add_scalar("epsilon", eps_threshold, i_episode)
                if (best_mean_reward is None or best_mean_reward < mean_reward) and i_episode > 100:
                    # For saving the model and possibly resuming training
                    torch.save({
                            'policy_net_state_dict': policy_net.state_dict(),
                            'target_net_state_dict': target_net.state_dict(),
                            'optimizer_policy_net_state_dict': optimizer.state_dict()
                            }, PATH)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                        ct = datetime.datetime.now()
                        new_reward_ts = ct.timestamp()
                        diff = new_reward_ts - old_reward_ts
                        if (options.detail_level != 'e'):
                            old_rewards_ts = log("Time between reward step: #\t" + str(diff))
                            s = 'Average Score: {:.3f}'.format(mean_reward)
                            elapsed = timeit.default_timer() - start_time
                            t = "Elapsed time: {}".format(timedelta(seconds=elapsed))
                            log("" + s + " " + t)

                    best_mean_reward = mean_reward
                
                break
    
        if i_episode % 10 == 0:
                writer.add_scalar('ten episodes average rewards', ten_rewards/10.0, i_episode)
                ten_rewards = 0

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Mean Reward at episode {i_episode}: {mean_reward}")

        # Declare termination condition of the training
        if i_episode >= 10000 and mean_reward > 0.999:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode+1, mean_reward))
            break

  
    print('Average Score: {:.3f}'.format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    env.close()


'''
Evaluation
'''
episode = 10
scores_window = collections.deque(maxlen=100)  # Last 100 scores

# Assuming env is properly initialized and policy_net is defined
env.cid = pb.connect(pb.DIRECT)

# Load the model
checkpoint = torch.load(PATH)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

# Evaluate the model
for i_episode in range(episode):
    env.reset()
    state, y_relative = get_screen(env)  # Adjusted to new function
    stacked_states = collections.deque(STACK_SIZE*[state], maxlen=STACK_SIZE)
    stacked_y_relatives = collections.deque(STACK_SIZE*[y_relative], maxlen=STACK_SIZE)  # Track y_relative
    
    for t in count():
        stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
        stacked_y_relatives_t = torch.cat(tuple(stacked_y_relatives), dim=1)  # Use the mean y_relative
        
        # Select and perform an action
        # Now using the policy network with both state and y_relative
        action = policy_net(stacked_states_t, stacked_y_relatives_t).max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action.item())
        
        # Observe new state and y_relative
        next_state, next_y_relative = get_screen(env)
        stacked_states.append(next_state)
        stacked_y_relatives.append(next_y_relative)  # Update stacked y_relatives
        
        if done:
            break
            
    print(f"Episode: {i_episode+1}, Reward: {reward}")
