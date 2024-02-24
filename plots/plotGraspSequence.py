'''

Code for creating and saving the figure sequence in Appendix A 

'''

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# from jaco_env_phase2 import jacoDiverseObjectEnv
from jaco_env_phase2 import jacoDiverseObjectEnv


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


STACK_SIZE = 4


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=7, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w,7,4),5,2),3,2),3,1),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h,7,4),5,2),3,2),3,1),3,1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)


preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(128, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(environment):
    global stacked_screens
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    rgb, depth, segmentation = environment._get_observation()
    screen = rgb.transpose((2, 0, 1))
    # screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return preprocess(screen).unsqueeze(0).to(device)

def get_screen_rgb(environment):
    global stacked_screens
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    rgb, depth, segmentation = environment._get_observation()
    return rgb

def get_screen_highres(environment):
    global stacked_screens
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    image = environment._get_observation_highres()
    return image

""" Evaluation of trained DQN model"""

PATH = 'models/learnDQN_128_simplified_bs_64_4_bz_25000.pt'
episode = 300
scores_window = collections.deque(maxlen=100)  # last 100 scores
# isTest=True -> perform grasping on test set of objects
# env = jacoDiverseObjectEnv(renders=True, isDiscrete=True, removeAutoXDistance=False, maxSteps=20, isTest=True)
env = jacoDiverseObjectEnv(actionRepeat=80, renders=True, isDiscrete=True, maxSteps=30, dv=0.02,
                           removeAutoXDistance=False, width=64, height=64, isTest=True)
env.reset()

init_screen = get_screen(environment=env)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n  # Get number of actions from gym action space
policy_net = DQN(screen_height, screen_width, n_actions).to(device)

# load trained model for the policy network
checkpoint = torch.load(PATH, map_location=device)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

s=0
f=0
reward = 0


action_name = ["forward", "forward", "forward", "right", "left" ,"down", "up", "grasp"]
# evaluate the trained model by selecting actions according to the policy net
for i_episode in range(episode):

    env.reset()
    state = get_screen(environment=env)
    stacked_states = collections.deque(STACK_SIZE*[state], maxlen=STACK_SIZE)
    for t in count():
        
        stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
        # Select and perform an action
        actions = policy_net(stacked_states_t)
        best_action = actions.max(1)[1].view(1, 1)
        actions_numpy = actions.detach().numpy()
        best_action_name = action_name[best_action.item()]

        plt.figure()
        plt.imshow(Image.fromarray(get_screen_highres(environment=env)))
        plt.title("Step " + str(t+1) + ": side view",size=14)
        plt.savefig("images/step_"+str(t+1)+"_sideview.pdf")
        plt.show()

        plt.figure()
        plt.imshow(get_screen_rgb(environment=env))
        plt.title("Step " + str(t+1) + " with action: " + best_action_name + " and Q(s," + best_action_name +") = " + str(round(actions_numpy[0][best_action.item()], 3)),size=14)
        plt.savefig("images/step_"+str(t+1)+".pdf")
        plt.show()


        # here the arm moves
        _, reward, done, _ = env.step(best_action.item())

        if best_action.item() == 7:
            plt.figure()
            plt.imshow(get_screen_rgb(environment=env))
            plt.title("Step " + str(t+1) + " with grasp",size=14)
            plt.savefig("images/step_"+str(t+1)+"_grasp.pdf")
            plt.show()
        
            plt.figure()
            plt.imshow(Image.fromarray(get_screen_highres(environment=env)))
            plt.title("Step " + str(t+1) + " sideview of grasp",size=14)
            plt.savefig("images/step_"+str(t+1)+"_grasp_sideview.pdf")
            plt.show()

        # Observe new state
        next_state = get_screen(environment=env)
        stacked_states.append(next_state)
        if done:
            break
    if reward==1:
        s=s+1
        break
    if reward==0:
        f=f+1
    print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")
    print("Successed: " + str(s) + "\tFailures: " + str(f) + "\t\tSuccessRate: " + str(s/(i_episode + 1)))
print("Successed: " + str(s) + "\tFailures: " + str(f) + "\t\tSuccessRate: " + str(s/(i_episode + 1)))

