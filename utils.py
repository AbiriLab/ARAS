import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image
import datetime
from config import *
import os
import pybullet as pb
import glob

# class DQN(nn.Module):
#     def __init__(self, h, w, outputs, stack_size):
#         super(DQN, self).__init__()
#         self.stack_size = stack_size
#         self.conv1 = nn.Conv2d(self.stack_size, 32, kernel_size=7, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

#         def conv2d_size_out(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1
        
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 7, 4), 5, 2), 3, 2), 3, 1), 3, 1)
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 7, 4), 5, 2), 3, 2), 3, 1), 3, 1)
#         linear_input_size = convw * convh * 64 + 4  # +4 for the additional scalar input 4 stacked 

#         self.linear = nn.Linear(linear_input_size, 512)
#         self.head = nn.Linear(512, outputs)

#     def forward(self, x, relative_position):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.conv5(x))
#         x = x.view(x.size(0), -1)
#         # Concatenate the relative position with the output of the conv layers
#         x = torch.cat((x, relative_position), dim=1)  # Ensure relative_position is correctly sized
#         # print("=======x", x.shape)
#         x = F.relu(self.linear(x))
#         return self.head(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, h, w, outputs, stack_size):
        super(DQN, self).__init__()
        self.stack_size = stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, kernel_size=7, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Embedding layer for the relative position
        self.embedding_dim = 3  # Dimension of the embedding space
        self.relative_position_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim)  # -1, 0, 1

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 7, 4), 5, 2), 3, 2), 3, 1), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 7, 4), 5, 2), 3, 2), 3, 1), 3, 1)
        linear_input_size = convw * convh * 64 + self.embedding_dim * self.stack_size  # Adjusted for the embedding

        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x, relative_position):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)

        # Embed the relative position and concatenate
        relative_position_embedded = self.relative_position_embedding((relative_position + 1).long())  # Adjust index for embedding
        # Flatten the embedded output from [1, 4, 3] to [1, 12]
        relative_position_embedded = relative_position_embedded.view(relative_position_embedded.size(0), -1)
        x = torch.cat((x, relative_position_embedded), dim=1)
        
        x = F.relu(self.linear(x))
        return self.head(x)

    
# Define replay buffer
class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.Transition = Transition
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves the experience."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity # Dynamic replay memory -> delete oldest entry first and add newest if buffer full

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Function for recieving PyBullet camera data as input image
def get_screen(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=1),
                        T.Resize(128, interpolation=Image.BICUBIC),
                        T.ToTensor()])

    global stacked_screens
    # Transpose screen into torch order (CHW).
    rgb, depth, segmentation, y_relative = env._get_observation()
    screen = rgb.transpose((2, 0, 1))   #[rgb.transpose((2, 0, 1)), depth.transpose((2, 0, 1)), segmentation] 

    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = preprocess(screen).unsqueeze(0)
    # print(screen.shape)
    y_relative = torch.tensor([y_relative], dtype=torch.float32, device=device).unsqueeze(0)
    # Resize, and add a batch dimension (BCHW)
    return screen.to(device), y_relative.to(device)


class ObjectPlacer:
    def __init__(self, urdfRoot, AutoXDistance=True, objectRandom=0.3):
        self._urdfRoot = urdfRoot
        self._AutoXDistance = AutoXDistance
        self._objectRandom = objectRandom


    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
        num_objects:
            Number of graspable objects. For now just the mug.

        Returns:
        A list of urdf filenames.
        """
        # Select path of folder containing objects, for now just the mug
        # If more objects in the path, a random objects is selected
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, 'objects/mug.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot, 'objects/mug.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames
    
    
    def _is_position_valid(self, new_pos, existing_positions, min_distance=0.15):
        """Check if the new position is at least min_distance away from all existing positions."""
        for pos in existing_positions:
            if abs(new_pos[1] - pos[1]) < min_distance:
                return False
        return True

    def _randomly_place_objects(self, urdfList):
        """Randomly place the objects on the table ensuring minimum distance between them."""
        objectUids = []
        existing_positions = []

        # attempt_limit = 100  # Set a reasonable attempt limit
        # attempts = 0

        for urdf_name in urdfList:
            valid_position_found = False

            while not valid_position_found:
        
                # xpos = random.uniform(0.16, 0.23)
                xpos = 0.23

                if self._AutoXDistance:
                    # width = 0.05 + (xpos - 0.16) / 0.7
                    # ypos = random.uniform(-width, width)
                    ypos = random.choice([-0.17, 0, 0.17])
                else:
                    ypos = random.uniform(0, 0.2)

                if self._is_position_valid((xpos, ypos), existing_positions):
                    valid_position_found = True
                else:
                    continue  # Find a new position

                zpos = -0.02
                angle = -np.pi / 2 + self._objectRandom * np.pi * random.random()
                orn = pb.getQuaternionFromEuler([0, 0, angle])
                urdf_path = os.path.join(self._urdfRoot, urdf_name)

                uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False)

                objectUids.append(uid)
                existing_positions.append((xpos, ypos))

                for _ in range(20):
                    pb.stepSimulation()

        return objectUids




   
