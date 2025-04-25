import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class DQN(nn.Module):
    def __init__(self, h, w, n_actions, stack_size):
        super(DQN, self).__init__()
        self.linear_input_size = self.calculate_output_size(h, w)
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(stack_size, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size + 16, 256)  # Adjust input size to include relative position embeddings
        self.fc2 = nn.Linear(256, n_actions)

        # Embedding layer for relative position features
        self.relative_embedding = nn.Embedding(3, 16)  # Embedding for relative positions (-1, 0, 1)

    def forward(self, x, relative_position):
        batch_size, stack_size, height, width = x.shape

        # Normalize the input segmentation IDs
        x = x.float() / 4.0

        # Apply convolutions and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the convolutional output
        x = x.view(batch_size, -1)

        # Extract the relative position of the last frame
        relative_position_last = (relative_position[:, -1] + 1).long() # Shape: [batch_size]

        # Process relative positions of the last frame through the embedding layer
        relative_position_features = self.relative_embedding(relative_position_last)  # Shape: [batch_size, 16]
        relative_position_features = relative_position_features.squeeze(dim=1)

        # Concatenate the relative position embedding with convolutional features
        x = torch.cat((x, relative_position_features), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        # Construct and apply action masking
        mask = self.construct_action_mask(batch_size, q_values.device)
        q_values += mask  # Add mask directly to Q-values

        return q_values

    def calculate_output_size(self, h, w):
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        # Width
        size_w = conv2d_size_out(w, kernel_size=3, stride=1, padding=1)  # Conv1
        size_w = conv2d_size_out(size_w, kernel_size=3, stride=2, padding=1)  # Conv2
        size_w = conv2d_size_out(size_w, kernel_size=3, stride=2, padding=1)  # Conv3

        # Height
        size_h = conv2d_size_out(h, kernel_size=3, stride=1, padding=1)  # Conv1
        size_h = conv2d_size_out(size_h, kernel_size=3, stride=2, padding=1)  # Conv2
        size_h = conv2d_size_out(size_h, kernel_size=3, stride=2, padding=1)  # Conv3

        # Output size for the fully connected layer
        linear_input_size = 64 * size_w * size_h  # 64 channels from conv3 output
        return linear_input_size

    def construct_action_mask(self, batch_size, device):
        n_actions = 5  # Forward, backward, right, left, hold
        mask = torch.zeros(batch_size, n_actions, device=device)

        # Example rule: Prevent actions 2 and 4
        mask[:, 2] = -1e9  
        mask[:, 4] = -1e9

        return mask





