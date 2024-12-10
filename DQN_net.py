import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, h, w, n_actions, stack_size):
        super(DQN, self).__init__()
        self.linear_input_size = self.calculate_output_size(h, w)
        self.embedding_dim = 3  # Dimension of the embedding for each unique ID
        self.embed = nn.Embedding(5, self.embedding_dim)  # Assuming IDs from 0 to 4, inclusive
        self.relative_pos_embed = nn.Embedding(3, self.embedding_dim)  # For -1, 0, 1

        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(self.embedding_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size + self.embedding_dim, 256)  # Adjust for concatenated relative position embedding
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x, relative_position):

        batch_size, stack_size, height, width = x.shape
        # print(height, width)
        x = x.long()  # Convert to long type for embedding
        relative_position = relative_position + 1  # Adjusting indexes for embedding (-1,0,1) to (0,1,2)

        # Embed the frames
        x = self.embed(x.view(batch_size * stack_size, height, width)).permute(0, 3, 1, 2)

        # Apply convolutions and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(batch_size, stack_size, -1)
        x = torch.mean(x, dim=1)

        # Embed the relative position and concatenate with the conv output
        relative_position = relative_position.long().view(batch_size * stack_size, -1)
        relative_pos_embedding = self.relative_pos_embed(relative_position).permute(0, 2, 1)
        relative_pos_embedding = relative_pos_embedding.view(batch_size, stack_size, -1)

        # relative_pos_embedding = torch.mean(relative_pos_embedding, dim=1)
        relative_pos_embedding = relative_pos_embedding[:, -1, :]

        # Concatenate along the feature dimension
        x = torch.cat((x, relative_pos_embedding), dim=1)  

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

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
