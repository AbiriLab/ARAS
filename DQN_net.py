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