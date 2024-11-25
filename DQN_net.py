import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelAttention(nn.Module):
#     """Channel Attention Module"""
#     def __init__(self, num_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class SpatialAttention(nn.Module):
#     """Spatial Attention Module"""
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = kernel_size // 2
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x * self.sigmoid(x)


class DQN(nn.Module):
    def __init__(self, h, w, n_actions, stack_size):
        super(DQN, self).__init__()
        self.linear_input_size = self.calculate_output_size(h, w)
        self.embedding_dim = 4  # Dimension of the embedding for each unique ID
        self.embed = nn.Embedding(11, self.embedding_dim)  # Assuming IDs from 0 to 10, inclusive
        self.relative_pos_embed = nn.Embedding(3, self.embedding_dim)  # For -1, 0, 1

        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(self.embedding_dim, 32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)


        # Fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size + self.embedding_dim, 512)  # Adjust for concatenated relative position embedding
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x, relative_position):

        batch_size, stack_size, height, width = x.shape
        # print(height, width)
        x = x.long()  # Convert to long type for embedding
        relative_position = relative_position + 1  # Adjusting indexes for embedding (-1,0,1) to (0,1,2)

        # Embed the frames
        # print(x.shape)
        x = self.embed(x.view(batch_size * stack_size, height, width)).permute(0, 3, 1, 2)
        # print(x.shape)

        # Apply convolutions and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        # Flatten and take mean across frames
        x = x.reshape(batch_size, stack_size, -1)
        # print(x.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)

        # Embed the relative position and concatenate with the conv output
        relative_position = relative_position.long().view(batch_size * stack_size, -1)
        # print(x.shape)
        relative_pos_embedding = self.relative_pos_embed(relative_position).permute(0, 2, 1)
        # print(relative_pos_embedding.shape)
        relative_pos_embedding = relative_pos_embedding.view(batch_size, stack_size, -1)
        # print(relative_pos_embedding.shape)
        relative_pos_embedding = torch.mean(relative_pos_embedding, dim=1)
        # print(relative_pos_embedding.shape)

        # Concatenate along the feature dimension
        x = torch.cat((x, relative_pos_embedding), dim=1)  

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print(x.shape)
        return x
    

    def calculate_output_size(self, h, w):

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        def pool2d_size_out(size, pool=2):
            return size // pool
        
        # Width
        size_w = conv2d_size_out(w, kernel_size=5, stride=2, padding=1) 
        size_w = pool2d_size_out(size_w, pool=2)

        size_w = conv2d_size_out(size_w, kernel_size=3, stride=2, padding=1)
        size_w = pool2d_size_out(size_w, pool=2)

        size_w = conv2d_size_out(size_w, kernel_size=3, stride=1, padding=1)
        size_w = pool2d_size_out(size_w, pool=2)

        size_w = conv2d_size_out(size_w, kernel_size=3, stride=1, padding=1)
        size_w = pool2d_size_out(size_w, pool=2)

        # height
        size_h = conv2d_size_out(h, kernel_size=5, stride=2, padding=1)
        size_h = pool2d_size_out(size_h, pool=2)

        size_h = conv2d_size_out(size_h, kernel_size=3, stride=2, padding=1)
        size_h = pool2d_size_out(size_h, pool=2)

        size_h = conv2d_size_out(size_h, kernel_size=3, stride=1, padding=1)
        size_h = pool2d_size_out(size_h, pool=2)

        size_h = conv2d_size_out(size_h, kernel_size=3, stride=1, padding=1)
        size_h = pool2d_size_out(size_h, pool=2)

        linear_input_size = 128 * size_w * size_h
        return linear_input_size
    

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

#         # Embedding layer for the relative position
#         self.embedding_dim = 3  # Dimension of the embedding space
#         self.relative_position_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim)  # -1, 0, 1

#         def conv2d_size_out(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1

#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 7, 4), 5, 2), 3, 2), 3, 1), 3, 1)
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 7, 4), 5, 2), 3, 2), 3, 1), 3, 1)
#         linear_input_size = convw * convh * 64 + self.embedding_dim * self.stack_size  # Adjusted for the embedding

#         self.linear = nn.Linear(linear_input_size, 512)
#         self.head = nn.Linear(512, outputs)

#     def forward(self, x, relative_position):

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)

#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = F.relu(x)

#         x = self.conv5(x)
#         x = F.relu(x)
#         x = x.view(x.size(0), -1)

#         # Embed the relative position and concatenate
#         relative_position_embedded = self.relative_position_embedding((relative_position + 1).long())  # Adjust index for embedding
#         # Flatten the embedded output from [1, 4, 3] to [1, 12]
#         relative_position_embedded = relative_position_embedded.view(relative_position_embedded.size(0), -1)
#         x = torch.cat((x, relative_position_embedded), dim=1)
        
#         x = F.relu(self.linear(x))
#         return self.head(x)