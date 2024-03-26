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
import cv2

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
    screen = segmentation.transpose((2, 0, 1))   #[rgb.transpose((2, 0, 1)), depth.transpose((2, 0, 1)), segmentation] 
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = preprocess(screen).unsqueeze(0)
    # print(screen.shape)
    y_relative = torch.tensor([y_relative], dtype=torch.float32, device=device).unsqueeze(0)
    # Resize, and add a batch dimension (BCHW)
    return screen.to(device), y_relative.to(device)



def show_image(image, window_name="Image", scale_factor=4):
    """
    Displays an image in real-time using OpenCV, with an option to resize (scale) the image.

    Parameters:
    - image (numpy.ndarray): The image to display.
    - window_name (str): The name of the window where the image will be displayed.
    - scale_factor (float): Factor by which to scale the image size.
    """
    # Calculate new dimensions
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(1)


def convert_segmentation_to_color(segmentation, numobj):
    """
    Converts a segmentation image to a colored RGB image based on provided segmentation ID to color mappings.

    Parameters:
    - segmentation (numpy.ndarray): The segmentation image.
    - segmentation_colors (dict): A dictionary mapping segmentation IDs to RGB colors.
    - default_color (list): Default RGB color for unmapped segmentation IDs.

    Returns:
    - numpy.ndarray: An RGB image where each segmentation ID is mapped to a specified color.
    """
    default_color=[0, 0, 0]
    tray = numobj + 4
    # BGR color format
    segmentation_colors = {
    0: [255, 255, 255], # Background white
    1: [0, 0, 0],     # Table black
    2: [130, 130, 130],   # Block
    3: [255, 0, 0],     # Robot
    4: [0, 255, 0],     # Mug
    tray:[0, 0, 255],  
                                                                                                                                                                                                                                                                                                                                              
    # Add more mappings as needed
    }

    # Initialize an empty RGB image with the same dimensions as the segmentation image
    height, width = segmentation.shape
    segmentation_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Populate the RGB image with colors based on segmentation IDs
    for seg_id, color in segmentation_colors.items():
        segmentation_rgb[segmentation == seg_id] = color

    # Set the default color for any other IDs
    segmentation_rgb[np.isin(segmentation, list(segmentation_colors.keys()), invert=True)] = default_color
    
    return segmentation_rgb


def mask_specific_object(segmentation, specific_seg_id, segmentation_colors, default_color=[0, 0, 0]):
    """
    Creates a mask for a specific object in the segmentation image and an RGB visualization of this mask.

    Parameters:
    - segmentation (numpy.ndarray): The segmentation image.
    - specific_seg_id (int): The segmentation ID of the specific object to focus on.
    - segmentation_colors (dict): A dictionary mapping segmentation IDs to RGB colors.
    - default_color (list): RGB color used for the specific object in the returned visualization.

    Returns:
    - numpy.ndarray: A binary mask where the specific object is True, and everything else is False.
    - numpy.ndarray: An RGB image visualizing the mask of the specific object.
    """
    # Create a binary mask where the specific object is True, and everything else is False
    specific_object_mask = segmentation == specific_seg_id

    # Create a visualization of the mask
    height, width = segmentation.shape
    specific_object_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    specific_object_rgb[specific_object_mask] = segmentation_colors.get(specific_seg_id, default_color)
    
    return specific_object_mask, specific_object_rgb


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

    def _randomly_place_objects(self, obj_urdfList, container_urdfList):
        """Randomly place the objects on the table ensuring minimum distance between them."""
        objectUids = []
        existing_positions = []

        # attempt_limit = 100  # Set a reasonable attempt limit
        # attempts = 0

        for urdf_path in obj_urdfList:
            valid_position_found = False

            while not valid_position_found:
        
                # xpos = random.uniform(0.16, 0.23)
                xpos = 0.30

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
                uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False)

                objectUids.append(uid)
                existing_positions.append((xpos, ypos))
        
        existing_positions = []
        container_uid = []

        for urdf_path in container_urdfList:
            valid_position_found = False
            
            while not valid_position_found:
                
                xpos = 0.53
                ypos = random.choice([-0.2, 0, 0.2])
                zpos = 0.2
                ###########
                # ypos = 0
                
                
                if self._is_position_valid((xpos, ypos), existing_positions, min_distance=0.1):
                    valid_position_found = True
                else:
                    continue  # Find a new position

                # Placing the trays
                orn = pb.getQuaternionFromEuler([0, 0, 0])
                uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False, globalScaling=.3)

                container_uid.append(uid)
                existing_positions.append((xpos, ypos))
        
        for _ in range(20):
            pb.stepSimulation()
        
        return objectUids, container_uid
    







   
