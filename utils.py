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
import matplotlib.pyplot as plt
from IPython.display import clear_output

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
    global stacked_screens
    screen, y_relative = env._get_observation()

    if RENDER and "ARAS" in modelPath:
        show_image(screen, window_name="Latent Image", scale_factor=4)

    screen = torch.from_numpy(screen.copy())
    screen = screen.unsqueeze(0).unsqueeze(0)
    y_relative = torch.tensor([y_relative], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

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
    np.random.seed(42)  # For reproducibility
    unique_labels = np.unique(image)
    color_map = np.random.randint(0, 255, (max(unique_labels) + 1, 3), dtype=np.uint8)

    # Apply the color map to the segmentation image
    color_segmentation_image = color_map[image]

    # Calculate new dimensions
    height, width = color_segmentation_image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))


    # Resize the image
    resized_image = cv2.resize(color_segmentation_image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow(window_name ,resized_image)
    cv2.waitKey(1)
            

def modify_segmentation(segmentation, intention_object_id, intention_container_id, gripper_state):
    """
    Converts a segmentation image to a colored RGB image based on provided segmentation ID to color mappings.

    Parameters:
    - segmentation (numpy.ndarray or tuple): The segmentation image from pybullet.
    - intention_object_id (int): ID of the object of interest
    - intention_container_id (int): ID of the container 
    - gripper_state (str): Current state of the gripper

    Returns:
    - numpy.ndarray: A modified segmentation image.
    """
    # Convert tuple to numpy array if necessary
    if isinstance(segmentation, tuple):
        # Assuming it's a tuple containing the segmentation array
        # Based on PyBullet documentation, segmentation is returned as part of getCameraImage
        # Convert to numpy array - adjust indexing if needed
        segmentation = np.array(segmentation)
    
    # Check if segmentation is 1D and needs reshaping
    if len(segmentation.shape) == 1:
        # Calculate the dimensions - we assume it's a square image if not provided
        total_pixels = segmentation.shape[0]
        side_length = int(np.sqrt(total_pixels))
        segmentation = segmentation.reshape(side_length, side_length)
        # print(f"Reshaped segmentation from 1D to 2D with shape: {segmentation.shape}")
    
    # bin = numobj + 3
    # BGR color format
    segmentation_ids = {
    0: 0,   # Background white
    1: 1,   # Table black
    2: 2,   # Robot
    intention_object_id: 3,   # Mug
    intention_container_id: 4,                                                                                                                                                                                                                                                                                                                                       
    # Add more mappings as needed
    }

    # Initialize an empty RGB image with the same dimensions as the segmentation image
    height, width = segmentation.shape
    modified_seg = np.ones((height, width), dtype=np.uint8)

    # Populate the modified segs 
    for obj, id in segmentation_ids.items():
        if obj == 0 or obj == 1 or obj == 2 or obj == intention_object_id: # Background, table, robot, mug
            modified_seg[segmentation == obj] = id
        
        elif obj == intention_container_id and gripper_state == 'close':
            modified_seg[segmentation == obj] = id

    # Set the default color for any other IDs
    modified_seg[np.isin(segmentation, list(segmentation_ids.keys()), invert=True)] = 1
    return modified_seg


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
    
    
    def _is_position_valid(self, new_pos, existing_positions, min_distance=0.1):
        """Check if the new position is at least min_distance away from all existing positions."""
        for pos in existing_positions:
            if abs(new_pos[1] - pos[1]) < min_distance:
                return False
        return True

    def _randomly_place_objects(self, obj_urdfList, container_urdfList):
        """Randomly place the objects on the table ensuring minimum distance between them."""
        objectUids = []
        existing_positions = []

        attempt_limit = 100  # Set a reasonable attempt limit

        for i, urdf_path in enumerate(obj_urdfList):

            if i == 0:
                xpos = 0.12
                ypos = random.uniform(-0.02, 0.05)
                zpos = 0
                angle = -np.pi / 2
                orn = pb.getQuaternionFromEuler([0, 0, angle])
                uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False, globalScaling=.90)
                objectUids.append(uid)
                existing_positions.append((xpos, ypos))

            else:

                valid_position_found = False
                attempts = 0

                while not valid_position_found and attempts < attempt_limit:
            
                    # xpos = random.uniform(0.12, 0.14)
                    xpos = 0.12

                    if self._AutoXDistance:
                        # width = 0.05 + (xpos - 0.16) / 0.7
                        # ypos = random.uniform(-width, width)
                        ypos = random.choice([-0.12, 0.12])
                        # ypos = 0
                    else:
                        ypos = random.choice([-0.18, 0.18])
                        # ypos = random.choice([-0.17, 0, 0.17])
                        # ypos = random.uniform(-0.20, 0.20)
                        
                        ######## Test #########
                        # ypos = -0.12



                    if self._is_position_valid((xpos, ypos), existing_positions):
                        valid_position_found = True
                        zpos = 0
                        angle = -np.pi / 2 
                        orn = pb.getQuaternionFromEuler([0, 0, angle])
                        uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False, globalScaling=.90)
                        objectUids.append(uid)
                        existing_positions.append((xpos, ypos))

                    else:
                        attempts += 1  # Find a new position
            

        existing_positions = []
        container_uid = []

        ypos_containers = [-0.15, 0.05, 0.24]
        for urdf_path, ypos in zip(container_urdfList, ypos_containers):
    
                xpos = 0.30

                # Placing the trays
                orn = pb.getQuaternionFromEuler([0, 0, 0])
                uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False, globalScaling=.27)

                container_uid.append(uid)
                existing_positions.append((xpos, ypos))
        
        for _ in range(20):
            pb.stepSimulation()
        
        return objectUids, container_uid
    







   
