import random
import os
from gym import spaces
import time
import math
import pybullet as pb
import jaco_model as jaco
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from enum import Enum, auto
from utils import ObjectPlacer
from utils import modify_segmentation
import torchvision.transforms as T
from PIL import Image
from config import *

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
largeValObservation = 100


class Action(Enum):
    HOLD = auto()
    LEFT = auto()
    RIGHT = auto()
    FORWARD = auto()
    BACKWARD = auto()
    GRASP = auto()


class jacoDiverseObjectEnv(gym.Env):
    """Class for jaco environment with diverse objects, currently just the mug.
    In each episode one object is chosen from a set of diverse objects (currently just mug).
    """

    def __init__(self,
                urdfRoot=pybullet_data.getDataPath(),
                actionRepeat=80,
                isEnableSelfCollision=True,
                renders=False,
                isDiscrete=False,
                maxSteps=30,
                dv=0.06,
                AutoXDistance=True, 
                AutoGrasp=True,
                objectRandom=0.3,
                cameraRandom=0,
                width=48,
                height=48,
                numObjects=1,
                numContainers=1,
                scenario='fixed',
                forward_limit = 0.28,
                left_limit=0.20,
                right_limit=-0.20,
                isTest=False):
        


        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dv = dv
        self._p = pb
        self._AutoXDistance = AutoXDistance
        self._AutoGrasp = AutoGrasp
        # self._objectRandom = objectRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._numContainers = numContainers
        self._scenario = scenario
        self._isTest = isTest
        self.object_placer = ObjectPlacer(urdfRoot, AutoXDistance, objectRandom)
        self._forward_limit = forward_limit
        self._left_limit = left_limit
        self._right_limit = right_limit

        # Define action space
        self.define_action_space()
        if self._renders:
            self.cid = pb.connect(pb.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = pb.connect(pb.GUI)
            pb.resetDebugVisualizerCamera(1.3, 180, -41, [0.3, -0.2, -0.33])
        else:
            self.cid = pb.connect(pb.DIRECT)

        # self.seed()

        self.viewer = None


    def define_action_space(self):
        actions = [Action.LEFT, Action.RIGHT, Action.HOLD]
        
        if not self._AutoXDistance:
            actions.extend([Action.FORWARD, Action.BACKWARD])

        if not self._AutoGrasp:
            actions.append(Action.GRASP)
        
        self.action_space = spaces.Discrete(len(actions))
        self.action_map = {i: action for i, action in enumerate(actions)}

    def _getGripper(self):
            gripper = np.array(pb.getLinkState(self._jaco.jacoUid, linkIndex=self._jaco.jacoEndEffectorIndex)[0])
            return gripper
    

    def _getBaseLink(self):
        pos, ori = pb.getBasePositionAndOrientation(self._jaco.jacoUid)
        com_p = (pos[0]+0.25, pos[1], pos[2]+0.85)
        return com_p

    def _is_active_scenario(self, target):
        return self._scenario == target or self._scenario == 'dynamic_both'

    def reset(self):
        look = [0.23, 0.2, 0.54]
        distance = 1.
        pitch = -56 + self._cameraRandom * np.random.uniform(-3, 3)
        yaw = 245 + self._cameraRandom * np.random.uniform(-3, 3)
        roll = 0
        self._view_matrix = pb.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = pb.computeProjectionMatrixFOV(fov, aspect, near, far)

        self._attempt = False
        self._gripperState = 'open'
        self._grasp_successfull = False
        self._env_step = 0
        self.terminated = 0
        self._dynamic_pickup_flag = False
        self._dynamic_dropoff_flag = False

        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setTimeStep(self._timeStep)

        plane = pb.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), 0, 0, -0.66)
        table = pb.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5, 0, -0.66, 0, 0, 0, 1)
        
        pb.setGravity(0, 0, -9.81)

        self._jaco = jaco.jaco(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, renders=self._renders)
        self._envStepCounter = 0
        pb.stepSimulation()

        obj_urdfList = self.object_placer._get_random_object(self._numObjects, self._isTest)

        container_urdfList = [os.path.join(self._urdfRoot, 'tray/tray.urdf')] * self._numContainers
        
        self._objectUids, self.container_uid = self.object_placer._randomly_place_objects(obj_urdfList, container_urdfList)
        for id in self._objectUids:
            pb.changeDynamics(id, -1, mass=0.03, lateralFriction=2, restitution=0.1, spinningFriction=0.4, contactStiffness=2000, contactDamping=8000)

        self.intention_object = random.choice(self._objectUids)
        self.intention_container = random.choice(self.container_uid)
        
        self._mugPos = np.array(pb.getBasePositionAndOrientation(self.intention_object)[0]) 
        pb.changeVisualShape(self.intention_object, -1, rgbaColor=[0, 1, 0, 1])

        self._mugPos[2] = self._mugPos[2] + 0.045
        
        self._containerPos = np.array(pb.getBasePositionAndOrientation(self.intention_container)[0])
        pb.changeVisualShape(self.intention_container, -1, rgbaColor=[0, 1, 0, 1])

        self.endEffectorPos_original = self._getGripper()
        
        self._gripper2mug_orignial = np.linalg.norm(self._mugPos[:2] - self.endEffectorPos_original[:2])  
        self._gripper2bin_orignial = np.linalg.norm(self._containerPos[:2] - self.endEffectorPos_original[:2])

        self._observation = self._get_observation()
        
        self.gripper_trajectory = []  



    def _get_observation(self):
        
        # # gripper view
        # com_p = self._getGripper()
        
        # base view
        com_p = self._getBaseLink()

        ori_euler = [3*math.pi/4, 0, math.pi/2]
        com_o = pb.getQuaternionFromEuler(ori_euler)
        rot_matrix = pb.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        camera_vector = rot_matrix.dot((0, 0, 1))  # z-axis
        up_vector = rot_matrix.dot((0, 1, 0))  # y-axis
        view_matrix = pb.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        aspect = self._width / self._height
        proj_matrix = pb.computeProjectionMatrixFOV(fov=60, aspect=aspect, nearVal=0.01, farVal=10.0)
        images = pb.getCameraImage(width=self._width, height=self._height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER)
        

        # Get rgb observation
        rgb = np.array(images[2], dtype=np.uint8).reshape(self._height, self._width, 4)[:, :, :3]
        rgb = rgb.transpose((2, 0, 1))
        rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255
        rgb = torch.from_numpy(rgb)

        preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(64, interpolation=Image.BICUBIC),
                    lambda img: np.asarray(img)
                    ])
        grayscale = preprocess(rgb)
 
        # Get segmentation observation
        segmentation = images[4]
        segmentation = np.array(segmentation)

        if len(segmentation.shape) == 1:
            segmentation = segmentation.reshape(self._height, self._width)

        segmentation = modify_segmentation(segmentation, self.intention_object, self.intention_container, self._gripperState)

        # User inputs based on scenarios
        if self._scenario == 'fixed':
            pass
     
        if self._is_active_scenario('dynamic_pickup') and self._gripperState == "open" and self._env_step > 2 and not self._dynamic_pickup_flag:
            pb.changeVisualShape(self.intention_object, -1, rgbaColor=[1, 0, 0, 1])
            new_objects = [obj for obj in self._objectUids if obj != self.intention_object]
            self.intention_object = random.choice(new_objects)
            self._mugPos = np.array(pb.getBasePositionAndOrientation(self.intention_object)[0])
            pb.changeVisualShape(self.intention_object, -1, rgbaColor=[0, 1, 0, 1])
            self._mugPos[2] += 0.045
            self._dynamic_pickup_flag = True

        if self._is_active_scenario('dynamic_dropoff') and self._gripperState == "close" and self._env_step > 20 and not self._dynamic_dropoff_flag:
            pb.changeVisualShape(self.intention_container, -1, rgbaColor=[.5, .5, .5, 1])
            new_objects = [obj for obj in self.container_uid if obj != self.intention_container]
            self.intention_container = random.choice(new_objects)
            self._containerPos = np.array(pb.getBasePositionAndOrientation(self.intention_container)[0])
            pb.changeVisualShape(self.intention_container, -1, rgbaColor=[0, 1, 0, 1])
            self._dynamic_dropoff_flag = True


        gripper_pos = self._getGripper()
        if self._gripperState == 'open':
            relative_position = self._mugPos[1] - gripper_pos[1]
            
        elif self._gripperState == 'close':
            relative_position = self._containerPos[1] - gripper_pos[1]
        
        relative_position = 0 if (abs(relative_position) < 0.025) else np.sign(relative_position)

        if "ARAS" in modelPath:
            observation = [segmentation, relative_position]
        else:
            observation = [grayscale, relative_position]

        return observation

    def step(self, action):
        dv = self._dv  
        dx, dy, dz, close_gripper = 0, 0, 0, 0

        if self._AutoXDistance:
            dx = dv
        
        if self._isDiscrete:
            action_enum = self.action_map[action]
            
            if action_enum == Action.LEFT:
                dy = dv
            elif action_enum == Action.RIGHT:
                dy = -dv
            elif action_enum == Action.FORWARD:
                dx = dv 
            elif action_enum == Action.BACKWARD:
                dx = -dv
            elif action_enum == Action.GRASP:
                close_gripper = 1
        else:
            dx = dv * action[0] if not self._AutoXDistance else dv
            dy = dv * action[1]
            dz = dv * action[2]
            close_gripper = 1 if action[3] >= 0.5 else 0
        
        return self._step_continuous([dx, dy, dz, close_gripper])


    def _step_continuous(self, action):

        if action[3]:
            action[0] = action[1] = action[2] = 0

        if self._AutoGrasp:
            action[3] = abs(self._mugPos[0] - self._getGripper()[0]) < 0.03
        
        gripper_pos = self._getGripper()
        self.gripper_trajectory.append(gripper_pos[:2]) 

        _cur, _ = pb.getBasePositionAndOrientation(self.intention_container)
        limit_action = np.sign(_cur[1] - gripper_pos[1]) * self._dv
        
        if gripper_pos[0] > self._forward_limit:
            action[0] = max(0, self._forward_limit - gripper_pos[0])
            action[1] = limit_action

        if gripper_pos[1] > self._left_limit:
            action[1] = max(0, self._left_limit - gripper_pos[1])
            action[0] = limit_action

        if gripper_pos[1] < self._right_limit:
            action[1] = min(0, self._right_limit - gripper_pos[1])
            action[0] = limit_action

        for _ in range(self._actionRepeat):
            pb.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                break
        
        if action[3] and self._gripperState == 'open':
            self._jaco.apply_grasp()
            self._gripperState = 'close'

        elif np.linalg.norm(np.array(self._containerPos)[:2] - np.array(self._getGripper())[:2]) < 0.06 and self._gripperState == 'close':
            self._jaco.apply_release()
            self._gripperState = 'open'
            self._attempt = True

        else:

            self._jaco.apply_move(action)

        self._env_step += 1


        reward = self._reward(self._observation[1], action) # self.observation[1] is the relative position
        self._observation = self._get_observation()
        done = self._termination()

        debug = {'task_success': self._taskSuccess}
        return self._observation, reward, done, debug  

    def _reward(self, command_action, taken_action):
        
        # Constants
        self._taskSuccess = 0
        failure_penalty = -0.5
        # max_dist_rew = 1
        placement_threshold = 0.1
        placement_success_reward = 1.0
        grasp_success_reward = 0.5
        # time_penalty = -0.01
        lack_of_progress_penalty = -0.01
        progress_reward_value = 0.005
        following_rew_value = 0.015
        following_penalty_value = -0.03

        following_rew = 0
        total_reward = 0
        normalized_reward = 0

        bin_position = self._containerPos
        cur_mugPos, _ = pb.getBasePositionAndOrientation(self.intention_object)
        gripperPos = self._getGripper()

        gripper2mug = np.linalg.norm(np.array(cur_mugPos)[:2] - np.array(gripperPos)[:2])
        mug2bin = np.linalg.norm(np.array(cur_mugPos)[:2] - np.array(bin_position)[:2])
        gripper2bin = np.linalg.norm(np.array(gripperPos)[:2] - np.array(bin_position)[:2])

        dx = int(np.sign(taken_action[0]))
        dy = int(np.sign(taken_action[1]))

        if dx < 0:
            following_rew -= 0.1
        # Test episode reward
        if self._isTest:
            return 1.0 if mug2bin < placement_threshold else 0.0

        if (command_action != 0 and command_action == dy) or (command_action == 0 and dx > 0):
            following_rew += following_rew_value
        else:
            following_rew += following_penalty_value

        if self._gripperState == 'open':
            
            if hasattr(self, '_prev_gripper2mug'):
                if gripper2mug < self._prev_gripper2mug:  
                    progress_reward = progress_reward_value 
                else:  
                    progress_reward = lack_of_progress_penalty
            else:
                if gripper2mug < self._gripper2mug_orignial:
                    progress_reward = progress_reward_value 
                else: 
                    progress_reward = lack_of_progress_penalty
                
    
            self._prev_gripper2mug = gripper2mug

        elif self._gripperState == 'close':
        
            if hasattr(self, '_prev_gripper2bin'):
                if gripper2bin < self._prev_gripper2bin: 
                    progress_reward = progress_reward_value
                else:  
                    progress_reward = lack_of_progress_penalty
            else:
                if gripper2bin < self._gripper2bin_orignial:
                    progress_reward = progress_reward_value 
                else: 
                    progress_reward = lack_of_progress_penalty

            self._prev_gripper2bin = gripper2bin

        progress_reward = np.clip(progress_reward, lack_of_progress_penalty, progress_reward_value)

        if self._attempt and mug2bin < placement_threshold:
            self._taskSuccess += 1
            total_reward = placement_success_reward

        elif cur_mugPos[2] > 0.05 and not self._grasp_successfull:
            total_reward = grasp_success_reward
            self._grasp_successfull = True

        # Handle failure or ongoing progress
        else:
            if self._env_step >= self._maxSteps:
                total_reward = failure_penalty
            else:
                total_reward = progress_reward + following_rew

        return [progress_reward, following_rew, total_reward]




    def _termination(self):
        return self._attempt or self._env_step >= self._maxSteps

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _reset = reset
        _step = step


