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
from utils import convert_segmentation_to_color, show_image

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
                AutoXDistance=True, #changed
                AutoGrasp=True,
                objectRandom=0.3,
                cameraRandom=0,
                width=48,
                height=48,
                numObjects=1,
                numContainers=1,
                isTest=False):
        
        """Initializes the jacoDiverseObjectEnv.
        Args:
        urdfRoot: The diretory from which to load environment URDF's.
        actionRepeat: The number of simulation steps to apply for each action.
        isEnableSelfCollision: If true, enable self-collision.
        renders: If true, render the bullet GUI.
        isDiscrete: If true, the action space is discrete. If False, the
            action space is continuous.
        maxSteps: The maximum number of actions per episode.
        dv: The velocity along each dimension for each action.
        AutoXDistance: If True, there is a "distance hack" where the gripper
            automatically moves in the x-direction for each action, except the grasp action. 
            If false, the environment is harder and the policy chooses the distance displacement.
        AutoGrasp: If True, agent will do the grasp action automatically when it reaches to the object
        objectRandom: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
        cameraRandom: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
        width: The image width.
        height: The observation image height.
        numObjects: The number of objects in the bin.
        isTest: If true, use the test set of objects. If false, use the train
            set of objects.
        """

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
        self._isTest = isTest
        self.object_placer = ObjectPlacer(urdfRoot, AutoXDistance, objectRandom)
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
        
        ########## Testing ##########
        # actions = [Action.FORWARD]

        self.action_space = spaces.Discrete(len(actions))
        self.action_map = {i: action for i, action in enumerate(actions)}

    def _getGripper(self):
            gripper = np.array(pb.getLinkState(self._jaco.jacoUid, linkIndex=self._jaco.jacoEndEffectorIndex)[0])
            # gripper[0] -= 0.1 # offset along x axis
            # gripper[2] += 0.2 # offset along z axis
            return gripper
    

    def _getBaseLink(self):
        pos, ori = pb.getBasePositionAndOrientation(self._jaco.jacoUid)
        com_p = (pos[0]+0.35, pos[1], pos[2]+0.3)
        return com_p


    def reset(self):
        # print("++++")
        """Environment reset called at the beginning of an episode."""
        # Set the camera settings.
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
        self._env_step = 0
        self.terminated = 0

        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setTimeStep(self._timeStep)

        # Load plane and table in the environment
        pb.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), 0, 0, -0.66)
        pb.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5, 0, -0.66, 0, 0, 0, 1)

        # Set gravity 
        pb.setGravity(0, 0, -9.81)

        # Load a block for placing the bins in the environment
        orn = pb.getQuaternionFromEuler([0, 0, math.radians(90)])
        block = pb.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), [0.53, 0.02, 0.1], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=True, globalScaling=10)
        pb.changeVisualShape(block, -1, rgbaColor=[0.8, 0.8, 0.8, 1])

        # Load jaco robotic arm into the environment
        self._jaco = jaco.jaco(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, renders=self._renders)
        self._envStepCounter = 0
        pb.stepSimulation()

        # pos, ori = pb.getBasePositionAndOrientation(self._jaco.jacoUid)
        # print(pos)
        # Load random graspable objects in the environment, currently just the mug
        obj_urdfList = self.object_placer._get_random_object(self._numObjects, self._isTest)

        # Load containers in the environment
        container_urdfList = [os.path.join(self._urdfRoot, 'tray/tray.urdf')] * self._numContainers
        
        # Place loaded object randomly in the environment
        self._objectUids, self.container_uid = self.object_placer._randomly_place_objects(obj_urdfList, container_urdfList)

        # Human intention ( For now this is static but later can be dynamic )
        self.intention_object = self._objectUids[0]
        self.intention_container = self.container_uid[0]
        
        # Get mug position in xyz
        self._mugPos = np.array(pb.getBasePositionAndOrientation(self.intention_object)[0]) # 3 is representing the mug
        pb.changeVisualShape(self.intention_object, -1, rgbaColor=[0, 1, 0, 1])

        # Adjust to "true" center of mug. Without self._mugPos[2] (z-direction) is the bottom of the cup
        self._mugPos[2] = self._mugPos[2] + 0.03
        
        # Get container position in xyz
        self._containerPos = np.array(pb.getBasePositionAndOrientation(self.intention_container)[0])
        pb.changeVisualShape(self.intention_container, -1, rgbaColor=[0, 1, 0, 1])

        # Get current gripper and mug position for current euclidean distance
        self.endEffectorPos_original = self._getGripper()
        
        # GetEuclideanDistances
        self._gripper2mug_orignial = np.linalg.norm(self._mugPos[:2] - self.endEffectorPos_original[:2])  
        self._gripper2bin_orignial = np.linalg.norm(self._containerPos[:2] - self.endEffectorPos_original[:2])

        # Get camera images & Human input
        self._observation = self._get_observation()
        # return np.array(self._observation[1])



    def _get_observation(self):
        """Captures the current environment state as an observation, including the relative y-axis position of the mug to the gripper."""
        
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
        
        # Image processing
        rgb = np.array(images[2], dtype=np.uint8).reshape(self._height, self._width, 4)[:, :, :3]
        depth_buffer = np.array(images[3], dtype=np.float32).reshape(self._height, self._width)
        depth = 10 * 0.01 / (10 - (10 - 0.01) * depth_buffer)
        depth = np.stack([depth, depth, depth], axis=0).reshape(self._height, self._width, 3)
        segmentation = images[4]
        segmentation = convert_segmentation_to_color(segmentation, self._numObjects)
        show_image(segmentation)

        # Human input
        gripper_pos = self._getGripper()
        if self._gripperState == 'open':
            relative_position = self._mugPos[1] - gripper_pos[1]
            
        elif self._gripperState == 'close':
            relative_position = self._containerPos[1] - gripper_pos[1]
        
        relative_position = 0 if (abs(relative_position) < 0.025) else np.sign(relative_position)

        # Constructing the observation
        observation = [rgb, depth, segmentation, relative_position]
        return observation

    def step(self, action):
        dv = self._dv  # Velocity per physics step.
        dx, dy, dz, close_gripper = 0, 0, 0, 0

        if self._AutoXDistance:
            dx = dv
        
        if self._isDiscrete:
            action_enum = self.action_map[action]
            
            if action_enum == Action.LEFT:
                dy = -dv
            elif action_enum == Action.RIGHT:
                dy = dv
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
        """Applies a continuous velocity-control action.

        Args:
        action: 4-vector parameterizing XYZ offset and grasp action
        Returns:
        observation: Next observation.
        reward: Float of the per-step reward as a result of taking the action.
        done: Bool of whether or not the episode has ended.
        debug: Dictionary of extra information provided by environment.
        """

        # If grasp action is true, no other movement in any direction
        if action[3]:
            action[0] = action[1] = action[2] = 0

        if self._AutoGrasp:
            action[3] = abs(self._mugPos[0] - self._getGripper()[0]) < 0.02

        for _ in range(self._actionRepeat):
            pb.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                break

        # Perform commanded action.
        self._env_step += 1
        self._jaco.apply_move(action)

        # If grasp action is true (action[3]==1), attempt grasp
        if action[3] and self._gripperState == 'open':
            self._jaco.apply_grasp()
            self._gripperState = 'close'

        if (self._containerPos[0] - self._getGripper()[0] < 0.02) and self._gripperState == 'close':
            self._jaco.apply_release()
            self._gripperState = 'open'
            self._attempt = True
            

        # Get new observation
        observation = self._get_observation()

        # If done is true, the episode ends
        done = self._termination()

        # Return reward
        reward = self._reward()

        debug = {'task_success': self._taskSuccess}
        return observation, reward, done, debug  

    
    def _reward(self):
        """Calculates the reward for the episode with modified strategy for grasping and placing."""
        self._taskSuccess = 0
        failure_penalty = -0.5  # Penalty for unsuccessful grasp attempts
        max_dist_rew = 0.2  # Maximum reward for distance improvement
        placement_threshold = 0.05 # Threshold for successful placement in the bin
        placement_success_reward = 1  # Big reward for successful placement in the bin
        time_penalty = -0.01          # Small penalty for each timestep to encourage efficiency
        
        # Get the information of the environment
        bin_position = self._containerPos  # You'll need to define the target bin's position
        cur_mugPos, _ = pb.getBasePositionAndOrientation(self.intention_object)
        gripperPos = self._getGripper()
        gripper2mug = np.linalg.norm(np.array(cur_mugPos)[:2] - np.array(gripperPos)[:2])
        mug2bin = np.linalg.norm(np.array(cur_mugPos)[:2] - np.array(bin_position)[:2])
        gripper2bin = np.linalg.norm(np.array(gripperPos)[:2] - np.array(bin_position)[:2])

        # Test Episode
        if self._isTest:
            reward = 1 if mug2bin < placement_threshold else 0
            return reward

        # Training Episode
        if self._gripperState == 'open':
            dist_rew = (abs((self._gripper2mug_orignial - gripper2mug) / self._gripper2mug_orignial)) * max_dist_rew    

        elif self._gripperState == 'close':
            dist_rew = (abs((self._gripper2bin_orignial - gripper2bin) / self._gripper2bin_orignial)) * max_dist_rew

        # If the mug is placed in the bin, give a big reward
        if self._attempt and mug2bin < placement_threshold:
            self._taskSuccess += 1
            reward = placement_success_reward
        else:
            # Apply failure penalty if the release action is taken but not successful
            if self._env_step == self._maxSteps:  # Assuming there's a way to check if grasp was the last action
                reward = failure_penalty
            else:
                reward = dist_rew + time_penalty
        # print("Reward: ", reward)   
        return reward



    
    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._attempt or self._env_step >= self._maxSteps


    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _reset = reset
        _step = step


