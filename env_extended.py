import random
import os
from gym import spaces
import time
import math
import pybullet as pb
import jaco_extended as jaco
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from enum import Enum, auto
from utils import ObjectPlacer

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
                maxSteps=8,
                dv=0.06,
                AutoXDistance=True, #changed
                AutoGrasp=True,
                objectRandom=0.3,
                cameraRandom=0,
                width=48,
                height=48,
                numObjects=1,
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
        self._isTest = isTest
        self.object_placer = ObjectPlacer(urdfRoot, AutoXDistance, objectRandom)
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
            gripper[0] -= 0.1 # offset along x axis
            gripper[2] += 0.2 # offset along z axis
            return gripper

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

        self._attempted_grasp = False
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

        # Load jaco robotic arm into the environment
        self._jaco = jaco.jaco(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        pb.stepSimulation()

        # Load random object in the environment, currently just the mug
        urdfList = self.object_placer._get_random_object(self._numObjects, self._isTest)
        
        # Place loaded object randomly in the environment
        self._objectUids = self.object_placer._randomly_place_objects(urdfList)
        
        # Get camera images (rgb,depth,segmentation)
        self._observation = self._get_observation()

        # The following is only important for the reward function of this phase
        
        # Get mug position in xyz
        self._mugPos = np.array(pb.getBasePositionAndOrientation(3)[0]) # 3 is representing the mug
        pb.changeVisualShape(3, -1, rgbaColor=[0, 1, 0, 1])

        # Adjust to "true" center of mug. Without self._mugPos[2] (z-direction) is the bottom of the cup
        self._mugPos[2] = self._mugPos[2] + 0.03
        
        # Get current gripper and mug position for current euclidean distance
        self.endEffectorPos_original = self._getGripper()
        
        # GetEuclideanDistance between mug and gripper
        self._dist_orignial = np.linalg.norm(self._mugPos - self.endEffectorPos_original)  
     
        # return np.array(self._observation[1])


    # def _randomly_place_objects(self, urdfList):
    #     """Randomly place the objects on the table.

    #     Args:
    #     urdfList: The list of urdf files to place on the table.

    #     Returns:
    #     The list of object unique ID's.
    #     """

    #     # Randomize positions of each object urdf.
    #     objectUids = []
    #     for urdf_name in urdfList:
    #         xpos = random.uniform(0.16, 0.23) 
                        
    #         # If distance hack is false, init mug in a square area on the table
    #         # If distance hack is true , init mug in a trapezoid area on the table to be feseable grasp
    #         if self._AutoXDistance:
    #             width = 0.05 + (xpos - 0.16) / 0.7
    #             ypos = random.uniform(-width, width) 
    #         else:
    #             ypos = random.uniform(0, 0.2)

    #         zpos = -0.02
    #         angle = -np.pi / 2 + self._objectRandom * np.pi * random.random()
    #         orn = pb.getQuaternionFromEuler([0, 0, angle])
    #         urdf_path = os.path.join(self._urdfRoot, urdf_name)
            
    #         uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]], useFixedBase=False)
            
    #         objectUids.append(uid)
            
    #         for _ in range(20):
    #             pb.stepSimulation()
        
    #     return objectUids

    def _get_observation(self):
        """Captures the current environment state as an observation, including the relative y-axis position of the mug to the gripper."""
        # gripper view
        # link_state = pb.getLinkState(self._jaco.jacoUid, linkIndex=6)
        # print(self._jaco.jacoEndEffectorIndex)
        # pos, ori = link_state[0], link_state[1]
        com_p = self._getGripper()
        # com_p = pos

        # # base view
        # pos, ori = pb.getBasePositionAndOrientation(self._jaco.jacoUid)
        # com_p = (pos[0]+0.35, pos[1], pos[2]+0.3)

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

        # Human input
        gripper_pos = self._getGripper()
        mug_pos = np.array(pb.getBasePositionAndOrientation(3)[0])
        relative_position = mug_pos[1] - gripper_pos[1]
        relative_position = 0 if (abs(relative_position) < 0.025) else np.sign(relative_position)
        # print(relative_position)

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

        else:
            self._action_taken = 'move'

        if self._AutoGrasp:
            action[3] = abs(self._mugPos[0] - self._getGripper()[0]) < 0.12

        # Perform commanded action.
        self._env_step += 1
        self._jaco.applyAction(action)
        for _ in range(self._actionRepeat):
            pb.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                break

        # If grasp action is true (action[3]==1), attempt grasp
        if action[3]:
            self._action_taken = 'grasp'
            # self._graspPosition = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)[0]
            # self._distXYZ_beforeGrasp = np.linalg.norm(posMug - self._graspPosition)
            
            finger_angle = 0.6
            tip_angle = finger_angle
            # Close fingers
            for _ in range(150):
                grasp_action = [0, 0, 0, finger_angle]
                self._jaco.applyAction(grasp_action)
                pb.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                finger_angle += 0.1/100.
                if finger_angle > 2:
                    finger_angle = 2 #Upper limit
            
            # Close fingertips
            for _ in range(100):
                pb.setJointMotorControlArray(
                        bodyUniqueId=self._jaco.jacoUid, 
                        jointIndices=self._jaco.fingertipIndices,
                        controlMode=pb.POSITION_CONTROL,
                        targetPositions=[tip_angle]*len(self._jaco.fingertipIndices),   
                        targetVelocities=[0]*len(self._jaco.fingertipIndices),
                        forces=[self._jaco.fingerThumbtipforce,self._jaco.fingertipforce,self._jaco.fingertipforce],
                        velocityGains=[1]*len(self._jaco.fingertipIndices)
                )
                pb.stepSimulation()
                tip_angle += 0.1 / 100.
                if finger_angle > 2:
                    finger_angle = 2 # Upper limit
            # Lift gripper after grasp    
            for _ in range(50):
                grasp_action = [0, 0, 0.01, finger_angle]
                self._jaco.applyAction(grasp_action)
                pb.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
            self._attempted_grasp = True
 

        # Get new observation
        observation = self._get_observation()

        # If done is true, the episode ends
        done = self._termination()

        # Return reward
        reward = self._reward()

        debug = {'grasp_success': self._graspSuccess}
        return observation, reward, done, debug  

    
    # def _reward(self):
    #     """Calculates the reward for the episode.
    #     range mugposechange: 0 to 0.1
    #     range dist: 0 to 0.65
    #     """

    #     # reward = 0
    #     self._graspSuccess = 0
    #     effective_dist = 0.12
    #     max_penalty = -0.1
    #     max_dist_rew = 0.2
    #     max_dist_range= 0.3
        
    #     for uid in self._objectUids:
    #         cur_mugPos, _ = pb.getBasePositionAndOrientation(uid)
    #         gripperPos = self._getGripper()
    #         dist = np.linalg.norm(np.array(cur_mugPos) - np.array(gripperPos))
    
    #         # For testing just return reward if mug is lifed successfully, just important for the evaluation
    #         if self._isTest:
    #             reward = 1 if cur_mugPos[2] > 0.05 else 0

    #         else:
    #             mugPose_change = np.linalg.norm(np.array(cur_mugPos[:2]) - np.array(self._mugPos)[:2]) # along x & y axis
    #             dist_rew = (abs((self._dist_orignial - dist) / self._dist_orignial) / max_dist_range) * max_dist_rew
    #             # print(dist_rew)

    #             pose_change_penalty = (mugPose_change / 0.1) * max_penalty if (mugPose_change > 1e-4) else 0
    #             dist_rew = dist_rew if (dist_rew > effective_dist) else 0.0
        
                    
    #             # If object is above height, provide reward.
    #             if cur_mugPos[2] > 0.05:
    #                 self._graspSuccess += 1
    #                 reward = 1
    #             else:
    #                 reward = dist_rew + pose_change_penalty
    #             break

    #     # print("Rew", reward)
    #     return reward
    
    def _reward(self):
        """Calculates the reward for the episode with modified strategy."""
        self._graspSuccess = 0
        grasp_failure_penalty = -0.5  # Penalty for unsuccessful grasp attempts
        time_penalty = -0.01          # Small penalty for each timestep to encourage efficiency
        effective_dist = 0.12
        max_penalty = -0.2            # Increased penalty for moving the mug
        max_dist_rew = 0.3
        max_dist_range = 0.3
        
        for uid in self._objectUids:
            cur_mugPos, _ = pb.getBasePositionAndOrientation(uid)
            gripperPos = self._getGripper()
            dist = np.linalg.norm(np.array(cur_mugPos) - np.array(gripperPos))

            if self._isTest:
                reward = 1 if cur_mugPos[2] > 0.05 else 0
            else:
                mugPose_change = np.linalg.norm(np.array(cur_mugPos[:2]) - np.array(self._mugPos)[:2])  # along x & y axis
                dist_rew = (abs((self._dist_orignial - dist) / self._dist_orignial) / max_dist_range) * max_dist_rew

                pose_change_penalty = (mugPose_change / 0.1) * max_penalty if (mugPose_change > 1e-4) else 0
                # dist_rew = max(dist_rew, 0.01)  # Ensure there's always a small reward for moving closer
                # dist_rew = dist_rew if (dist_rew > effective_dist) else 0.0
                dist_rew = 0
                # print(dist_rew)
                
                if cur_mugPos[2] > 0.05:
                    self._graspSuccess += 1
                    reward = 1
                else:
                    # Apply grasp failure penalty if the grasp action is taken but not successful
                    if self._action_taken == 'grasp':  # Assuming there's a way to check if grasp was the last action
                        reward = grasp_failure_penalty
                    else:
                        reward = dist_rew + pose_change_penalty + time_penalty
            break
        
        # print("Rew", reward)
        return reward


    
    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._attempted_grasp or self._env_step >= self._maxSteps


    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _reset = reset
        _step = step


