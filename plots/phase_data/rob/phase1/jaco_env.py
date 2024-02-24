# from pybullet_envs.bullet.jacoGymEnv import jacoGymEnv
from jacoGymEnv import jacoGymEnv
import random
import os
from gym import spaces
import time
import math
import pybullet as pb
import jaco
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
largeValObservation = 100


class jacoDiverseObjectEnv(jacoGymEnv):
    """Class for jaco environment with mug."""

    def __init__(self,
                urdfRoot=pybullet_data.getDataPath(),
                actionRepeat=80,
                isEnableSelfCollision=True,
                renders=False,
                isDiscrete=False,
                maxSteps=8,
                dv=0.06,
                removeAutoXDistance=True, #changed
                blockRandom=0.3,
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
        removeAutoXDistance: If false, there is a "height hack" where the gripper
            automatically moves down for each action. If true, the environment is
            harder and the policy chooses the height displacement.
        blockRandom: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
        cameraRandom: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
        width: The image width.
        height: The observation image height.
        numObjects: The number of objects in the bin.
        isTest: If true, use the test set of objects. If false, use the train
            set of objects.
        """
        '''super(jacoDiverseObjectEnv, self).__init__(urdfRoot=pybullet_data.getDataPath(),
                                                actionRepeat=1,
                                                isEnableSelfCollision=True,
                                                renders=False,
                                                isDiscrete=False,
                                                maxSteps=1000)'''
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
        self._removeAutoXDistance = removeAutoXDistance
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest

        if self._renders:
            self.cid = pb.connect(pb.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = pb.connect(pb.GUI)
            pb.resetDebugVisualizerCamera(1.3, 180, -41, [0.3, -0.2, -0.33])
        else:
            self.cid = pb.connect(pb.DIRECT)
        self.seed()
        print(self.seed(1))
        '''
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        '''
        if (self._isDiscrete):
            if self._removeAutoXDistance:
                self.action_space = spaces.Discrete(7)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
            if self._removeAutoXDistance:
                self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
        self.viewer = None


    def reset(self):
        """Environment reset called at the beginning of an episode.
        """
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
        # pb.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])  # not necessary

        pb.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"),0,0,-0.66)
        pb.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -0.66,
                0.000000, 0.000000, 0.0, 1.0)

        pb.setGravity(0, 0, -10)
        self._jaco = jaco.jaco(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        pb.stepSimulation()

        urdfList = self._get_random_object(self._numObjects, self._isTest)
        self._objectUids = self._randomly_place_objects(urdfList)
        self._observation = self._get_observation()

        return np.array(self._observation[1])


    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
        urdfList: The list of urdf files to place in the bin.

        Returns:
        The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0.23 #+ self._blockRandom * random.random()
            ypos = random.uniform(-0.2,0.2) #self._blockRandom * (random.random() - 0.5) *1.5   

            angle = -np.pi / 2 + self._blockRandom * np.pi * random.random()
            orn = pb.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, urdf_name)
            uid = pb.loadURDF(urdf_path, [xpos, ypos, -0.02], [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            for _ in range(20):
                pb.stepSimulation()
        return objectUids

    def _get_observation(self):
        """Return the observation as an image (eye-in-hand).
        """

        pos, ori = pb.getBasePositionAndOrientation(self._jaco.jacoUid)
        com_p = (pos[0]+0.35, pos[1], pos[2]+0.3)
        ori_euler = [3*math.pi/4,0,math.pi/2] #[3*math.pi/4,0,math.pi/2]
        com_o = pb.getQuaternionFromEuler(ori_euler)
        #com_p, com_o, _, _, _, _ = pb.getLinkState(self._jaco.jacoUid, 8, computeForwardKinematics=True) # 6, 7
        rot_matrix = pb.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)  # reshape list of 9 values to a 3x3 matrix

        #show camera position in environment (mug displays camera)
        #mug = pb.loadURDF(os.path.join(self._urdfRoot, 'objects/mug.urdf'),com_p,com_o)
        
        # com_euler = p.getEulerFromQuaternion(com_o)
        # roll, pitch, yaw = com_euler

        com_p = list(com_p)
        # [1] += 0.08; [2] -= 0.05
        # [1] += 0.1; [2] -= 0.05
        # com_p[1] += 0.08  # 0.1
        # com_p[1] += 0.1  # 0.1
        com_p[2] -= -0.01  # 0.05, 0.1
        #print('------------------------------------------------')
        #print('com_p:', list(com_p))
        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotate camera vector and up vector
        camera_vector = rot_matrix.dot(init_camera_vector)
        #print('------------------------------------------------')
        #print('camera_vector:', camera_vector)

        # print('com_p + 0.1 * camera_vector:\t', com_p + 0.1 * camera_vector)

        up_vector = rot_matrix.dot(init_up_vector)

        #some work here
        view_matrix = pb.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)

        h = 128 #self._width  # 64
        w = 128 #self._height  # 64
        far = 10.0
        near = 0.01
        aspect = w / h

        proj_matrix = pb.computeProjectionMatrixFOV(fov=60,
                                                aspect=aspect,  # 4:3 aspect ratio
                                                nearVal=0.01,  # 0.1, 0.02
                                                farVal=10.0)  # 100.0, 2.0

        images = pb.getCameraImage(width=w,
                                height=h,
                                viewMatrix=view_matrix,
                                projectionMatrix=proj_matrix,
                                renderer=pb.ER_TINY_RENDERER)

        # get rgb observation
        rgb = np.array(images[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (h, w, 4))  # * 1. / 255.
        rgb = rgb[:, :, :3]  # discard alpha channel

        # get depth observation
        depth_buffer = np.array(images[3], dtype=np.float32)
        depth_buffer = np.reshape(depth_buffer, (h, w))
        depth = far * near / (far - (far - near) * depth_buffer)
        depth = np.stack([depth, depth, depth], axis=0)
        depth = np.reshape(depth, (h, w, 3))
        
        segmentation = images[4]
        
        return rgb, depth, segmentation


    def step(self, action):
        """Environment step.

        Args:
        action: 5-vector parameterizing XYZ offset, vertical angle offset
        (radians), and grasp angle (radians).
        Returns:
        observation: Next observation.
        reward: Float of the per-step reward as a result of taking the action.
        done: Bool of whether or not the episode has ended.
        debug: Dictionary of extra information provided by environment.
        """
        dv = self._dv  # velocity per physics step.
        if self._isDiscrete:
        # Static type assertion for integers.
            assert isinstance(action, int)
            if self._removeAutoXDistance:
                dx = [0, -dv, dv, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv][action]
            else:
                dx = dv 
                dy = [0, 0, 0, -dv, dv, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv][action]
        else:
            dy = dv * action[1]
            dz = dv * action[2]
            if self._removeAutoXDistance:
                dx = dv * action[0]
            else:
                dx = dv

        return self._step_continuous([dx, dy, dz, 0])  #0 for fingerAngle          


    def _step_continuous(self, action):
        """Applies a continuous velocity-control action.

        Args:
        action: 5-vector parameterizing XYZ offset, vertical angle offset
        (radians), and grasp angle (radians).
        Returns:
        observation: Next observation.
        reward: Float of the per-step reward as a result of taking the action.
        done: Bool of whether or not the episode has ended.
        debug: Dictionary of extra information provided by environment.
        """
        # Perform commanded action.
        self._env_step += 1
        self._jaco.applyAction(action)
        for _ in range(self._actionRepeat):
            pb.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                break

        # If we are close to the mug, attempt grasp.
        state = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)
        end_effector_pos = state[0]
        mugpos =  np.array(pb.getBasePositionAndOrientation(3)[0])
        
        if end_effector_pos[0] >=0.2:
            # print(state[0])
            # print(mugpos)
            finger_angle = 0.6 
            tip_angle = finger_angle
            for _ in range(150):
                grasp_action = [0, 0, 0, finger_angle]
                self._jaco.applyAction(grasp_action)
                pb.stepSimulation()
                #if self._renders:
                #  time.sleep(self._timeStep)
                finger_angle += 0.1 / 100.
                if finger_angle > 2:
                    finger_angle = 2 #upper limit
            #added, close fingertips
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
                    finger_angle = 2 #upper limit
            for _ in range(150):
                grasp_action = [0, 0, 0.001, finger_angle]
                self._jaco.applyAction(grasp_action)
                pb.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                finger_angle += 0.1 / 100.
                if finger_angle < 2:
                    finger_angle = 2
            self._attempted_grasp = True
            state = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)
            mugpos = np.array(pb.getBasePositionAndOrientation(3)[0])
            # print(state[0])
            # print(mugpos)

        observation = self._get_observation()
        done = self._termination()
        reward = self._reward()

        debug = {'grasp_success': self._graspSuccess}
        return observation, reward, done, debug  

    
    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        self._graspSuccess = 0
        #pos_jaco = pb.getLinkState(self._jaco.jacoUid,self._jaco.jacoEndEffectorIndex)[0]

        for uid in self._objectUids:
            pos, _ = pb.getBasePositionAndOrientation(uid)
            # If mug is above height, provide reward.
            if pos[2] > 0.1:
                self._graspSuccess += 1
                reward = reward + 1
                break
        return reward

    
    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._attempted_grasp or self._env_step >= self._maxSteps


    def _get_random_object(self, num_objects, test):
        """
        Args:
        num_objects:
            Number of graspable objects. For now just the mug.

        Returns:
        A list of urdf filenames.
        """
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

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _reset = reset
        _step = step