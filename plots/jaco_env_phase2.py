import random
import os
from gym import spaces
import time
import math
import pybullet as pb
import jaco_phase2 as jaco
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
largeValObservation = 100


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
                removeAutoXDistance=True, #changed
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
        removeAutoXDistance: If false, there is a "distance hack" where the gripper
            automatically moves in the x-direction for each action, except the grasp action. 
            If true, the environment is harder and the policy chooses the distance displacement.
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
        self._removeAutoXDistance = removeAutoXDistance
        self._objectRandom = objectRandom
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

        #define number of actions
        if (self._isDiscrete):
            if self._removeAutoXDistance:
                self.action_space = spaces.Discrete(8)
            else:
                self.action_space = spaces.Discrete(8)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dy, dz, closeGripper
            if self._removeAutoXDistance:
                self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, closeGripper
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

        # Load plane and table in the environment
        pb.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"),0,0,-0.66)
        pb.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -0.66,
                0.000000, 0.000000, 0.0, 1.0)

        # Set gravity 
        pb.setGravity(0, 0, -9.81)

        # Load jaco robotic arm into the environment
        self._jaco = jaco.jaco(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        pb.stepSimulation()

        # Load random object in the environment, currently just the mug
        urdfList = self._get_random_object(self._numObjects, self._isTest)

        # Place loaded object randomly in the environment
        self._objectUids = self._randomly_place_objects(urdfList)
        
        # Get camera images (rgb,depth,segmentation)
        self._observation = self._get_observation()

        # The following is only important for the reward function of this phase
        
        # Get mug position in xyz
        self._mugPos = np.array(pb.getBasePositionAndOrientation(3)[0])
        # Adjust to "true" center of mug. Without self._mugPos[2] (z-direction) is the bottom of the cup
        self._mugPos[2] = self._mugPos[2] + 0.03

        # Get current gripper and mug position for current euclidean distance
        self.endEffectorPos_original = np.array(pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)[0])
        # Delete x pos
        posGripper = np.delete(self.endEffectorPos_original,[0])
        posMug = np.delete(self._mugPos,[0])
        
        # GetEuclideanDistance between mug and gripper
        # Without distance hack
        self._dist_orignial = np.linalg.norm(self._mugPos-self.endEffectorPos_original)  
        # With distance hack
        self._dist_orignialXZ = np.linalg.norm(posMug-posGripper)        

        

        return np.array(self._observation[1])


    def _randomly_place_objects(self, urdfList):
        """Randomly place the objects on the table.

        Args:
        urdfList: The list of urdf files to place on the table.

        Returns:
        The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []
        for urdf_name in urdfList:
            xpos = random.uniform(0.16, 0.23)   
                        
            # If distance hack is false, init mug in a square area on the table
            # If distance hack is true , init mug in a trapezoid area on the table
            if self._removeAutoXDistance:
                ypos = random.uniform(0,0.2)
            else:
                width = 0.1 + (0.16-xpos)/(-0.7)
                ypos = random.uniform(-width,width)  

            zpos = -0.02
            angle = -np.pi / 2 + self._objectRandom * np.pi * random.random()
            orn = pb.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, urdf_name)
            uid = pb.loadURDF(urdf_path, [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            for _ in range(20):
                pb.stepSimulation()
        return objectUids

    def _get_observation(self):
        """Return the observation as an image.
        """

        # Initializing camera position and orientation
        pos, ori = pb.getBasePositionAndOrientation(self._jaco.jacoUid)
        com_p = (pos[0]+0.35, pos[1], pos[2]+0.3)
        ori_euler = [3*math.pi/4,0,math.pi/2] 
        com_o = pb.getQuaternionFromEuler(ori_euler)
        rot_matrix = pb.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)  # reshape list of 9 values to a 3x3 matrix

        com_p = list(com_p)
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

        # Compute view matrix of cameras view in the environment
        view_matrix = pb.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)

        # Properties of camera
        # Heigth and width of the image
        h = 128 #self._width  
        w = 128 #self._height  
        far = 10.0
        near = 0.01
        # Aspect ratio
        aspect = w / h

        proj_matrix = pb.computeProjectionMatrixFOV(fov=60,
                                                aspect=aspect,  
                                                nearVal=0.01,  # 0.1, 0.02
                                                farVal=10.0)  # 100.0, 2.0

        images = pb.getCameraImage(width=w,
                                height=h,
                                viewMatrix=view_matrix,
                                projectionMatrix=proj_matrix,
                                renderer=pb.ER_TINY_RENDERER)

        # Get rgb observation
        rgb = np.array(images[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (h, w, 4))  # * 1. / 255.
        rgb = rgb[:, :, :3]  # discard alpha channel

        # Get depth observation
        depth_buffer = np.array(images[3], dtype=np.float32)
        depth_buffer = np.reshape(depth_buffer, (h, w))
        depth = far * near / (far - (far - near) * depth_buffer)
        depth = np.stack([depth, depth, depth], axis=0)
        depth = np.reshape(depth, (h, w, 3))

        # Get segmentation observation
        segmentation = images[4]
        
        return rgb, depth, segmentation


    def step(self, action):
        """Environment step.

        Args:
        action: 4-vector parameterizing XYZ offset and grasp action.
        Returns:
        observation: Next observation.
        reward: Float of the per-step reward as a result of taking the action.
        done: Bool of whether or not the episode has ended.
        debug: Dictionary of extra information provided by environment.
        """

        dv = self._dv  # Velocity per physics step.

        """
        If orientation should be used, add roll pitch yaw in every case and in the applyAction function
        """

        # For discrete action space
        if self._isDiscrete:
        # Static type assertion for integers.
            assert isinstance(action, int)
            if self._removeAutoXDistance:
                dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
                close_gripper = [0, 0, 0, 0, 0, 0, 0, 1][action]
            else:
                #changed from original
                dx = dv 
                dy = [0, 0, 0, -dv, dv, 0, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv, 0][action]
                close_gripper = [0, 0, 0, 0, 0, 0, 0, 1][action]
        # For continous action space
        else:
            dz = dv * action[2]
            dy = dv * action[1]
            if action[3] < 0.5: 
                close_gripper = 0   # False
            else:
                close_gripper = 1   # True
            if self._removeAutoXDistance:
                dz = dv * action[2]
            else:
                dx = dv

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
        if action[3] != 0:
            action[0] = 0 
            action[1] = 0
            action[2] = 0  

            # Get current gripper and mug position for current euclidean distance, only for reward function relevant and used!
            posGripper = np.array(pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)[0])
            posMug = np.array(pb.getBasePositionAndOrientation(3)[0])
            posMug[2] = posMug[2] + 0.03
            # Delete x pos
            posGripperXZ = np.delete(posGripper,[0])
            posMugXZ = np.delete(posMug,[0])
            # GetEuclideanDistance between mug and gripper before action
            self._distXZ_beforeGrasp = np.linalg.norm(posMugXZ-posGripperXZ)

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
        if action[3] != 0:
            self._graspPosition = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)[0]
            self._distXYZ_beforeGrasp = np.linalg.norm(posMug-self._graspPosition)
            
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

    
    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if the objects height is above .05 at the end of the
        episode. 
        In addition, small pos/neg reward is provided depending on the grippers position to the mug after a grasp action
        """
        reward = 0
        self._graspSuccess = 0
        
        for uid in self._objectUids:
            pos, _ = pb.getBasePositionAndOrientation(uid)

            # For testing just return reward if mug is lifed successfully, just important for the evaluation
            if self._isTest:
                if pos[2] > 0.05:
                    reward = 1
            else:

                # Two hyperparameters, not optimal searched yet. Try to find the best combination of both in a future hyperparameter search
                step_punishment = 0.01  #[0.005,0.01,0.02]
                distance_factor = 0.2   #[0.05,0.1,0.2]

                if (self._attempted_grasp == True):
                    # Without distance hack
                    if self._removeAutoXDistance:
                        reward = reward + distance_factor*(self._dist_orignial-self._distXYZ_beforeGrasp)/self._dist_orignial  #distance_factor * X 
                    # With distance hack
                    else:
                        reward = reward + distance_factor*(self._dist_orignialXZ-self._distXZ_beforeGrasp)/self._dist_orignialXZ #distance_factor * X


                # If object is above height, provide reward.
                if pos[2] > 0.05:
                        self._graspSuccess += 1
                        step_penalty = step_punishment*self._env_step 
                        reward = reward - step_penalty
                        # Clip reward between -1 and 1, improves stability
                        if reward > 1:
                            reward = 1
                        if reward < -1:
                            reward = -1
                break
            
        return reward

    
    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._attempted_grasp or self._env_step >= self._maxSteps


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

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _reset = reset
        _step = step