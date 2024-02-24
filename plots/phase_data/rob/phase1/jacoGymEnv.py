import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as pb
import jaco
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class jacoGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=1000):
    #print("jacoGymEnv __init__")
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

    #change it maybe all to p instead of pb? nescessary
    self._p = pb
    if self._renders:
      cid = pb.connect(pb.SHARED_MEMORY)
      if (cid < 0):
        cid = pb.connect(pb.GUI)
      pb.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      pb.connect(pb.DIRECT)
    # timinglog = pb.startStateLogging(pb.STATE_LOGGING_PROFILE_TIMINGS, "jacoTimings.json")
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    # print("observationDim")
    # print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None


    def reset(self):
        #print("jacoGymEnv _reset")
        self.terminated = 0
        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setTimeStep(self._timeStep)
        # pb.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])  # not necessary

        pb.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                0.000000, 0.000000, 0.0, 1.0)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0 + 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = pb.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = pb.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                                orn[0], orn[1], orn[2], orn[3])

        pb.setGravity(0, 0, -10)
        self._jaco = jaco.jaco(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        pb.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)


    def __del__(self):
        pb.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = self._jaco.getObservation()
        gripperState = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoGripperIndex)
        gripperPos = gripperState[0]
        gripperOrn = gripperState[1]
        blockPos, blockOrn = pb.getBasePositionAndOrientation(self.blockUid)

        invGripperPos, invGripperOrn = pb.invertTransform(gripperPos, gripperOrn)
        gripperMat = pb.getMatrixFromQuaternion(gripperOrn)
        dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
        dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
        dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

        gripperEul = pb.getEulerFromQuaternion(gripperOrn)
        #print("gripperEul")
        #print(gripperEul)
        blockPosInGripper, blockOrnInGripper = pb.multiplyTransforms(invGripperPos, invGripperOrn,
        blockPos, blockOrn)
        projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
        blockEulerInGripper = pb.getEulerFromQuaternion(blockOrnInGripper)
        #print("projectedBlockPos2D")
        #print(projectedBlockPos2D)
        #print("blockEulerInGripper")
        #print(blockEulerInGripper)

        #we return the relative x,y position and euler angle of block in gripper space
        blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

        #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
        #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
        #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

        self._observation.extend(list(blockInGripperPosXYEulZ))
        return self._observation

    def step(self, action):
        if (self._isDiscrete):
            dv = 0.005
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        else:
            #print("action[0]=", str(action[0]))
            dv = 0.005
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.05
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        return self.step2(realAction)

    def step2(self, action):
        for i in range(self._actionRepeat):
            self._jaco.applyAction(action)
            pb.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()
   
            #print("self._envStepCounter")
        #print(self._envStepCounter)

        done = self._termination()
        npaction = np.array([
            action[3]
        ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
        actionCost = np.linalg.norm(npaction) * 10.
        #print("actionCost")
        #print(actionCost)
        reward = self._reward() - actionCost
        #print("reward")
        #print(reward)

        #print("len=%r" % len(self._observation))

        return np.array(self._observation), reward, done, {}

    
    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._jaco.jacoUid)

        #TODO check if this is nescessary and what it does or the env and how to adjust it
        #probably cam moving with robot instead steadz cam?
        # add code here ------------------------------------------------------------------------------------
        # Center of mass position and orientation (of link-7)
        com_p, com_o, _, _, _, _ = pb.getLinkState(self._jaco.jacoUid, 6, computeForwardKinematics=True)
        rot_matrix = pb.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)  # reshape list of 9 values to a 3x3 matrix
        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotate camera vector and up vector
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix = pb.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)

        '''view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)'''

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                        nearVal=0.1,
                                                        farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                height=RENDER_HEIGHT,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)
        # renderer=self._p.ER_BULLET_HARDWARE_OPENGL

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        #print (self._jaco.endEffectorPos[2])
        state = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)
        actualEndEffectorPos = state[0]

        #print("self._envStepCounter")
        #print(self._envStepCounter)
        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True
        maxDist = 0.005
        closestPoints = pb.getClosestPoints(self._jaco.trayUid, self._jaco.jacoUid, maxDist)

        if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
            self.terminated = 1

            #print("terminating, closing gripper, attempting grasp")
            #start grasp and terminate
            fingerAngle = 0.3
            for i in range(100):
                graspAction = [0, 0, 0.0001, 0, fingerAngle]
                self._jaco.applyAction(graspAction)
                pb.stepSimulation()
                fingerAngle = fingerAngle - (0.3 / 100.)
                if (fingerAngle < 0):
                    fingerAngle = 0

            for i in range(1000):
                graspAction = [0, 0, 0.001, 0, fingerAngle]
                self._jaco.applyAction(graspAction)
                pb.stepSimulation()
                blockPos, blockOrn = pb.getBasePositionAndOrientation(self.blockUid)
                if (blockPos[2] > 0.23):
                    #print("BLOCKPOS!")
                    #print(blockPos[2])
                    break
                state = pb.getLinkState(self._jaco.jacoUid, self._jaco.jacoEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2] > 0.5):
                    break

            self._observation = self.getExtendedObservation()
            return True
        return False

    def _reward(self):

        #rewards is height of target object
        blockPos, blockOrn = pb.getBasePositionAndOrientation(self.blockUid)
        closestPoints = pb.getClosestPoints(self.blockUid, self._jaco.jacoUid, 1000, -1,
                                        self._jaco.jacoEndEffectorIndex)

        reward = -1000

        numPt = len(closestPoints)
        #print(numPt)
        if (numPt > 0):
            #print("reward:")
            reward = -closestPoints[0][8] * 10
        if (blockPos[2] > 0.2):
            reward = reward + 10000
            print("successfully grasped a block!!!")
            #print("self._envStepCounter")
            #print(self._envStepCounter)
            #print("self._envStepCounter")
            #print(self._envStepCounter)
            #print("reward")
            #print(reward)
        #print("reward")
        #print(reward)
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step