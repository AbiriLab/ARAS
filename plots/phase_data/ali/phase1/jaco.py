import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as pb
import numpy as np
import copy
import math
import pybullet_data

class jaco:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerThumbForce = 3
        self.fingerAForce = 2
        self.fingerBForce = 2
        self.fingerforce = 6
        self.fingerIndices = [9,11,13]
        self.fingertipforce = 2 
        self.fingerThumbtipforce = 3 * self.fingertipforce
        self.fingerTipForce = 2
        self.fingertipIndices = [10,12,14]
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 42  # = 14+14+14 ll ul rp???
        self.useOrientation = 1
        self.jacoEndEffectorIndex = 8 #use to get pose of end effector
        #self.jacoGripperIndex:
        self.jacoThumbIndex = 9
        self.jacoFingerAIndex = 11 
        self.jacoFingerBIndex = 13
        #lower limits for null space
        self.ll = [0,0,-6.28318530718,0.820304748437,0.331612557879,-6.28318530718,-6.28318530718,-6.28318530718,0,0,0,0,0,0,0]
        #upper limits for null space
        self.ul = [0,0,6.28318530718,5.46288055874,5.9515727493,6.28318530718,6.28318530718,6.28318530718,0,2,0,2,0,2,0]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.rp = [0,0.25,1.85,3,1.25,3.5,4.5,3,3,0.5,0,0.5,0,0.5,0]
        #joint damping coefficents
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]
        self.reset()

    def reset(self):
        
        objects = pb.loadURDF(os.path.join('jaco_reza/j2n6s300_color copy.urdf',), useFixedBase=True)
        #convert into to tuple
        objects = (objects,)
        #print(type(objects))
        self.jacoUid = objects[0]

        #keep robot at same position after every reset
        jaco_orientation_euler = [0,math.pi/2,0]
        jaco_orientation_quaternion = pb.getQuaternionFromEuler(jaco_orientation_euler)
        pb.resetBasePositionAndOrientation(self.jacoUid,[-0.75,0,0.5],jaco_orientation_quaternion)

        self.jointPositions = [
            0,0,0,1.5,1.55,0,0,math.pi,0,0,0.25,0,0.25,0,0.25
        ]
        self.numJoints = pb.getNumJoints(self.jacoUid)
        
        for jointIndex in range(self.numJoints):
            pb.resetJointState(self.jacoUid, jointIndex, self.jointPositions[jointIndex])
            pb.setJointMotorControl2(self.jacoUid,
                              jointIndex,
                              pb.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce) 

        # self.trayUid = pb.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
        #                       0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
    
        #change dynamics of gripper and finger to grasp objects
        #TODO: Change dynamics directly in urdf file
        for i in range(9,14):
            pb.changeDynamics(self.jacoUid,i,mass=0.3,lateralFriction=1,restitution=0,spinningFriction=0.1,contactStiffness=30000,contactDamping=1000)
            #print(pb.getDynamicsInfo(jacoUid,i))
            #print(pb.getLinkState(jacoUid,i))
        pb.changeDynamics(self.jacoUid,8,mass=0.5,lateralFriction=1,restitution=0.0,spinningFriction=0.1,contactStiffness=10,contactDamping=10)
        #What is exactly the jaco and jaco EndEffektor???    
        self.endEffectorPos = [0, 0, 0.035] #changed from list(pb.getLinkState(self.jacoUid, self.jacoEndEffectorIndex)[0]) 
        self.endEffectorOrientation = pb.getLinkState(self.jacoUid, self.jacoEndEffectorIndex)[1]
        #print(pb.getLinkState(self.jacoUid,self.jacoEndEffectorIndex)[0])
        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = pb.getJointInfo(self.jacoUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                #print("motorname")
                #print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)


    def getActionDimension(self):
        if (self.useInverseKinematics):
            return len(self.motorIndices)
        return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    #just for Thumb, also include other grippers later
    def getObservation(self):
        observation = []
        stateThumb = pb.getLinkState(self.jacoUid, self.jacoThumbIndex)
        pos = stateThumb[0]
        orn = stateThumb[1]
        euler = pb.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation
    
    def applyAction(self, motorCommands):

        #print ("self.numJoints")
        #print (self.numJoints)
        if (self.useInverseKinematics):

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            fingerAngle = motorCommands[3]

            state = pb.getLinkState(self.jacoUid, self.jacoEndEffectorIndex)
            actualEndEffectorPos = state[0]

            self.endEffectorPos[0] = self.endEffectorPos[0] + dx
            self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            self.endEffectorPos[2] = self.endEffectorPos[2] + dz
          
            # if self.endEffectorPos[2] < 0.01:
            #     self.endEffectorPos[2] = 0.01
            
            
            pos = self.endEffectorPos
            #print(pos)
            orn = state[1] #self.endEffectorOrientation #state[1] # leave Wirst Orientation like it is for now        # -math.pi,yaw]
            
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = pb.calculateInverseKinematics(self.jacoUid, self.jacoEndEffectorIndex, pos,
                                                            orn, self.ll, self.ul, self.jr, self.rp)
                else:
                    jointPoses = pb.calculateInverseKinematics(self.jacoUid,
                                                            self.jacoEndEffectorIndex,
                                                            pos,
                                                            lowerLimits=self.ll,
                                                            upperLimits=self.ul,
                                                            jointRanges=self.jr,
                                                            restPoses=self.rp)
            else:
                if (self.useOrientation == 1):
                    jointPoses = pb.calculateInverseKinematics(self.jacoUid,
                                                            self.jacoEndEffectorIndex,
                                                            pos, # state[0],#pos,
                                                            orn, #state[1], #orn,
                                                            jointDamping=self.jd)
                else:
                    jointPoses = pb.calculateInverseKinematics(self.jacoUid, self.jacoEndEffectorIndex, pos)

            list_jointPoses = list(jointPoses)
            list_jointPoses.insert(0,0)
            list_jointPoses.insert(1,0)
            list_jointPoses.insert(8,0)

            #if tip revolute these are not needed
            #list_jointPoses.insert(10,0)
            #list_jointPoses.insert(12,0)
            #list_jointPoses.insert(14,0)
            

            jointPoses = tuple(list_jointPoses)

            if (self.useSimulation):
                    pb.setJointMotorControlArray(
                        bodyUniqueId=self.jacoUid,
                        jointIndices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                        controlMode=pb.POSITION_CONTROL,
                        targetPositions=jointPoses,   
                        targetVelocities=[0]*len(jointPoses),
                        forces=[self.maxForce]*len(jointPoses),
                        velocityGains=[1]*len(jointPoses)
                    )
                    pb.stepSimulation()
            else:
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    pb.resetJointState(self.jacoUid, i, jointPoses[i])

            if fingerAngle != 0:
                #thumb
                pb.setJointMotorControlArray(
                            bodyUniqueId=self.jacoUid,
                            jointIndices=self.fingerIndices,
                            controlMode=pb.POSITION_CONTROL,
                            targetPositions=[fingerAngle]*len(self.fingerIndices),   
                            targetVelocities=[0]*len(self.fingerIndices),
                            forces=[self.fingerThumbForce,self.fingerAForce,self.fingerBForce],
                            velocityGains=[1]*len(self.fingerIndices)
                )
                # TODO: Check if this grasp is working like that
                # pb.setJointMotorControl2(self.jacoUid,
                #                         9,
                #                         pb.POSITION_CONTROL,
                #                         targetPosition=fingerAngle,
                #                         force=self.fingerThumbForce)
                # #fingerA
                # pb.setJointMotorControl2(self.jacoUid,
                #                         11,
                #                         pb.POSITION_CONTROL,
                #                         targetPosition=fingerAngle,
                #                         force=self.fingerAForce)
                # #fingerB
                # pb.setJointMotorControl2(self.jacoUid,
                #                         13,
                #                         pb.POSITION_CONTROL,
                #                         targetPosition=fingerAngle,
                #                         force=self.fingerBForce)


        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                pb.setJointMotorControl2(self.jacoUid,
                                        motor,
                                        pb.POSITION_CONTROL,
                                        targetPosition=motorCommands[action],
                                        force=self.maxForce)