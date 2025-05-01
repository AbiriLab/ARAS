"""
This file implements a Hindsight Optimization approach for shared autonomy in robotic manipulation.
The HindsightOptimizer class maintains belief distributions over potential goals (objects and containers)
and uses these beliefs to select actions. It generates synthetic user inputs based on the current state
and intended goal, updates beliefs based on these inputs, and selects actions that maximize the expected
value across all potential goals weighted by their probabilities. The implementation supports both
object pickup and container placement phases, with different belief distributions for each phase.
"""

import numpy as np
from collections import deque
class HindsightOptimizer:
    def __init__(self, env):

        self.env = env
        self.object_beliefs = None
        self.container_beliefs = None
        self.object_id_to_idx = {}
        self.container_id_to_idx = {}
        
        # Parameters
        self.belief_update_rate = 2.0  
        self.distance_threshold = 0.06  
        self.user_input_noise = 0.1  
        
        self.action_mapping = {
            0: "HOLD",   
            1: "LEFT",    
            2: "RIGHT",  
            3: "FORWARD", 
            4: "BACKWARD" 
        }
        
        self.prev_inputs = deque(maxlen=3)
        
        self.initialized = False
        
    def initialize_beliefs(self):
        if hasattr(self.env, '_objectUids') and hasattr(self.env, 'container_uid'):
            self.object_beliefs = np.ones(len(self.env._objectUids)) / len(self.env._objectUids)
            self.container_beliefs = np.ones(len(self.env.container_uid)) / len(self.env.container_uid)
            
            self.object_id_to_idx = {obj_id: i for i, obj_id in enumerate(self.env._objectUids)}
            self.container_id_to_idx = {cont_id: i for i, cont_id in enumerate(self.env.container_uid)}
            
            self.initialized = True
            return True
        return False
        
    def generate_synthetic_user_input(self):

        if not self.initialized:
            if not self.initialize_beliefs():
                return "neutral"  
        
        gripper_pos = self.env._getGripper()[:2]
        
        if self.env._gripperState == 'open':
            delta_y = self.env._mugPos[1] - gripper_pos[1]
        else:
            delta_y = self.env._containerPos[1] - gripper_pos[1]
                

        delta_y = 0 if (abs(delta_y) < 0.025) else np.sign(delta_y)
        if delta_y == 0:
            user_input = "neutral"
        elif delta_y > 0:
            user_input = "left"
        else:
            user_input = "right"
            
        self.prev_inputs.append(user_input)
        
        return user_input
    
    def update_beliefs(self, user_input):

        if not self.initialized:
            if not self.initialize_beliefs():
                return
                
        gripper_pos = self.env._getGripper()[:2]
        
        if self.env._gripperState == 'open':
            for i, obj_id in enumerate(self.env._objectUids):
                obj_pos, _ = self.env._p.getBasePositionAndOrientation(obj_id)
                obj_pos = np.array(obj_pos[:2])
                
                delta_y = obj_pos[1] - gripper_pos[1]
                
                if user_input == "left" and delta_y > 0:
                    self.object_beliefs[i] *= self.belief_update_rate
                elif user_input == "right" and delta_y < 0:
                    self.object_beliefs[i] *= self.belief_update_rate
                elif user_input == "neutral" and abs(delta_y) < 0.05:
                    self.object_beliefs[i] *= self.belief_update_rate
            
            if np.sum(self.object_beliefs) > 0:
                self.object_beliefs = self.object_beliefs / np.sum(self.object_beliefs)
                
        else:
            for i, cont_id in enumerate(self.env.container_uid):
                cont_pos, _ = self.env._p.getBasePositionAndOrientation(cont_id)
                cont_pos = np.array(cont_pos[:2])
                
                delta_y = cont_pos[1] - gripper_pos[1]
                
                if user_input == "left" and delta_y > 0:
                    self.container_beliefs[i] *= self.belief_update_rate
                elif user_input == "right" and delta_y < 0:
                    self.container_beliefs[i] *= self.belief_update_rate
                elif user_input == "neutral" and abs(delta_y) < 0.05:
                    self.container_beliefs[i] *= self.belief_update_rate
            
            if np.sum(self.container_beliefs) > 0:
                self.container_beliefs = self.container_beliefs / np.sum(self.container_beliefs)
    
    def compute_action_for_goal(self, goal_pos):

        gripper_pos = self.env._getGripper()[:2]
        delta_x = goal_pos[0] - gripper_pos[0]
        delta_y = goal_pos[1] - gripper_pos[1]
        distance = np.linalg.norm(np.array(goal_pos) - np.array(gripper_pos))
        
        if distance < self.distance_threshold:
            return 3  

        abs_x = abs(delta_x)
        abs_y = abs(delta_y) 
        unnormalized_probs = np.array([abs_y + 0.04, abs_x])
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        prob_lateral = normalized_probs[0]
        prob_forward_backward = normalized_probs[1]
        category = np.random.choice([0, 1], p=[prob_lateral, prob_forward_backward])

        if category == 0:
       
            if delta_y > 0:
                return 0 
            else:
                return 1  
        else:
           
            if delta_x > 0:
                return 3  
            else:
                return 4  
    
    def select_action(self, user_input=None):

        if not self.initialized:
            if not self.initialize_beliefs():
                return 0  
                
        if user_input is None:
            user_input = self.generate_synthetic_user_input()
        
        self.update_beliefs(user_input)
        
        action_values = np.zeros(5)  
        
        if self.env._gripperState == 'open':
            for i, obj_id in enumerate(self.env._objectUids):
                obj_pos, _ = self.env._p.getBasePositionAndOrientation(obj_id)
                obj_pos = np.array(obj_pos[:2])
                
                optimal_action = self.compute_action_for_goal(obj_pos)
                
                action_values[optimal_action] += self.object_beliefs[i]
        else:
            for i, cont_id in enumerate(self.env.container_uid):
                cont_pos, _ = self.env._p.getBasePositionAndOrientation(cont_id)
                cont_pos = np.array(cont_pos[:2])
                optimal_action = self.compute_action_for_goal(cont_pos)
                action_values[optimal_action] += self.container_beliefs[i]
        
        
        return np.argmax(action_values)
    
    def reset(self):
        self.initialized = False
        self.initialize_beliefs()
        
        self.prev_inputs.clear()
        
    