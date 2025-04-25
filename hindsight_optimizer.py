import numpy as np
import torch
import random
from collections import deque
import pybullet as pb

class HindsightOptimizer:
    def __init__(self, env):
        """
        Hindsight optimizer for shared autonomy using synthetic user inputs.
        
        Args:
            env: The jacoDiverseObjectEnv instance
        """
        self.env = env
        
        # Initialize with empty beliefs - will be populated after env.reset()
        self.object_beliefs = None
        self.container_beliefs = None
        self.object_id_to_idx = {}
        self.container_id_to_idx = {}
        
        # Parameters
        self.belief_update_rate = 2.0  # Stronger belief updates
        self.distance_threshold = 0.06  # Threshold for considering goals reached
        self.user_input_noise = 0.1  # Probability of random input (simulating user error)
        
        # Action mapping from environment
        self.action_mapping = {
            0: "HOLD",    # Hold position
            1: "LEFT",    # Move left
            2: "RIGHT",   # Move right
            3: "FORWARD", # Move forward
            4: "BACKWARD" # Move backward
        }
        
        # Store the previous inputs for consistency
        self.prev_inputs = deque(maxlen=3)
        
        # Flag to track if we've initialized beliefs
        self.initialized = False
        
    def initialize_beliefs(self):
        """Initialize beliefs after environment has been reset and objects created."""
        if hasattr(self.env, '_objectUids') and hasattr(self.env, 'container_uid'):
            # Initialize belief distributions
            self.object_beliefs = np.ones(len(self.env._objectUids)) / len(self.env._objectUids)
            self.container_beliefs = np.ones(len(self.env.container_uid)) / len(self.env.container_uid)
            
            # Mapping of object IDs to indices
            self.object_id_to_idx = {obj_id: i for i, obj_id in enumerate(self.env._objectUids)}
            self.container_id_to_idx = {cont_id: i for i, cont_id in enumerate(self.env.container_uid)}
            
            self.initialized = True
            return True
        return False
        
    def generate_synthetic_user_input(self):
        """
        Generate synthetic user input based on the current state and intended goal.
        Returns:
            str: "left", "right", or "neutral"
        """
        if not self.initialized:
            if not self.initialize_beliefs():
                return "neutral"  # Default if not initialized
        
        gripper_pos = self.env._getGripper()[:2]
        
        # Determine target based on gripper state
        if self.env._gripperState == 'open':
            # Target is the intended object (mug)
            target_pos = self.env._mugPos[:2]
            delta_y = self.env._mugPos[1] - gripper_pos[1]
            true_intention_idx = self.object_id_to_idx.get(self.env.intention_object, 0)
        else:
            # Target is the intended container (green container)
            target_pos = self.env._containerPos[:2]
            delta_y = self.env._containerPos[1] - gripper_pos[1]
            true_intention_idx = self.container_id_to_idx.get(self.env.intention_container, 0)
                

        delta_y = 0 if (abs(delta_y) < 0.025) else np.sign(delta_y)
        # Generate user input based on relative position
        if delta_y == 0:
            user_input = "neutral"
        elif delta_y > 0:
            user_input = "left"
        else:
            user_input = "right"
            
        # Store the input
        self.prev_inputs.append(user_input)
        
        return user_input
    
    def update_beliefs(self, user_input):
        """
        Update belief distributions based on user input.
        
        Args:
            user_input: User input ("left", "right", or "neutral")
        """
        if not self.initialized:
            if not self.initialize_beliefs():
                return
                
        gripper_pos = self.env._getGripper()[:2]
        
        # Update beliefs based on which phase we're in
        if self.env._gripperState == 'open':
            # Update object beliefs
            for i, obj_id in enumerate(self.env._objectUids):
                obj_pos, _ = self.env._p.getBasePositionAndOrientation(obj_id)
                obj_pos = np.array(obj_pos[:2])
                
                # Calculate direction to object
                delta_y = obj_pos[1] - gripper_pos[1]
                
                if user_input == "left" and delta_y > 0:
                    # User wants to go left and object is to the left
                    self.object_beliefs[i] *= self.belief_update_rate
                elif user_input == "right" and delta_y < 0:
                    # User wants to go right and object is to the right
                    self.object_beliefs[i] *= self.belief_update_rate
                elif user_input == "neutral" and abs(delta_y) < 0.05:
                    # User is satisfied with y position
                    self.object_beliefs[i] *= self.belief_update_rate
            
            # Normalize object beliefs
            if np.sum(self.object_beliefs) > 0:
                self.object_beliefs = self.object_beliefs / np.sum(self.object_beliefs)
                
        else:
            # Update container beliefs
            for i, cont_id in enumerate(self.env.container_uid):
                cont_pos, _ = self.env._p.getBasePositionAndOrientation(cont_id)
                cont_pos = np.array(cont_pos[:2])
                
                # Calculate direction to container
                delta_y = cont_pos[1] - gripper_pos[1]
                
                if user_input == "left" and delta_y > 0:
                    # User wants to go left and container is to the left
                    self.container_beliefs[i] *= self.belief_update_rate
                elif user_input == "right" and delta_y < 0:
                    # User wants to go right and container is to the right
                    self.container_beliefs[i] *= self.belief_update_rate
                elif user_input == "neutral" and abs(delta_y) < 0.05:
                    # User is satisfied with y position
                    self.container_beliefs[i] *= self.belief_update_rate
            
            # Normalize container beliefs
            if np.sum(self.container_beliefs) > 0:
                self.container_beliefs = self.container_beliefs / np.sum(self.container_beliefs)
    
    def compute_action_for_goal(self, goal_pos):
        """
        Compute optimal action for a specific goal.
        
        Args:
            goal_pos: Position of the goal
            
        Returns:
            action_idx: Index of the action to take
        """
        # Get current gripper position
        gripper_pos = self.env._getGripper()[:2]
        
        # Calculate direction to goal
        delta_x = goal_pos[0] - gripper_pos[0]
        delta_y = goal_pos[1] - gripper_pos[1]
        
        # Calculate distance to goal
        distance = np.linalg.norm(np.array(goal_pos) - np.array(gripper_pos))
        
        # Simple policy:
        if distance < self.distance_threshold:
            # Close to goal, hold position (for grasping/releasing)
            return 0  # HOLD
        elif abs(delta_y) > abs(delta_x):
            # Y-axis movement is more important
            if delta_y > 0:
                return 0  # LEFT
            else:
                return 1  # RIGHT
        else:
            # X-axis movement is more important
            if delta_x > 0:
                return 3  # FORWARD
            else:
                return 4  # BACKWARD
    
    def select_action(self, user_input=None):
        """
        Select action based on belief over goals and user input.
        
        Args:
            user_input: User input ("left", "right", or "neutral"). 
                       If None, will generate synthetic input.
            
        Returns:
            action_idx: Index of the selected action
        """
        if not self.initialized:
            if not self.initialize_beliefs():
                return 0  # Default to HOLD if not initialized
                
        if user_input is None:
            user_input = self.generate_synthetic_user_input()
        
        # Update beliefs based on user input
        self.update_beliefs(user_input)
        
        # Compute action values based on current phase
        action_values = np.zeros(5)  # 5 actions: HOLD, LEFT, RIGHT, FORWARD, BACKWARD
        
        if self.env._gripperState == 'open':
            # Picking phase - weighted by object beliefs
            for i, obj_id in enumerate(self.env._objectUids):
                obj_pos, _ = self.env._p.getBasePositionAndOrientation(obj_id)
                obj_pos = np.array(obj_pos[:2])
                
                # Get optimal action for this object
                optimal_action = self.compute_action_for_goal(obj_pos)
                
                # Add weighted value to this action
                action_values[optimal_action] += self.object_beliefs[i]
        else:
            # Placing phase - weighted by container beliefs
            for i, cont_id in enumerate(self.env.container_uid):
                cont_pos, _ = self.env._p.getBasePositionAndOrientation(cont_id)
                cont_pos = np.array(cont_pos[:2])
                
                # Get optimal action for this container
                optimal_action = self.compute_action_for_goal(cont_pos)
                
                # Add weighted value to this action
                action_values[optimal_action] += self.container_beliefs[i]
        
        # Direct user input override with higher weight
        if user_input == "left":
            action_values[1] *= 2.0  # Boost LEFT action
        elif user_input == "right":
            action_values[2] *= 2.0  # Boost RIGHT action
        elif user_input == "neutral":
            # action_values[0] *= 1.5  # Mild boost to HOLD action
            action_values[3] *= 2.0  # Mild boost to FORWARD action
        
        # Select action with highest value
        return np.argmax(action_values)
    
    def reset(self):
        """Reset beliefs when environment resets."""
        self.initialized = False
        self.initialize_beliefs()
        
        # Clear previous inputs
        self.prev_inputs.clear()
        
    def debug_info(self):
        """Return debug information about current beliefs and goals."""
        if not self.initialized:
            return {"message": "Optimizer not initialized yet"}
            
        gripper_pos = self.env._getGripper()[:2]
        phase = "PICKUP" if self.env._gripperState == 'open' else "PLACE"
        
        debug_data = {
            "phase": phase,
            "gripper_position": gripper_pos.tolist(),
        }
        
        # Add target information based on phase
        if phase == "PICKUP":
            # Add object data
            target_pos = self.env._mugPos[:2]
            target_obj_id = self.env.intention_object
            target_idx = self.object_id_to_idx.get(target_obj_id, -1)
            
            # Calculate relative position using same logic as in generate_synthetic_user_input
            delta_y = self.env._mugPos[1] - gripper_pos[1]
            normalized_delta_y = 0 if (abs(delta_y) < 0.025) else np.sign(delta_y)
            
            # Get top 3 object beliefs
            object_beliefs = [(i, float(belief)) for i, belief in enumerate(self.object_beliefs)]
            top_beliefs = sorted(object_beliefs, key=lambda x: x[1], reverse=True)[:3]
            
            debug_data.update({
                "target_object_position": target_pos.tolist(),
                "target_object_idx": target_idx,
                "top_object_beliefs": top_beliefs,
                "delta_y": float(delta_y),
                "normalized_delta_y": float(normalized_delta_y)
            })
        else:
            # Add container data - Use the intended container (green one)
            target_pos = self.env._containerPos[:2]
            target_container_id = self.env.intention_container
            target_idx = self.container_id_to_idx.get(target_container_id, -1)
            
            # Calculate relative position using same logic as in generate_synthetic_user_input
            delta_y = self.env._containerPos[1] - gripper_pos[1]
            normalized_delta_y = 0 if (abs(delta_y) < 0.025) else np.sign(delta_y)
            
            # Get top 3 container beliefs
            container_beliefs = [(i, float(belief)) for i, belief in enumerate(self.container_beliefs)]
            top_beliefs = sorted(container_beliefs, key=lambda x: x[1], reverse=True)[:3]
            
            debug_data.update({
                "target_container_position": target_pos.tolist(),
                "target_container_idx": target_idx,
                "top_container_beliefs": top_beliefs,
                "delta_y": float(delta_y), 
                "normalized_delta_y": float(normalized_delta_y)
            })
            
        return debug_data
