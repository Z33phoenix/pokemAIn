import torch
import numpy as np
import os
from src.agent.director import CognitiveDirector
from src.agent.specialists.nav_brain import CrossQNavAgent
from src.agent.specialists.battle_brain import RainbowBattleAgent
# Assuming you will implement a MenuAgent later
# from src.agent.specialists.menu_brain import MenuAgent 

class HierarchicalAgent:
    """
    The Vessel.
    
    Structure:
    - Director (The Mind): Determines the current GOAL (Explore, Heal, Fight).
    - Specialists (The Limbs): Execute specific tasks based on the goal.
    """
    def __init__(self, action_dim=7, device="cpu", nav_lr=1e-4, battle_lr=1e-4):
        self.device = device
        
        # 1. The Mind (Logic-Based Director)
        # Note: The Director itself is now mostly logic, but it holds the Shared Vision Encoder
        self.director = CognitiveDirector().to(device)
        
        # 2. The Specialists (Neural Networks)
        restricted_action_dim = 6 
        
        self.nav_brain = CrossQNavAgent(
            input_dim=512, 
            action_dim=restricted_action_dim, 
            lr=nav_lr
        ).to(device)
        
        self.battle_brain = RainbowBattleAgent(
            input_dim=512, 
            action_dim=restricted_action_dim, 
            lr=battle_lr
        ).to(device)
        
        # self.menu_brain = MenuAgent(...) 
        
        self.active_specialist = "nav" # Default

    def get_action(self, obs, info, epsilon=0.1):
        """
        The Loop:
        1. Director thinks -> Decides Goal.
        2. Director activates specific Brain.
        3. Active Brain -> Outputs Action.
        """
        # Prepare Tensor (Batch Dimension for Vision)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # --- PHASE 1: COGNITION ---
        # The Director updates its internal Blackboard and decides what we WANT to do.
        # It returns the 'agent_idx' (0=Nav, 1=Battle, 2=Menu) and the shared visual features.
        agent_idx, features, current_goal = self.director(obs_t, info)
        
        # --- PHASE 2: ROUTING ---
        if agent_idx == 1:
            self.active_specialist = "battle"
            # Battle Brain needs to know if the goal is "catch" vs "kill" (optional future expansion)
            action = self.battle_brain.get_action(features)
            
        elif agent_idx == 2:
            self.active_specialist = "menu"
            # Placeholder for Menu Logic
            # action = self.menu_brain.get_action(features, goal=current_goal)
            action = 0 # No-op for now
            
        else:
            self.active_specialist = "nav"
            # If the Director wants to 'backtrack', we might need to override the Neural Net
            if current_goal == "backtrack_center":
                # Check if Graph has a path
                path_action = self.director.get_backtrack_action()
                if path_action is not None:
                    return path_action, agent_idx
            
            # Otherwise, use the Nav Neural Net
            action = self.nav_brain.get_action(features, epsilon)

        return action, agent_idx

    def save(self, checkpoint_dir="checkpoints", tag="latest"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Save the Vision Encoder (Inside Director)
        torch.save(self.director.state_dict(), os.path.join(checkpoint_dir, f"director_{tag}.pth"))
        torch.save(self.nav_brain.state_dict(), os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth"))
        torch.save(self.battle_brain.state_dict(), os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth"))

    def load(self, checkpoint_dir="checkpoints", tag="latest"):
        dir_path = os.path.join(checkpoint_dir, f"director_{tag}.pth")
        nav_path = os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth")
        bat_path = os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth")
        
        if os.path.exists(dir_path):
            self.director.load_state_dict(torch.load(dir_path, map_location=self.device, weights_only=True))
            self.nav_brain.load_state_dict(torch.load(nav_path, map_location=self.device, weights_only=True))
            self.battle_brain.load_state_dict(torch.load(bat_path, map_location=self.device, weights_only=True))
            print(f"[INFO] Loaded Agent ({tag})")
        else:
            print(f"[INFO] No checkpoints found for tag '{tag}'. Starting fresh.")