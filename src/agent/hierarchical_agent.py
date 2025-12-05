import torch
import numpy as np
import os
from src.agent.director import Director
from src.agent.specialists.nav_brain import CrossQNavAgent
from src.agent.specialists.battle_brain import RainbowBattleAgent

class HierarchicalAgent:
    """
    The Master Class.
    
    Structure:
    - Shared Vision (Encoder) lives inside the Director.
    - Director: Decides WHICH brain is active.
    - Specialists: Decide WHAT button to press.
    """
    def __init__(self, action_dim=7, device="cpu", nav_lr=1e-4, battle_lr=1e-4):
        self.device = device
        
        # 1. The Boss (Director)
        self.director = Director(action_dim=2).to(device) 
        
        # 2. The Specialists
        restricted_action_dim = 6 
        
        # Pass the config LRs to the brains
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
        
        self.current_specialist_idx = 0

    def get_action(self, obs, info, epsilon=0.1):
        """
        The Main Loop:
        1. Director observes -> Extracts Features -> Selects Specialist.
        2. Selected Specialist -> Uses Features -> Selects Action.
        """
        # Prepare Tensor
        # FIXED: Added .unsqueeze(0) to create Batch Dimension (1, 1, 84, 84)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # A. Director Step
        # Returns: specialist_index (0/1), features (512-dim), force_action (from Graph)
        specialist_idx, features, force_action = self.director.select_specialist(obs_t, info, epsilon)
        
        # Force NAV only for pre-training
        specialist_idx = 0
        self.current_specialist_idx = specialist_idx
        
        # B. Check for Graph Override (Backtracking)
        if force_action is not None:
            return force_action, specialist_idx
            
        # C. Specialist Step
        if specialist_idx == 0:
            # Navigation Brain (CrossQ)
            action = self.nav_brain.get_action(features, epsilon)
        else:
            # Battle Brain (Rainbow)
            action = self.battle_brain.get_action(features)
            
        return action, specialist_idx

    def save(self, checkpoint_dir="checkpoints", tag="latest"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        torch.save(self.director.state_dict(), os.path.join(checkpoint_dir, f"director_{tag}.pth"))
        torch.save(self.nav_brain.state_dict(), os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth"))
        torch.save(self.battle_brain.state_dict(), os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth"))

    def load(self, checkpoint_dir="checkpoints", tag="latest"):
        # Construct paths with the tag (e.g., "director_latest.pth")
        dir_path = os.path.join(checkpoint_dir, f"director_{tag}.pth")
        nav_path = os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth")
        bat_path = os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth")
        
        if os.path.exists(dir_path):
            # FIXED: Added weights_only=True to suppress security warnings
            self.director.load_state_dict(torch.load(dir_path, map_location=self.device, weights_only=True))
            self.nav_brain.load_state_dict(torch.load(nav_path, map_location=self.device, weights_only=True))
            self.battle_brain.load_state_dict(torch.load(bat_path, map_location=self.device, weights_only=True))
            print(f"[INFO] Loaded Agent ({tag}) from {checkpoint_dir}")
        else:
            print(f"[INFO] No checkpoints found for tag '{tag}'. Starting fresh.")