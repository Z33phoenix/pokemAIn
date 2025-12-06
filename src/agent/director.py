import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.vision.encoder import NatureCNN

@dataclass
class Blackboard:
    """
    The 'Consciousness' of the AI. 
    This persists across all agents and holds the truth of the world.
    Populated by a Vision/OCR wrapper (which adheres to your visual-only constraint).
    """
    # Self State
    party_avg_hp_percent: float = 1.0
    lead_pokemon_hp_percent: float = 1.0
    is_poisoned: bool = False
    in_battle: bool = False
    in_menu: bool = False
    
    # World State
    current_map_id: int = 0
    last_town_map_id: int = 0
    items: Dict[str, int] = field(default_factory=dict) # e.g., {'potion': 2}
    
    # Metrics
    steps_since_last_heal: int = 0
    battle_win_rate_last_10: float = 0.5

class DriveSystem:
    """
    Calculates the 'Urge' to do something based on the Blackboard.
    Returns a score between 0.0 and 1.0.
    """
    @staticmethod
    def get_survival_score(bb: Blackboard) -> float:
        # Urge increases exponentially as HP drops
        # Formula: (1 - hp)^2. If HP is 0.1, score is 0.81. If HP is 0.9, score is 0.01
        score = (1.0 - bb.lead_pokemon_hp_percent) ** 2
        if bb.is_poisoned:
            score += 0.5 # Panic boost
        return min(score, 1.0)

    @staticmethod
    def get_grind_score(bb: Blackboard) -> float:
        # If we are losing battles, we need to grind.
        return 1.0 - bb.battle_win_rate_last_10

    @staticmethod
    def get_explore_score(bb: Blackboard) -> float:
        # Default urge to move forward
        return 0.3

class CognitiveDirector(nn.Module):
    def __init__(self):
        super().__init__()
        # The Shared Eyes
        self.vision = NatureCNN()
        
        # The Logic Core
        self.blackboard = Blackboard()
        self.current_goal = "explore"
        
        # Graph Memory for backtracking
        self.graph_path = []

    def get_backtrack_action(self):
        """Returns the next action if we are in a hard-coded pathing mode"""
        if self.graph_path:
            return self.graph_path.pop(0)
        return None

    def forward(self, x, info):
        """
        Args:
            x: Tensor image (Batch, C, H, W)
            info: Dict containing metadata (OCR results, etc)
        """
        # 1. Vision Pass (Always happens)
        features = self.vision(x)
        
        # 2. Update Knowledge
        self.update_perception(info)
        
        # 3. Logic Step
        agent_idx = self.think()
        
        return agent_idx, features, self.current_goal

    def update_perception(self, info):
        # ... logic to update Blackboard from info dict ...
        pass

    def think(self):
        # ... logic to return 0, 1, or 2 based on DriveSystem ...
        # (See previous response for the DriveSystem logic)
        bb = self.blackboard
        
        # Example Logic:
        if bb.in_battle: return 1
        if bb.in_menu: return 2
        return 0