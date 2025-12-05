import numpy as np
from src.env import ram_map

class RewardSystem:
    def __init__(self):
        self.total_xp = 0
        self.visited_maps = set()
        self.seen_coords = set()
        self.last_hp_fraction = 1.0
        
    def reset(self):
        self.total_xp = 0
        self.visited_maps = set()
        self.seen_coords = set()
        self.last_hp_fraction = 1.0

    def compute_reward(self, info, memory_bus):
        """
        Calculates the reward based on RAM state.
        Args:
            info (dict): The info dict from the environment step.
            memory_bus: The PyBoy memory object to read direct addresses.
        """
        reward = 0.0
        
        # --- 1. XP REWARD (Battle Specialist Goal) ---
        # Read 3 bytes for Experience (Big Endian)
        # Note: This reads XP for the first Pokémon in the party only.
        xp_h = memory_bus[ram_map.EXP_FIRST_MON]
        xp_m = memory_bus[ram_map.EXP_FIRST_MON + 1]
        xp_l = memory_bus[ram_map.EXP_FIRST_MON + 2]
        current_xp = (xp_h << 16) | (xp_m << 8) | xp_l
        
        if current_xp > self.total_xp:
            if self.total_xp > 0: # Skip the initial jump from 0
                diff = current_xp - self.total_xp
                # Logarithmic scaling prevents massive rewards later in game
                reward += np.log(diff + 1.0) * 5.0
            self.total_xp = current_xp

        # --- 2. EXPLORATION REWARD (Navigation Specialist Goal) ---
        # Map Discovery
        map_id = info["map_id"]
        if map_id not in self.visited_maps:
            reward += 10.0 # Big reward for finding a new town/route
            self.visited_maps.add(map_id)
            self.seen_coords.clear() # Reset coordinate memory on new map
            
        # Coordinate Discovery (Micro-exploration)
        # Encourages moving around the current map rather than standing still
        coord_key = (map_id, info["x"], info["y"])
        if coord_key not in self.seen_coords:
            reward += 0.01 # Breadcrumb reward
            self.seen_coords.add(coord_key)
            
        # --- 3. SURVIVAL PENALTY ---
        # If HP drops, penalize
        # Note: We need max HP to calculate fraction
        hp_cur = (memory_bus[ram_map.HP_CURRENT] << 8) | memory_bus[ram_map.HP_CURRENT + 1]
        hp_max = (memory_bus[ram_map.HP_MAX] << 8) | memory_bus[ram_map.HP_MAX + 1]
        
        if hp_max > 0:
            current_hp_fraction = hp_cur / hp_max
            if current_hp_fraction < self.last_hp_fraction:
                # Penalize damage taken
                reward -= (self.last_hp_fraction - current_hp_fraction) * 2.0
            self.last_hp_fraction = current_hp_fraction

        return reward