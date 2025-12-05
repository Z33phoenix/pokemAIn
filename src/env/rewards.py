import numpy as np
import cv2
from src.env import ram_map

class RewardSystem:
    def __init__(self):
        self.total_xp = 0
        self.visited_maps = set()
        self.seen_coords = set()
        self.seen_screens = set() 
        self.last_hp_fraction = 1.0
        
        # Tracking
        self.last_coord = (0, 0, 0)
        self.steps_stagnant = 0

    def reset(self):
        self.total_xp = 0
        self.visited_maps = set()
        self.seen_coords = set()
        self.seen_screens = set()
        self.last_hp_fraction = 1.0
        self.last_coord = (0, 0, 0)
        self.steps_stagnant = 0

    def get_screen_hash(self, obs):
        if hasattr(obs, 'cpu'): obs = obs.cpu().numpy()
        if obs.ndim == 3: img = obs[0]
        else: img = obs
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        quantized = (small // 16) * 16
        return quantized.tobytes()

    def compute_reward(self, info, memory_bus, obs, action):
        reward = 0.0
        step_is_productive = False 
        
        current_screen_hash = self.get_screen_hash(obs)
        
        # --- 1. VISUAL NOVELTY ---
        if current_screen_hash not in self.seen_screens:
            self.seen_screens.add(current_screen_hash)
            step_is_productive = True 

        # --- 2. XP REWARD ---
        xp_h = memory_bus[ram_map.EXP_FIRST_MON]
        xp_m = memory_bus[ram_map.EXP_FIRST_MON + 1]
        xp_l = memory_bus[ram_map.EXP_FIRST_MON + 2]
        current_xp = (xp_h << 16) | (xp_m << 8) | xp_l
        
        if current_xp > self.total_xp:
            if self.total_xp > 0:
                diff = current_xp - self.total_xp
                reward += np.log(diff + 1.0) * 5.0
                step_is_productive = True 
            self.total_xp = current_xp

        # --- 3. EXPLORATION REWARD ---
        map_id = info["map_id"]
        x, y = info["x"], info["y"]
        current_coord = (map_id, x, y)
        
        if map_id not in self.visited_maps:
            reward += 10.0 
            self.visited_maps.add(map_id)
            self.seen_coords.clear()
            self.seen_screens.clear()
            step_is_productive = True
            
        if current_coord not in self.seen_coords:
            reward += 0.1
            self.seen_coords.add(current_coord)
            step_is_productive = True

        # --- 4. WALL BUMP PENALTY (NEW) ---
        # Action Indices: 0=Down, 1=Left, 2=Right, 3=Up
        feet_moved = (current_coord != self.last_coord)
        
        if not feet_moved and action < 4:
            # We tried to move (Action 0-3) but coordinates didn't change.
            # This is a Wall Bump.
            reward -= 0.1 
            
        # --- 5. STALEMATE PENALTY ---
        if feet_moved or step_is_productive:
            self.steps_stagnant = 0
            self.last_coord = current_coord
        else:
            self.steps_stagnant += 1
            
        if self.steps_stagnant > 50: 
            reward -= 0.1
        if self.steps_stagnant > 200: 
            reward -= 1.0 

        # --- 6. SURVIVAL ---
        hp_cur = (memory_bus[ram_map.HP_CURRENT] << 8) | memory_bus[ram_map.HP_CURRENT + 1]
        hp_max = (memory_bus[ram_map.HP_MAX] << 8) | memory_bus[ram_map.HP_MAX + 1]
        if hp_max > 0:
            current_hp_fraction = hp_cur / hp_max
            if current_hp_fraction < self.last_hp_fraction:
                reward -= (self.last_hp_fraction - current_hp_fraction) * 2.0
            self.last_hp_fraction = current_hp_fraction

        return reward