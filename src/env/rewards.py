import numpy as np

class RewardSystem:
    def __init__(self):
        # --- Navigation Weights ---
        self.W_STEP_BASE   = -0.01   # Lower cost to encourage long-term exploration
        self.W_TILE_NEW    =  0.50   # Boosted to encourage mapping
        self.W_MAP_NEW     =  2.00   # Big event
        self.W_WALL_BUMP   = -0.10
        self.W_STALE       = -0.20   # Lower penalty so it doesn't panic-spin
        self.STALE_THRESH  =  120    # More patience

        # --- Gameplay Weights (New) ---
        self.W_CATCH_MON   =  5.0    # Huge reward for getting the starter
        self.W_LEVEL_UP    =  2.0    # Reward for grinding
        self.W_IN_BATTLE   =  0.05   # Small drip reward for STAYING in battle (don't run)
        self.W_WIN_BATTLE  =  3.0    # Reward for winning (this is harder to detect without RAM, but we try)

        # --- Tracking ---
        self.visited_maps = set()
        self.seen_coords = set()
        self.last_coord = (0, 0, 0)
        self.steps_stagnant = 0
        
        # State tracking for diffs
        self.max_party_size = 0
        self.total_levels = 0

    def reset(self):
        self.visited_maps.clear()
        self.seen_coords.clear()
        self.last_coord = (0, 0, 0)
        self.steps_stagnant = 0
        self.max_party_size = 0
        self.total_levels = 0

    def compute_reward(self, info, memory_bus, obs, action):
        reward = 0.0
        
        # 1. Navigation Logic
        # ------------------------------------------
        map_id = info.get("map_id", 0)
        x, y = info.get("x", 0), info.get("y", 0)
        current_coord = (map_id, x, y)

        # Exploration Rewards
        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            reward += self.W_MAP_NEW
        
        if current_coord not in self.seen_coords:
            self.seen_coords.add(current_coord)
            reward += self.W_TILE_NEW

        # Movement / Stagnation penalties
        feet_moved = (current_coord != self.last_coord)
        if not feet_moved and action < 4: 
            reward += self.W_WALL_BUMP
            self.steps_stagnant += 1
            if self.steps_stagnant > self.STALE_THRESH:
                reward += self.W_STALE
        else:
            self.steps_stagnant = 0
            self.last_coord = current_coord
        
        # Apply Base Step Cost
        reward += self.W_STEP_BASE

        # 2. Gameplay Logic (The Fix)
        # ------------------------------------------
        # We need to extract these from 'info' or 'memory_bus'
        # If your environment provides them directly:
        party_size = info.get("party_size", 0) 
        # Sum of all pokemon levels in party (proxy for XP gain)
        current_levels = sum(info.get("levels", [0])) 
        
        # Reward: Caught a Pokemon (Starter included)
        if party_size > self.max_party_size:
            diff = party_size - self.max_party_size
            reward += (self.W_CATCH_MON * diff)
            self.max_party_size = party_size
        
        # Reward: Leveled Up (Encourages fighting Pidgeys)
        if current_levels > self.total_levels:
            # First frame might initialize at 5, ignore the jump from 0 to 5
            if self.total_levels > 0:
                reward += self.W_LEVEL_UP
            self.total_levels = current_levels

        # Reward: In Battle (Drip feed)
        # Check if the address implies battle (game specific) 
        # Or if the wrapper provides a boolean
        if info.get("in_battle", False): 
            reward += self.W_IN_BATTLE

        # ------------------------------------------

        return float(np.clip(reward, -5.0, 5.0))