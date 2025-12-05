import numpy as np
from src.env import ram_map  # still imported, but we won't use it right now

class RewardSystem:
    """
    Extremely simple, nav-focused reward.

    - Encourages moving to new coordinates.
    - Big bonus for entering a new map.
    - Small step cost every frame.
    - Penalty for wall bumps and long stagnation.
    """

    def __init__(self):
        # weights
        self.W_STEP_BASE   = -0.02   # was -0.05
        self.W_TILE_NEW    =  0.30   # was 0.20
        self.W_MAP_NEW     =  1.00   # same
        self.W_WALL_BUMP   = -0.10   # was -0.20
        self.W_STALE       = -0.30   # was -0.50
        self.STALE_THRESH  = 80      # same

        # tracking
        self.visited_maps = set()
        self.seen_coords = set()
        self.last_coord = (0, 0, 0)
        self.steps_stagnant = 0

    def reset(self):
        self.visited_maps.clear()
        self.seen_coords.clear()
        self.last_coord = (0, 0, 0)
        self.steps_stagnant = 0

    def compute_reward(self, info, memory_bus, obs, action):
        # base step cost
        reward = self.W_STEP_BASE

        # position info from env
        map_id = info["map_id"]
        x, y = info["x"], info["y"]
        current_coord = (map_id, x, y)

        # map novelty
        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            reward += self.W_MAP_NEW

        # coordinate novelty
        if current_coord not in self.seen_coords:
            self.seen_coords.add(current_coord)
            reward += self.W_TILE_NEW

        # movement / wall bump
        feet_moved = (current_coord != self.last_coord)
        if not feet_moved and action < 4:  # 0–3 are movement actions
            reward += self.W_WALL_BUMP

        # stagnation
        if feet_moved:
            self.steps_stagnant = 0
            self.last_coord = current_coord
        else:
            self.steps_stagnant += 1
            if self.steps_stagnant > self.STALE_THRESH:
                reward += self.W_STALE

        # keep values sane for CrossQ
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward
