import io
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyboy
from pyboy.utils import WindowEvent

from src.env import ram_map

class PokemonRedGym(gym.Env):
    """
    Gym Wrapper for Pokémon Red.
    - Observation: 84x84 Grayscale Image (Visual).
    - Action: Discrete(8) (Buttons).
    - Info: RAM data (for Director/Graph only).
    """
    
    def __init__(self, rom_path='pokemon_red.gb', state_path='initial.state', headless=True, emulation_speed=0, sim_scale=3, max_steps=2048):
        super().__init__()
        
        window_backend = "headless" if headless else "SDL2"
        self.pyboy = pyboy.PyBoy(rom_path, window=window_backend, sound_volume=0)
        self.pyboy.set_emulation_speed(emulation_speed)
        
        # ... (Action/Release lists remain the same) ...
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_START
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_START
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.memory = self.pyboy.memory
        
        # State Loading
        self.state_path = state_path
        if os.path.exists(self.state_path):
            print(f"Loading initial state from {self.state_path}...")
            with open(self.state_path, "rb") as f:
                self.pyboy.load_state(f)
        else:
            print("No state file found. Starting fresh.")
            for _ in range(40): self.pyboy.tick()
        
        self.reset_state_buffer = io.BytesIO()
        self.pyboy.save_state(self.reset_state_buffer)
        
        # EPISODE MANAGEMENT
        self.step_count = 0
        self.max_steps = max_steps # Now configurable!

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # RELOAD the initial state
        self.reset_state_buffer.seek(0)
        self.pyboy.load_state(self.reset_state_buffer)
            
        return self._get_obs(), self._get_info()

    def step(self, action):
        # 1. Perform Action
        self.pyboy.send_input(self.valid_actions[action])
        
        # Run 24 frames (approx 0.4s real-time)
        for i in range(24):
            if i == 8: # Release button tap
                self.pyboy.send_input(self.release_actions[action])
            self.pyboy.tick()

        # 2. Get Data
        obs = self._get_obs()
        info = self._get_info()
        
        # 3. Calculate Reward (Placeholder until rewards.py is integrated)
        reward = 0 
        
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        raw_screen = self.pyboy.screen.image 
        gray = raw_screen.convert("L") 
        resized = gray.resize((84, 84))
        return np.array(resized, dtype=np.uint8)[None, ...]

    def _get_info(self):
        return {
            "map_id": self.memory[ram_map.MAP_N],
            "x": self.memory[ram_map.X_POS],
            "y": self.memory[ram_map.Y_POS],
            "battle_active": self.memory[ram_map.BATTLE_TYPE] != 0,
            "party_size": self.memory[ram_map.PARTY_SIZE]
        }

    def render(self):
        pass
        
    def close(self):
        self.pyboy.stop()