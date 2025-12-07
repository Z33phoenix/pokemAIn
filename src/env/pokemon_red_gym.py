import io
import os
import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyboy
from pyboy.utils import WindowEvent

from src.env import ram_map


class PokemonRedGym(gym.Env):
    """
    Minimal Gymnasium wrapper around PyBoy configured purely via YAML.

    Observations are 84x84 grayscale frames, actions are discrete button presses,
    and info dictionaries expose a handful of RAM-derived signals for the
    Director and reward functions.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        window_backend = "headless" if config.get("headless", True) else "SDL2"
        self.pyboy = pyboy.PyBoy(
            config.get("rom_path", "pokemon_red.gb"),
            window=window_backend,
            sound_volume=0,
        )
        self.pyboy.set_emulation_speed(config.get("emulation_speed", 1))

        self.action_repeat = int(config.get("action_repeat", 24))
        self.release_frame = int(config.get("release_frame", 8))
        self.max_steps = int(config.get("max_steps", 2048))
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.RELEASE_BUTTON_SELECT,
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        self.memory = self.pyboy.memory
        self.state_buffers: List[Tuple[str, io.BytesIO]] = []
        self._state_queue: List[int] = []
        self._last_state_index: int | None = None

        default_state = os.path.join("states", "initial.state")
        requested_path = config.get("state_path", default_state)
        fallback_paths = [requested_path, default_state, "initial.state"]
        candidate_states = self._collect_state_files(fallback_paths)

        if candidate_states:
            self.state_buffers = self._load_state_buffers(candidate_states)
            initial_buffer = self._choose_reset_buffer()
            if initial_buffer is not None:
                self.pyboy.load_state(initial_buffer)
        else:
            # Burn a few frames so the BIOS screen passes.
            for _ in range(40):
                if not self.pyboy.tick():
                    break

        self.step_count = 0
        self.window_closed = False

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.window_closed:
            raise RuntimeError("Cannot reset environment: PyBoy window is closed.")
        self.step_count = 0
        reset_buffer = self._choose_reset_buffer()
        if reset_buffer is not None:
            self.pyboy.load_state(reset_buffer)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        if self.window_closed:
            raise RuntimeError("PyBoy window has been closed by the user.")

        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.action_repeat):
            if i == self.release_frame:
                self.pyboy.send_input(self.release_actions[action])
            if not self.pyboy.tick():
                self.window_closed = True
                info = {"window_closed": True}
                obs = self._get_obs()
                return obs, 0.0, True, True, info

        self.step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        raw_screen = self.pyboy.screen.image
        gray = raw_screen.convert("L")
        resized = gray.resize((84, 84))
        return np.array(resized, dtype=np.uint8)[None, ...]

    def _get_info(self) -> Dict[str, Any]:
        hp_current, hp_max = ram_map.read_player_hp(self.memory)
        enemy_hp_current, enemy_hp_max = ram_map.read_enemy_hp(self.memory)
        pos_x, pos_y = ram_map.read_player_position(self.memory)
        menu_last_index = ram_map.read_last_menu_item(self.memory)
        menu_has_options = menu_last_index > 0
        menu_open_flag = ram_map.is_menu_open(self.memory)
        menu_open = bool(menu_open_flag)
        menu_cursor = ram_map.read_menu_cursor(self.memory)
        menu_target = ram_map.read_menu_target(self.memory)
        menu_depth = ram_map.read_menu_depth(self.memory)
        return {
            "map_id": ram_map.read_map_id(self.memory),
            "x": pos_x,
            "y": pos_y,
            "battle_active": ram_map.is_battle_active(self.memory),
            "party_size": ram_map.read_party_size(self.memory),
            "hp_percent": (hp_current / hp_max) if hp_max > 0 else 0.0,
            "hp_current": hp_current,
            "hp_max": hp_max,
            "enemy_hp_percent": (enemy_hp_current / enemy_hp_max) if enemy_hp_max > 0 else 0.0,
            "enemy_hp_current": enemy_hp_current,
            "enemy_hp_max": enemy_hp_max,
            "menu_open": menu_open,
            "menu_cursor": menu_cursor,
            "menu_target": menu_target,
            "menu_depth": menu_depth,
            "menu_has_options": menu_has_options,
            "menu_last_index": menu_last_index,
        }

    def close(self):
        self.pyboy.stop()

    # ------------------------------------------------------------------ #
    # State management helpers
    # ------------------------------------------------------------------ #
    def _collect_state_files(self, paths: List[str]) -> List[str]:
        """Expand a list of files/directories into concrete .state file paths."""
        seen = []
        for path in paths:
            if not path:
                continue
            if os.path.isdir(path):
                for entry in os.listdir(path):
                    full_path = os.path.join(path, entry)
                    if os.path.isfile(full_path) and entry.lower().endswith(".state"):
                        seen.append(full_path)
            elif os.path.isfile(path):
                seen.append(path)
        # Preserve deterministic ordering while removing duplicates.
        deduped = []
        for p in seen:
            if p not in deduped:
                deduped.append(p)
        return deduped

    def _load_state_buffers(self, state_paths: List[str]) -> List[Tuple[str, io.BytesIO]]:
        buffers: List[Tuple[str, io.BytesIO]] = []
        for path in state_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                buffers.append((path, io.BytesIO(data)))
            except Exception:
                continue
        return buffers

    def _choose_reset_buffer(self) -> io.BytesIO | None:
        """Return a BytesIO buffer for a randomly chosen state (or None if unavailable)."""
        if not self.state_buffers:
            return None
        if not self._state_queue:
            self._state_queue = list(range(len(self.state_buffers)))
            random.shuffle(self._state_queue)
            if (
                self._last_state_index is not None
                and len(self._state_queue) > 1
                and self._state_queue[0] == self._last_state_index
            ):
                # Avoid immediately repeating the previous state when multiple options exist.
                self._state_queue.append(self._state_queue.pop(0))
        idx = self._state_queue.pop(0)
        self._last_state_index = idx
        _, buffer = self.state_buffers[idx]
        buffer.seek(0)
        return buffer
