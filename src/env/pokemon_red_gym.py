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

    Observations are 96x96 grayscale frames, actions are discrete button presses,
    and info dictionaries expose a handful of RAM-derived signals for the
    Director and reward functions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize PyBoy, load optional states, and prepare observation/action spaces."""
        super().__init__()
        self.config = config
        # PyBoy 2.x expects "null" instead of the deprecated "headless"/"dummy" window.
        window_backend = "null" if config.get("headless", True) else "SDL2"
        self.pyboy = pyboy.PyBoy(
            config.get("rom_path", "pokemon_red.gb"),
            window=window_backend,
            sound_volume=0,
        )
        # Stash ROM path/bytes for warp parsing fallbacks.
        self.rom_path = config.get("rom_path", "pokemon_red.gb")
        try:
            with open(self.rom_path, "rb") as f:
                self.rom_bytes = f.read()
            # Attach for downstream consumers that expect these attributes.
            try:
                setattr(self.pyboy, "_rom_bytes", self.rom_bytes)
                setattr(self.pyboy, "rom_path", self.rom_path)
            except Exception:
                pass
        except Exception:
            self.rom_bytes = None
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
            low=0, high=255, shape=(1, 96, 96), dtype=np.uint8
        )
        self.memory = self.pyboy.memory
        self.state_buffers: List[Tuple[str, io.BytesIO]] = []
        self._state_queue: List[int] = []
        self._last_state_index: int | None = None

        default_state = os.path.join("states", "initial.state")
        requested_path = config.get("state_path", default_state)
        fallback_paths = [requested_path, default_state, "initial.state"]
        candidate_states = self._collect_state_files(fallback_paths)
        self._prev_map_id: int | None = None

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
        """Reset the emulator to a queued state (or ROM boot) and return (obs, info)."""
        super().reset(seed=seed)
        if self.window_closed:
            raise RuntimeError("Cannot reset environment: PyBoy window is closed.")
        self.step_count = 0
        reset_buffer = self._choose_reset_buffer()
        if reset_buffer is not None:
            self.pyboy.load_state(reset_buffer)
        obs = self._get_obs()
        info = self._get_info()
        self._prev_map_id = info.get("map_id")
        return obs, info

    def step(self, action: int):
        """Execute an action, advance frames with repeats, and return the Gym step tuple."""
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
        current_map = info.get("map_id")
        if current_map is not None and current_map != self._prev_map_id:
            #print(f"[DEBUG][MAP] Map changed {self._prev_map_id} -> {current_map} | warps: {info.get('map_warps')}")
            self._prev_map_id = current_map

        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def save_state(self, path: str):
        """Persist current emulator state to the given path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, path: str):
        """Load emulator state from a given path."""
        with open(path, "rb") as f:
            self.pyboy.load_state(f)
        self.step_count = 0
        self._prev_map_id = ram_map.read_map_id(self.memory)

    def _get_obs(self):
        """Capture the current 96x96 grayscale observation from PyBoy."""
        raw_screen = self.pyboy.screen.image
        gray = raw_screen.convert("L")
        resized = gray.resize((96, 96))
        return np.array(resized, dtype=np.uint8)[None, ...]

    def _compute_party_power(self) -> float:
        """Heuristic party strength proxy based on current HP fraction."""
        # Simple proxy: use first mon HP fraction as power; extend later if needed.
        hp_current, hp_max = ram_map.read_player_hp(self.memory)
        return (hp_current / hp_max) if hp_max > 0 else 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Read relevant RAM offsets into a structured info dict."""
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
        badge_flags = ram_map.read_badge_flags(self.memory)
        badges = {
            "brock": ram_map.read_flag_brock(self.memory),
            "misty": ram_map.read_flag_misty(self.memory),
            "lt_surge": ram_map.read_flag_lt_surge(self.memory),
            "erika": ram_map.read_flag_erika(self.memory),
            "koga": ram_map.read_flag_koga(self.memory),
            "sabrina": ram_map.read_flag_sabrina(self.memory),
            "blaine": ram_map.read_flag_blaine(self.memory),
            "giovanni": ram_map.read_flag_giovanni(self.memory),
        }
        map_width = ram_map.read_map_width(self.memory)
        map_height = ram_map.read_map_height(self.memory)
        map_connection_flags = ram_map.read_map_connection_flags(self.memory)
        map_connections = ram_map.read_map_connections(self.memory)
        current_map_id = ram_map.read_map_id(self.memory)
        map_warps = ram_map.read_map_warps(self.pyboy, current_map_id, rom_bytes=self.rom_bytes)
        quest_flags = {
            "town_map": ram_map.read_flag_town_map(self.memory),
            "oak_parcel": ram_map.read_flag_oak_parcel(self.memory),
            "lapras": ram_map.read_flag_lapras(self.memory),
            "snorlax_vermilion": ram_map.read_flag_snorlax_vermilion(self.memory),
            "snorlax_celadon": ram_map.read_flag_snorlax_celadon(self.memory),
            "ss_anne": ram_map.read_flag_ss_anne(self.memory),
            "mewtwo": ram_map.read_flag_mewtwo(self.memory),
        }
        return {
            "map_id": current_map_id,
            "map_width": map_width,
            "map_height": map_height,
            "map_connection_flags": map_connection_flags,
            "map_connections": map_connections,
            "map_warps": map_warps,
            "x": pos_x,
            "y": pos_y,
            "battle_active": ram_map.is_battle_active(self.memory),
            "party_size": ram_map.read_party_size(self.memory),
            # Treat empty party as full HP to avoid spurious low-HP penalties.
            "hp_percent": (hp_current / hp_max) if hp_max > 0 else 1.0,
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
            "badge_count": ram_map.read_badge_count(self.memory),
            "badges": badges,
            "quest_flags": quest_flags,
            "party_power": self._compute_party_power(),
        }

    def close(self):
        """Stop the PyBoy emulator."""
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
        """Load .state files into in-memory buffers for fast resets."""
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
