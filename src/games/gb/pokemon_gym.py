import io
import os
import random
from typing import Any, Dict, List, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyboy
from pyboy.utils import WindowEvent

from src.core.game_interface import GameEnvironment, MemoryInterface, GameDataProvider
from src.core.visual_processor import UnifiedVisualProcessor, create_unified_observation_space
from src.games.gb import ram_map
from src.games.gb.game_data import PokemonRedData
from src.games.gb.text_decoder import TextDecoder


class PokemonGBMemory(MemoryInterface):
    """
    Memory interface implementation for Pokemon Game Boy games (PyBoy).

    Provides abstracted memory access for Pokemon GB ROM's RAM structure.
    Works with Red, Blue, Yellow, Gold, Silver, Crystal, etc.
    """

    def __init__(self, pyboy_memory):
        self.memory = pyboy_memory

    def read_u8(self, address: int) -> int:
        """Read unsigned 8-bit value at address."""
        return self.memory[address]

    def read_u16_be(self, address: int) -> int:
        """Read unsigned 16-bit value (big-endian) at address."""
        return (self.memory[address] << 8) | self.memory[address + 1]

    def read_u16_le(self, address: int) -> int:
        """Read unsigned 16-bit value (little-endian) at address."""
        return self.memory[address] | (self.memory[address + 1] << 8)

    def read_bytes(self, address: int, length: int) -> bytes:
        """Read raw bytes starting at address."""
        return bytes(self.memory[address:address + length])

    # High-level game state queries

    def get_player_position(self) -> Tuple[int, int, int]:
        """Get (map_id, x, y) position."""
        map_id = ram_map.read_map_id(self.memory)
        x, y = ram_map.read_player_position(self.memory)
        return (map_id, x, y)

    def get_player_hp(self) -> Tuple[int, int]:
        """Get (current_hp, max_hp)."""
        return ram_map.read_player_hp(self.memory)

    def get_enemy_hp(self) -> Tuple[int, int]:
        """Get (current_hp, max_hp) for enemy Pokemon."""
        return ram_map.read_enemy_hp(self.memory)

    def get_party_size(self) -> int:
        """Get number of Pokemon in party."""
        return ram_map.read_party_size(self.memory)

    def get_badges(self) -> int:
        """Get number of badges earned."""
        return ram_map.read_badge_count(self.memory)

    def is_battle_active(self) -> bool:
        """Check if currently in battle."""
        return ram_map.is_battle_active(self.memory)

    def is_menu_open(self) -> bool:
        """Check if menu is open."""
        return ram_map.is_menu_open(self.memory)

    def get_first_pokemon_experience(self) -> int:
        """Get experience points of first Pokemon in party."""
        return ram_map.read_first_mon_exp(self.memory)

    def get_party_power(self) -> float:
        """Get normalized party strength (0.0-1.0)."""
        hp_current, hp_max = self.get_player_hp()
        return (hp_current / hp_max) if hp_max > 0 else 0.0


class PokemonGBGym(GameEnvironment):
    """
    Pokemon Game Boy environment implementation.

    Gymnasium wrapper around PyBoy configured purely via YAML.
    Implements the GameEnvironment interface for hot-swapping.
    Supports all Pokemon GB games: Red, Blue, Yellow, Gold, Silver, Crystal, etc.

    Observations are unified canvas 160x240 grayscale frames, actions are discrete button presses,
    and info dictionaries expose RAM-derived signals for the Director and reward functions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize PyBoy, load optional states, and prepare observation/action spaces."""
        super().__init__(config)
        self.config = config
        # PyBoy 2.x expects "null" instead of the deprecated "headless"/"dummy" window.
        window_backend = "null" if config.get("headless", True) else "SDL2"
        self.pyboy = pyboy.PyBoy(
            config.get("rom_path", "pokemon_red.gb"),
            window=window_backend,
            sound_volume=0,
            cgb=True,
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
        self.observation_space = create_unified_observation_space()
        self.memory = self.pyboy.memory

        # Create abstracted interfaces for hot-swapping
        self._memory_interface = PokemonGBMemory(self.memory)
        # Game data provider will be configured based on ROM
        self._game_data = self._create_game_data_provider()
        # Text decoder for cursor-based attention
        self.text_decoder = TextDecoder(self.pyboy)

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

    def _create_game_data_provider(self) -> GameDataProvider:
        """
        Create appropriate game data provider based on ROM.
        
        For now, defaults to PokemonRedData which works for Red/Blue/Yellow.
        Future versions can detect ROM type and return appropriate data provider.
        """
        # TODO: Add ROM detection logic to support other games (Gold/Silver/Crystal)
        # For now, PokemonRedData works for the original GB Pokemon games
        return PokemonRedData()

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
        """Capture the current unified canvas observation (160, 240, 1) from PyBoy."""
        raw_screen = self.pyboy.screen.image
        return UnifiedVisualProcessor.process_gb_screen(raw_screen)

    def _compute_party_power(self) -> float:
        """Heuristic party strength proxy based on current HP fraction."""
        # Simple proxy: use first mon HP fraction as power; extend later if needed.
        hp_current, hp_max = ram_map.read_player_hp(self.memory)
        return (hp_current / hp_max) if hp_max > 0 else 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Read relevant RAM offsets into a structured info dict."""
        # 1. Battle & Health Stats (Needed for Rewards)
        hp_current, hp_max = ram_map.read_player_hp(self.memory)
        enemy_hp_current, enemy_hp_max = ram_map.read_enemy_hp(self.memory)
        battle_active = ram_map.is_battle_active(self.memory)
        
        # 2. Navigation "Senses" (Needed for LLM & Director)
        current_map_id = ram_map.read_map_id(self.memory)
        pos_x, pos_y = ram_map.read_player_position(self.memory)
        
        # Optimization: Only read heavy map data if exploring
        if not battle_active:
            map_connections = ram_map.read_map_connections(self.memory)
            # Pass rom_bytes for warp parsing
            map_warps = ram_map.read_map_warps(self.pyboy, current_map_id, rom_bytes=self.rom_bytes)
            # This now returns enriched data (type: TRAINER/ITEM/NPC)
            sprites = ram_map.read_sprites(self.memory)
        else:
            map_connections, map_warps, sprites = {}, [], []

        return {
            # --- State for LLM & Director ---
            "map_id": current_map_id,
            "x": pos_x,
            "y": pos_y,
            "badges": ram_map.read_badge_count(self.memory), # Int
            "party_size": ram_map.read_party_size(self.memory),
            "map_connections": map_connections,
            "map_warps": map_warps,
            "sprites": sprites,
            
            # --- State for Reward System ---
            "battle_active": battle_active,
            "menu_open": ram_map.is_menu_open(self.memory),
            "hp_current": hp_current,
            "hp_max": hp_max,
            "hp_percent": (hp_current / hp_max) if hp_max > 0 else 1.0,
            "enemy_hp_current": enemy_hp_current,
            "enemy_hp_max": enemy_hp_max,
            "enemy_hp_percent": (enemy_hp_current / enemy_hp_max) if enemy_hp_max > 0 else 0.0,
            # Simple Party Power heuristic
            "party_power": (hp_current / hp_max) if hp_max > 0 else 0.0,
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

    # ------------------------------------------------------------------ #
    # GameEnvironment Interface Implementation
    # ------------------------------------------------------------------ #

    def get_memory_interface(self) -> MemoryInterface:
        """Get memory interface for reading game state."""
        return self._memory_interface

    def get_game_data(self) -> GameDataProvider:
        """Get game data provider for lookups."""
        return self._game_data

    def get_action_space_size(self) -> int:
        """Get number of possible actions."""
        return len(self.valid_actions)

    def get_action_names(self) -> List[str]:
        """Get human-readable action names."""
        return [
            "DOWN",
            "LEFT",
            "RIGHT",
            "UP",
            "A",
            "B",
            "START",
            "SELECT",
        ]
