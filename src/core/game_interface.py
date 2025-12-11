"""
Abstract interfaces for game environments.

This module defines the Strategy Pattern for hot-swapping between different
Pokémon games (Red GB, Emerald GBA, etc.) without changing training code.

Architecture:
- GameEnvironment: Abstract gym interface (like gymnasium.Env)
- GameDataProvider: Abstract game-specific data (maps, Pokemon, items)
- MemoryInterface: Abstract memory reading (RAM/VRAM)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym


class GameDataProvider(ABC):
    """
    Abstract interface for game-specific data lookups.

    Each game (Red, Emerald, etc.) implements this to provide:
    - Map ID to name mappings
    - Pokemon ID to name mappings
    - Item ID to name mappings
    - Other game-specific metadata
    """

    @abstractmethod
    def map_id_to_name(self, map_id: int) -> str:
        """Convert map ID to human-readable name."""
        pass

    @abstractmethod
    def pokemon_id_to_name(self, pokemon_id: int) -> str:
        """Convert Pokemon ID to name."""
        pass

    @abstractmethod
    def item_id_to_name(self, item_id: int) -> str:
        """Convert item ID to name."""
        pass

    @abstractmethod
    def get_all_map_ids(self) -> List[int]:
        """Get list of all valid map IDs."""
        pass

    @abstractmethod
    def location_exists(self, map_id: int) -> bool:
        """Check if a map ID is valid."""
        pass

    @abstractmethod
    def get_game_name(self) -> str:
        """Get the game name (e.g., 'Pokemon Red', 'Pokemon Emerald')."""
        pass

    @abstractmethod
    def get_total_badges(self) -> int:
        """Get total number of badges in this game (8 for Red/Emerald)."""
        pass


class MemoryInterface(ABC):
    """
    Abstract interface for reading game memory.

    Provides high-level game state queries that work across different
    emulators (PyBoy, mGBA, etc.) and games.
    """

    @abstractmethod
    def read_u8(self, address: int) -> int:
        """Read unsigned 8-bit value at address."""
        pass

    @abstractmethod
    def read_u16_be(self, address: int) -> int:
        """Read unsigned 16-bit value (big-endian) at address."""
        pass

    @abstractmethod
    def read_u16_le(self, address: int) -> int:
        """Read unsigned 16-bit value (little-endian) at address."""
        pass

    @abstractmethod
    def read_bytes(self, address: int, length: int) -> bytes:
        """Read raw bytes starting at address."""
        pass

    # High-level game state queries (abstracted across games)

    @abstractmethod
    def get_player_position(self) -> Tuple[int, int, int]:
        """Get (map_id, x, y) position."""
        pass

    @abstractmethod
    def get_player_hp(self) -> Tuple[int, int]:
        """Get (current_hp, max_hp)."""
        pass

    @abstractmethod
    def get_enemy_hp(self) -> Tuple[int, int]:
        """Get (current_hp, max_hp) for enemy Pokemon."""
        pass

    @abstractmethod
    def get_party_size(self) -> int:
        """Get number of Pokemon in party."""
        pass

    @abstractmethod
    def get_badges(self) -> int:
        """Get number of badges earned."""
        pass

    @abstractmethod
    def is_battle_active(self) -> bool:
        """Check if currently in battle."""
        pass

    @abstractmethod
    def is_menu_open(self) -> bool:
        """Check if menu is open."""
        pass

    @abstractmethod
    def get_first_pokemon_experience(self) -> int:
        """Get experience points of first Pokemon in party."""
        pass

    @abstractmethod
    def get_party_power(self) -> float:
        """Get normalized party strength (0.0-1.0)."""
        pass


class GameEnvironment(ABC, gym.Env):
    """
    Abstract base class for Pokémon game environments.

    All games (Red GB, Emerald GBA, etc.) must implement this interface.
    This is the main hot-swap point - training code only depends on this.

    Follows Gymnasium interface with game-specific extensions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the game environment.

        Args:
            config: Game-specific configuration (rom_path, headless, etc.)
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            (observation, info) tuple
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: Integer action from action space

        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources (emulator, windows, etc.)."""
        pass

    @abstractmethod
    def save_state(self, path: str):
        """Save emulator state to disk."""
        pass

    @abstractmethod
    def load_state(self, path: str):
        """Load emulator state from disk."""
        pass

    # Game-specific methods that all games must implement

    @abstractmethod
    def get_memory_interface(self) -> MemoryInterface:
        """Get memory interface for reading game state."""
        pass

    @abstractmethod
    def get_game_data(self) -> GameDataProvider:
        """Get game data provider for lookups."""
        pass

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Get current observation (screen pixels)."""
        pass

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """
        Get current game state info.

        Must include these keys (standardized across all games):
        - map_id: int
        - x: int
        - y: int
        - badges: int
        - party_size: int
        - battle_active: bool
        - menu_open: bool
        - hp_current: int
        - hp_max: int
        - hp_percent: float
        - enemy_hp_current: int
        - enemy_hp_max: int
        - enemy_hp_percent: float
        - party_power: float

        Game-specific extensions allowed (map_connections, warps, sprites, etc.)
        """
        pass

    @abstractmethod
    def get_action_space_size(self) -> int:
        """Get number of possible actions."""
        pass

    @abstractmethod
    def get_action_names(self) -> List[str]:
        """Get human-readable action names."""
        pass


class GameProfile:
    """
    Container for game-specific configuration and metadata.

    Used by the environment factory to instantiate the correct game.
    """

    def __init__(
        self,
        game_id: str,
        game_name: str,
        platform: str,  # "gb", "gba", "nds", etc.
        emulator: str,  # "pyboy", "mgba", "melonds", etc.
        rom_extensions: List[str],
        env_class: type,  # The GameEnvironment subclass
        data_provider_class: type,  # The GameDataProvider subclass
    ):
        self.game_id = game_id
        self.game_name = game_name
        self.platform = platform
        self.emulator = emulator
        self.rom_extensions = rom_extensions
        self.env_class = env_class
        self.data_provider_class = data_provider_class

    def create_environment(self, config: Dict[str, Any]) -> GameEnvironment:
        """Create an instance of this game's environment."""
        return self.env_class(config)

    def create_data_provider(self) -> GameDataProvider:
        """Create an instance of this game's data provider."""
        return self.data_provider_class()
