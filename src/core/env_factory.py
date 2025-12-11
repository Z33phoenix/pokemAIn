"""
Environment factory for hot-swapping between different Pokemon games.

This module provides a factory function to create game environments based on
configuration. It allows seamless switching between Pokemon Red (GB),
Pokemon Emerald (GBA), and other games without changing training code.

Usage:
    config = {...}  # Game configuration
    env = create_environment("pokemon_red", config)

    # Access game data through the environment
    game_data = env.get_game_data()
    map_name = game_data.map_id_to_name(0x00)  # "Pallet Town"
"""

from typing import Dict, Any, Optional
from src.core.game_interface import GameEnvironment, GameProfile


# ============================================================================
# Game Registry
# ============================================================================

def _get_game_profiles() -> Dict[str, GameProfile]:
    """
    Get all registered game profiles.

    Returns:
        Dict mapping game IDs to GameProfile objects.
    """
    from src.games.gb.pokemon_red_gym import PokemonRedGym
    from src.games.gb.game_data import PokemonRedData

    profiles = {
        "pokemon_red": GameProfile(
            game_id="pokemon_red",
            game_name="Pokemon Red",
            platform="gb",
            emulator="pyboy",
            rom_extensions=[".gb"],
            env_class=PokemonRedGym,
            data_provider_class=PokemonRedData,
        ),
        # Future games will be added here:
        # "pokemon_emerald": GameProfile(
        #     game_id="pokemon_emerald",
        #     game_name="Pokemon Emerald",
        #     platform="gba",
        #     emulator="mgba",
        #     rom_extensions=[".gba"],
        #     env_class=PokemonEmeraldGym,
        #     data_provider_class=PokemonEmeraldData,
        # ),
    }
    return profiles


GAME_PROFILES = _get_game_profiles()


# ============================================================================
# Factory Functions
# ============================================================================

def create_environment(
    game_id: str,
    config: Dict[str, Any],
    auto_detect: bool = True
) -> GameEnvironment:
    """
    Create a game environment for the specified game.

    Args:
        game_id: Game identifier (e.g., "pokemon_red", "pokemon_emerald")
        config: Environment configuration (rom_path, headless, etc.)
        auto_detect: If True and game_id not found, try to auto-detect from ROM path

    Returns:
        GameEnvironment instance for the specified game

    Raises:
        ValueError: If game_id is unknown and auto-detect fails

    Examples:
        >>> config = {"rom_path": "pokemon_red.gb", "headless": True}
        >>> env = create_environment("pokemon_red", config)
        >>> game_data = env.get_game_data()
        >>> print(game_data.get_game_name())  # "Pokemon Red"
    """
    game_id = game_id.lower().strip()

    # Try exact match first
    if game_id in GAME_PROFILES:
        profile = GAME_PROFILES[game_id]
        return profile.create_environment(config)

    # Try auto-detection from ROM path if enabled
    if auto_detect and "rom_path" in config:
        detected_game_id = detect_game_from_rom(config["rom_path"])
        if detected_game_id and detected_game_id in GAME_PROFILES:
            profile = GAME_PROFILES[detected_game_id]
            return profile.create_environment(config)

    # If we get here, game is not found
    available_games = ", ".join(GAME_PROFILES.keys())
    raise ValueError(
        f"Unknown game ID: '{game_id}'. "
        f"Available games: {available_games}"
    )


def detect_game_from_rom(rom_path: str) -> Optional[str]:
    """
    Auto-detect game from ROM file extension.

    Args:
        rom_path: Path to ROM file

    Returns:
        Game ID string, or None if not detected

    Examples:
        >>> detect_game_from_rom("pokemon_red.gb")
        'pokemon_red'
        >>> detect_game_from_rom("pokemon_emerald.gba")
        'pokemon_emerald'
    """
    rom_path_lower = rom_path.lower()

    for game_id, profile in GAME_PROFILES.items():
        for ext in profile.rom_extensions:
            if rom_path_lower.endswith(ext):
                return game_id

    return None


def get_available_games() -> Dict[str, str]:
    """
    Get all available games.

    Returns:
        Dict mapping game IDs to game names

    Examples:
        >>> games = get_available_games()
        >>> print(games)
        {'pokemon_red': 'Pokemon Red', 'pokemon_emerald': 'Pokemon Emerald'}
    """
    return {
        game_id: profile.game_name
        for game_id, profile in GAME_PROFILES.items()
    }


def get_game_profile(game_id: str) -> Optional[GameProfile]:
    """
    Get the profile for a specific game.

    Args:
        game_id: Game identifier

    Returns:
        GameProfile object, or None if not found
    """
    return GAME_PROFILES.get(game_id.lower().strip())


# ============================================================================
# Convenience Functions
# ============================================================================

def create_game_data_provider(game_id: str):
    """
    Create a game data provider without creating the full environment.

    Useful for accessing game-specific data (map names, etc.) without
    needing to spin up an emulator.

    Args:
        game_id: Game identifier

    Returns:
        GameDataProvider instance

    Examples:
        >>> game_data = create_game_data_provider("pokemon_red")
        >>> print(game_data.map_id_to_name(0x00))  # "Pallet Town"
    """
    profile = get_game_profile(game_id)
    if profile is None:
        available_games = ", ".join(GAME_PROFILES.keys())
        raise ValueError(
            f"Unknown game ID: '{game_id}'. "
            f"Available games: {available_games}"
        )
    return profile.create_data_provider()
