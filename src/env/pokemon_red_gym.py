"""
DEPRECATED: This module has moved to src.games.gb.pokemon_red_gym

The Pokemon Red gym environment has been moved to support hot-swapping
between different Pokemon games (Red GB, Emerald GBA, etc.).

For new code, import from the new location:
    from src.games.gb.pokemon_red_gym import PokemonRedGym

Or use the environment factory (recommended):
    from src.core.env_factory import create_environment
    env = create_environment("pokemon_red", config)
"""

# Backward compatibility - re-export from new location
from src.games.gb.pokemon_red_gym import PokemonRedGym, PokemonRedMemory

__all__ = ['PokemonRedGym', 'PokemonRedMemory']
