"""
Pokemon Game Boy game implementation.

This package contains all Pokemon GB-specific code:
- PokemonGBGym: Main environment (supports Red/Blue/Yellow/Gold/Silver/Crystal)
- PokemonRedData: Map/Pokemon/Item name lookups (works for original GB games)
- PokemonGBMemory: RAM reading interface
- game_data: Static lookup tables
- ram_map: Memory offset definitions
- text_decoder: VRAM text decoding

Export the main entry point for hot-swapping.
"""

from src.games.gb.pokemon_gym import PokemonGBGym, PokemonGBMemory
from src.games.gb.game_data import PokemonRedData

# For backwards compatibility, provide the old name
PokemonRedGym = PokemonGBGym
PokemonRedMemory = PokemonGBMemory

__all__ = ['PokemonGBGym', 'PokemonGBMemory', 'PokemonRedData', 'PokemonRedGym', 'PokemonRedMemory']
