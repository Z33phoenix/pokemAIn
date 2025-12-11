"""
Pokemon Red (Game Boy) game implementation.

This package contains all Pokemon Red-specific code:
- PokemonRedGym: Main environment
- PokemonRedData: Map/Pokemon/Item name lookups
- PokemonRedMemory: RAM reading interface
- game_data: Static lookup tables
- ram_map: Memory offset definitions
- text_decoder: VRAM text decoding

Export the main entry point for hot-swapping.
"""

from src.games.gb.pokemon_red_gym import PokemonRedGym
from src.games.gb.game_data import PokemonRedData

__all__ = ['PokemonRedGym', 'PokemonRedData']
