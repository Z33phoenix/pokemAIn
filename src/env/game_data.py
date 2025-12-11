"""
DEPRECATED: This module has moved to src.games.gb.game_data

Game-specific data (maps, Pokemon, items) has been moved to support
hot-swapping between different Pokemon games.

For new code, import from the new location:
    from src.games.gb.game_data import (
        PokemonRedData,
        map_id_to_name,
        pokemon_id_to_name,
        item_id_to_name
    )

Or access through the environment (recommended):
    from src.core.env_factory import create_environment
    env = create_environment("pokemon_red", config)
    game_data = env.get_game_data()
    map_name = game_data.map_id_to_name(0x00)  # "Pallet Town"
"""

# Backward compatibility - re-export from new location
from src.games.gb.game_data import (
    PokemonRedData,
    MAP_ID_TO_NAME,
    MAP_NAME_TO_ID,
    POKEMON_ID_TO_NAME,
    POKEMON_NAME_TO_ID,
    ITEM_ID_TO_NAME,
    ITEM_NAME_TO_ID,
    map_id_to_name,
    map_name_to_id,
    pokemon_id_to_name,
    pokemon_name_to_id,
    item_id_to_name,
    item_name_to_id,
)

__all__ = [
    'PokemonRedData',
    'MAP_ID_TO_NAME',
    'MAP_NAME_TO_ID',
    'POKEMON_ID_TO_NAME',
    'POKEMON_NAME_TO_ID',
    'ITEM_ID_TO_NAME',
    'ITEM_NAME_TO_ID',
    'map_id_to_name',
    'map_name_to_id',
    'pokemon_id_to_name',
    'pokemon_name_to_id',
    'item_id_to_name',
    'item_name_to_id',
]
