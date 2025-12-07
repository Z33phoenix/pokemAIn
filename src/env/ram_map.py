"""
Utility helpers and offsets for interpreting Pokemon Red RAM.

This module is the single source of truth for every RAM lookup made by the
environment, rewards, and director logic. Always go through the helpers below
instead of scattering numeric addresses throughout the codebase.
"""
from typing import Tuple, Optional

# -----------------------------------------------------------------------------
# Player + Party Offsets
# -----------------------------------------------------------------------------
HP_CURRENT = 0xD16C  # 2 bytes
HP_MAX = 0xD16E  # 2 bytes
ENEMY_HP_CURRENT = 0xCFE6  # 2 bytes (wEnemyMonHP)
ENEMY_HP_MAX = 0xCFE8  # 2 bytes
PARTY_SIZE = 0xD163  # 1 byte
X_POS = 0xD362  # 1 byte
Y_POS = 0xD361  # 1 byte
MAP_N = 0xD35E  # 1 byte (map identifier)

# -----------------------------------------------------------------------------
# Battle / Status Offsets
# -----------------------------------------------------------------------------
BATTLE_TYPE = 0xD057  # 0=None, 1=Wild, 2=Trainer

# -----------------------------------------------------------------------------
# Menu / Cursor Offsets  (from DataCrystal Menu_Data)
# https://datacrystal.tcrf.net/wiki/Pokemon_Red_and_Blue/RAM_map#Menu_Data
# -----------------------------------------------------------------------------
MENU_CURSOR_Y = 0xCC24  # Y position of cursor for top menu item (id 0)
MENU_CURSOR_X = 0xCC25  # X position of cursor for top menu item (id 0)

MENU_SELECTED_ITEM_ID = 0xCC26  # Currently selected menu item (topmost is 0)
MENU_HIDDEN_TILE = 0xCC27       # Tile "hidden" by the menu cursor
MENU_LAST_ITEM_ID = 0xCC28      # ID of the last menu item
MENU_KEY_BITMASK = 0xCC29       # Bitmask applied to key port for current menu
MENU_PREV_SELECTED_ITEM_ID = 0xCC2A  # ID of previously selected menu item

MENU_LAST_CURSOR_PARTY = 0xCC2B      # Last cursor position on party / Bill's PC
MENU_LAST_CURSOR_ITEM = 0xCC2C       # Last cursor position on item screen
MENU_LAST_CURSOR_START_BATTLE = 0xCC2D  # Last cursor on START / battle menu

PARTY_ACTIVE_INDEX = 0xCC2F          # Index in party of Pokémon currently sent out

CURSOR_TILE_PTR = 0xCC30             # 2-byte pointer to cursor tile in C3A0 buffer
MENU_SELECT_HIGHLIGHT = 0xCC35       # Item highlighted with Select (01=first, 00=none, etc.)
MENU_FIRST_DISPLAYED_ITEM_ID = 0xCC36  # ID of first displayed menu item

# -----------------------------------------------------------------------------
# Experience / Progression
# -----------------------------------------------------------------------------
EXP_FIRST_MON = 0xD179  # 3 bytes (big endian)
BADGE_FLAGS = 0xD356  # bitfield for gym badges

# ----------------------------------------------------------------------------- #
# Event / Quest flags (subset used for walkthrough-guided goals)
# ----------------------------------------------------------------------------- #
FLAG_TOWN_MAP = 0xD5F3
FLAG_OAK_PARCEL = 0xD60D
FLAG_LAPRAS = 0xD72E
FLAG_GIOVANNI = 0xD751
FLAG_BROCK = 0xD755
FLAG_MISTY = 0xD75E
FLAG_LT_SURGE = 0xD773
FLAG_ERIKA = 0xD77C
FLAG_KOGA = 0xD792
FLAG_BLAINE = 0xD79A
FLAG_SABRINA = 0xD7B3
FLAG_SNORLAX_VERMILION = 0xD7D8
FLAG_SNORLAX_CELADON = 0xD7E0
FLAG_SS_ANNE = 0xD803
FLAG_MEWTWO_BIT = (0xD5C0, 1)  # tuple of (address, bit)

# Optional flags; uncomment/add as needed:
# FLAG_FLASH = ...
# FLAG_CUT = ...
# FLAG_SURF = ...
# FLAG_STRENGTH = ...
# FLAG_SILPH_SCOPE = ...
# FLAG_POKEFLUTE = ...
# FLAG_SAFFRON_OPEN = ...
# FLAG_SILPH_CLEARED = ...
# FLAG_SNORLAX_CLEARED = ...
# FLAG_MANSION_KEY = ...
# FLAG_CHAMPION = ...
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_u16(memory, address: int) -> int:
    """Reads an unsigned 16-bit value from RAM (big endian)."""
    return (memory[address] << 8) | memory[address + 1]


def read_player_hp(memory) -> Tuple[int, int]:
    """Return the player's current and max HP as a tuple of ints."""
    return _read_u16(memory, HP_CURRENT), _read_u16(memory, HP_MAX)


def read_enemy_hp(memory) -> Tuple[int, int]:
    """Return the opponent's current and max HP as a tuple of ints."""
    return _read_u16(memory, ENEMY_HP_CURRENT), _read_u16(memory, ENEMY_HP_MAX)


def read_hp_fraction(memory) -> float:
    """Return the player's HP fraction in [0, 1]."""
    hp_cur, hp_max = read_player_hp(memory)
    return (hp_cur / hp_max) if hp_max > 0 else 0.0


def read_player_position(memory) -> Tuple[int, int]:
    """Return the player's overworld tile coordinates (x, y)."""
    return memory[X_POS], memory[Y_POS]


def read_map_id(memory) -> int:
    """Return the current overworld map identifier."""
    return memory[MAP_N]


def read_party_size(memory) -> int:
    """Return the number of Pokemon in the player's party."""
    return memory[PARTY_SIZE]


def is_battle_active(memory) -> bool:
    """Return True when RAM reports the player is in battle."""
    return memory[BATTLE_TYPE] != 0


def read_first_mon_exp(memory) -> int:
    """Return the raw experience value for the first Pokemon slot."""
    return (
        (memory[EXP_FIRST_MON] << 16)
        | (memory[EXP_FIRST_MON + 1] << 8)
        | memory[EXP_FIRST_MON + 2]
    )


# ----------------------------------------------------------------------------- #
# Progression / flags helpers
# ----------------------------------------------------------------------------- #
def _flag(memory, addr: int, bit: int | None = None) -> bool:
    """Return the boolean value of a flag byte (or bit within the byte)."""
    val = memory[addr]
    if bit is None:
        return bool(val)
    return bool(val & (1 << bit))


def read_badge_flags(memory) -> int:
    """Return the raw badge bitfield byte."""
    return memory[BADGE_FLAGS]


def read_badge_count(memory) -> int:
    """Count the number of set bits in the badge flag byte."""
    flags = read_badge_flags(memory)
    return int(bin(flags).count("1"))


def read_flag_town_map(memory) -> bool:
    """Return True if the Town Map flag is set."""
    return _flag(memory, FLAG_TOWN_MAP)


def read_flag_oak_parcel(memory) -> bool:
    """Return True if Oak's Parcel quest flag is set."""
    return _flag(memory, FLAG_OAK_PARCEL)


def read_flag_lapras(memory) -> bool:
    """Return True if the Lapras event flag is set."""
    return _flag(memory, FLAG_LAPRAS)


def read_flag_giovanni(memory) -> bool:
    """Return True if the Giovanni encounter flag is set."""
    return _flag(memory, FLAG_GIOVANNI)


def read_flag_brock(memory) -> bool:
    """Return True if Brock has been defeated (Boulder Badge)."""
    return _flag(memory, FLAG_BROCK)


def read_flag_misty(memory) -> bool:
    """Return True if Misty has been defeated (Cascade Badge)."""
    return _flag(memory, FLAG_MISTY)


def read_flag_lt_surge(memory) -> bool:
    """Return True if Lt. Surge has been defeated (Thunder Badge)."""
    return _flag(memory, FLAG_LT_SURGE)


def read_flag_erika(memory) -> bool:
    """Return True if Erika has been defeated (Rainbow Badge)."""
    return _flag(memory, FLAG_ERIKA)


def read_flag_koga(memory) -> bool:
    """Return True if Koga has been defeated (Soul Badge)."""
    return _flag(memory, FLAG_KOGA)


def read_flag_blaine(memory) -> bool:
    """Return True if Blaine has been defeated (Volcano Badge)."""
    return _flag(memory, FLAG_BLAINE)


def read_flag_sabrina(memory) -> bool:
    """Return True if Sabrina has been defeated (Marsh Badge)."""
    return _flag(memory, FLAG_SABRINA)


def read_flag_snorlax_vermilion(memory) -> bool:
    """Return True if the Vermilion Snorlax blockade has been cleared."""
    return _flag(memory, FLAG_SNORLAX_VERMILION)


def read_flag_snorlax_celadon(memory) -> bool:
    """Return True if the Celadon Snorlax blockade has been cleared."""
    return _flag(memory, FLAG_SNORLAX_CELADON)


def read_flag_ss_anne(memory) -> bool:
    """Return True if the S.S. Anne quest flag is set."""
    return _flag(memory, FLAG_SS_ANNE)


def read_flag_mewtwo(memory) -> bool:
    """Return True if the Mewtwo-related bit is set."""
    addr, bit = FLAG_MEWTWO_BIT
    return _flag(memory, addr, bit=bit)


# -----------------------------------------------------------------------------
# Menu helpers
# -----------------------------------------------------------------------------
def read_menu_cursor(memory) -> Tuple[int, int]:
    """Return (row, col) menu cursor position for the top menu item."""
    return memory[MENU_CURSOR_Y], memory[MENU_CURSOR_X]


def read_current_menu_item(memory) -> int:
    """Return the currently selected menu item id (0 = topmost)."""
    return memory[MENU_SELECTED_ITEM_ID]


def read_last_menu_item(memory) -> int:
    """Return the id of the last menu item in the current menu."""
    return memory[MENU_LAST_ITEM_ID]


def read_party_active_index(memory) -> int:
    """Return the index in party of the Pokemon currently sent out."""
    return memory[PARTY_ACTIVE_INDEX]


def read_cursor_tile_pointer(memory) -> int:
    """Return the 16-bit pointer to the cursor tile in C3A0 buffer."""
    return _read_u16(memory, CURSOR_TILE_PTR)


def is_menu_open(memory) -> bool:
    """
    Heuristic: menus actively seize input by applying a key mask that differs
    from the default 0xFF pass-through value and expose a non-zero menu item
    count. Dialogue/text boxes leave the mask at 0xFF and report zero items.
    """
    key_mask = memory[MENU_KEY_BITMASK]
    last_item = memory[MENU_LAST_ITEM_ID]
    return (key_mask != 0xFF) and (last_item > 0)


def read_menu_target(memory) -> int:
    """Alias for the selected menu item, used by higher-level modules."""
    return read_current_menu_item(memory)


def read_menu_depth(memory) -> Optional[int]:
    """
    Depth is not exposed in the known offsets; return None to keep callers
    explicit about the absence of this signal.
    """
    return None
