"""
Utility helpers and offsets for interpreting Pokemon Red RAM.

This module is the single source of truth for every RAM lookup made by the
environment, rewards, and director logic. Always go through the helpers below
instead of scattering numeric addresses throughout the codebase.
"""
from typing import Tuple

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
# Poké Mart inventory block (DataCrystal "Pokémon Mart" section)
# -----------------------------------------------------------------------------
MART_TOTAL_ITEMS = 0xCF7B
MART_ITEMS_START = 0xCF7C
MART_MAX_ITEMS = 10

# -----------------------------------------------------------------------------
# Experience / Progression
# -----------------------------------------------------------------------------
EXP_FIRST_MON = 0xD179  # 3 bytes (big endian)

# -----------------------------------------------------------------------------
# Bag Items + Money (from DataCrystal Items / Money sections)
# https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Items
# -----------------------------------------------------------------------------
ITEMS_TOTAL = 0xD31D
ITEMS_START = 0xD31E  # Item 1 id; then quantity, then item 2 id, ... up to D345
ITEMS_END_MARKER = 0xD346

MONEY_BCD_1 = 0xD347  # Money Byte 1 (hundred-thousands / ten-thousands)
MONEY_BCD_2 = 0xD348  # Money Byte 2 (thousands / hundreds)
MONEY_BCD_3 = 0xD349  # Money Byte 3 (tens / ones)

# Known item IDs (Gen 1 internal ids) used for shaping in menu RL.
# These are kept small/specific so higher-level code doesn't depend on a full
# item table here.
ITEM_ID_POKE_BALL = 0x04
ITEM_ID_POTION = 0x14
ITEM_ID_NUGGET = 0x31

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_u16(memory, address: int) -> int:
    """Reads an unsigned 16-bit value from RAM (big endian)."""
    return (memory[address] << 8) | memory[address + 1]


def read_player_hp(memory) -> Tuple[int, int]:
    """Returns (current_hp, max_hp) as integers."""
    return _read_u16(memory, HP_CURRENT), _read_u16(memory, HP_MAX)


def read_enemy_hp(memory) -> Tuple[int, int]:
    """Returns (current_hp, max_hp) for the opposing Pokemon."""
    return _read_u16(memory, ENEMY_HP_CURRENT), _read_u16(memory, ENEMY_HP_MAX)


def read_hp_fraction(memory) -> float:
    """Returns the player's HP as a value in [0, 1]."""
    hp_cur, hp_max = read_player_hp(memory)
    return (hp_cur / hp_max) if hp_max > 0 else 0.0


def read_player_position(memory) -> Tuple[int, int]:
    """Returns (x, y) overworld tile coordinates."""
    return memory[X_POS], memory[Y_POS]


def read_map_id(memory) -> int:
    """Returns the current overworld map identifier."""
    return memory[MAP_N]


def read_party_size(memory) -> int:
    """Returns the number of Pokemon in the player's party."""
    return memory[PARTY_SIZE]


def is_battle_active(memory) -> bool:
    """Returns True when the player is currently in a battle."""
    return memory[BATTLE_TYPE] != 0


def read_first_mon_exp(memory) -> int:
    """Returns the raw experience value for the first Pokemon slot."""
    return (
        (memory[EXP_FIRST_MON] << 16)
        | (memory[EXP_FIRST_MON + 1] << 8)
        | memory[EXP_FIRST_MON + 2]
    )


def read_money(memory) -> int:
    """Return the player's current money as an integer.

    Money is stored as 3 bytes of packed BCD (D347-D349) representing a value
    in the range [0, 999999]. Each nibble is a decimal digit.
    """
    b1 = memory[MONEY_BCD_1]
    b2 = memory[MONEY_BCD_2]
    b3 = memory[MONEY_BCD_3]
    digits = [
        (b1 >> 4) & 0xF,
        b1 & 0xF,
        (b2 >> 4) & 0xF,
        b2 & 0xF,
        (b3 >> 4) & 0xF,
        b3 & 0xF,
    ]
    value = 0
    for d in digits:
        if d > 9:
            # Guard against invalid BCD; clamp digit.
            d = 9
        value = value * 10 + d
    return value


def read_bag_items(memory) -> list[tuple[int, int]]:
    """Return a list of (item_id, quantity) tuples from the player's bag.

    The bag layout is a simple flat list: total count at D31D, then repeatingn    (id, quantity) pairs until the end marker at D346.
    """
    total = memory[ITEMS_TOTAL]
    items: list[tuple[int, int]] = []
    addr = ITEMS_START
    for _ in range(total):
        if addr >= ITEMS_END_MARKER:
            break
        item_id = memory[addr]
        qty = memory[addr + 1]
        if item_id == 0xFF:
            break
        items.append((int(item_id), int(qty)))
        addr += 2
    return items


def count_item_in_bag(memory, item_id: int) -> int:
    """Return the total quantity of a specific item id in the bag."""
    total = 0
    for iid, qty in read_bag_items(memory):
        if iid == item_id:
            total += qty
    return total


# -----------------------------------------------------------------------------
# Menu helpers
# -----------------------------------------------------------------------------
def read_menu_cursor(memory) -> Tuple[int, int]:
    """Returns (row, col) menu cursor position for the top menu item."""
    return memory[MENU_CURSOR_Y], memory[MENU_CURSOR_X]


def read_current_menu_item(memory) -> int:
    """Returns the currently selected menu item id (0 = topmost)."""
    return memory[MENU_SELECTED_ITEM_ID]


def read_last_menu_item(memory) -> int:
    """Returns the id of the last menu item in the current menu."""
    return memory[MENU_LAST_ITEM_ID]


def read_party_active_index(memory) -> int:
    """Returns index in party of the Pokemon currently sent out."""
    return memory[PARTY_ACTIVE_INDEX]


def read_cursor_tile_pointer(memory) -> int:
    """Returns the 16-bit pointer to the cursor tile in C3A0 buffer."""
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


def is_mart_inventory_active(memory) -> bool:
    """
    Returns True when the Poké Mart inventory buffer (CF7B+) is populated.

    DataCrystal documents CF7B as the "Total Items" field for whatever mart
    script is currently active, followed by up to 10 item ids. Outside of a
    mart interaction this field is zeroed, so we can rely on it to detect
    whether the agent is shopping without hard-coding map ids.
    """
    total = memory[MART_TOTAL_ITEMS]
    if total == 0 or total == 0xFF:
        return False
    total = min(int(total), MART_MAX_ITEMS)
    first_item = memory[MART_ITEMS_START]
    if first_item in (0x00, 0xFF):
        return False
    # Ensure the buffer actually contains that many entries.
    for i in range(total):
        item = memory[MART_ITEMS_START + i]
        if item == 0xFF:
            return i > 0
    return True
