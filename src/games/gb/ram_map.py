"""
Utility helpers and offsets for interpreting Pokemon Red RAM.
This module is the single source of truth for every RAM lookup made by the
environment, rewards, and director logic.
"""
from typing import Tuple, Optional, List, Dict, Any

# -----------------------------------------------------------------------------#
# Player + Party Offsets
# -----------------------------------------------------------------------------#
HP_CURRENT = 0xD16C  # 2 bytes
HP_MAX = 0xD16E  # 2 bytes
ENEMY_HP_CURRENT = 0xCFE6  # 2 bytes (CFE6-CFE7)
ENEMY_HP_MAX = 0xCFF4  # 2 bytes (CFF4-CFF5) - FIXED from 0xCFE8
PARTY_SIZE = 0xD163  # 1 byte
X_POS = 0xD362  # 1 byte
Y_POS = 0xD361  # 1 byte
MAP_N = 0xD35E  # 1 byte (map identifier)
PARTY_LEVEL_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# -----------------------------------------------------------------------------#
# Battle / Status Offsets
# -----------------------------------------------------------------------------#
BATTLE_TYPE = 0xD057  # 0=None, 1=Wild, 2=Trainer

# -----------------------------------------------------------------------------#
# Map Header Offsets
# -----------------------------------------------------------------------------#
MAP_TILESET = 0xD367
MAP_HEIGHT = 0xD368
MAP_WIDTH = 0xD369
MAP_CONNECTION_FLAGS = 0xD370
MAP_CONN1_START = 0xD371
MAP_CONN2_START = 0xD37C
MAP_CONN3_START = 0xD387
MAP_CONN4_START = 0xD392
MAP_CONN_BLOCK_SIZE = 0x0B

MAP_HEADER_PTR_TABLE = 0x01AE
MAP_HEADER_BANK_TABLE = 0xC23D

# -----------------------------------------------------------------------------#
# Menu / Cursor Offsets
# -----------------------------------------------------------------------------#
MENU_CURSOR_Y = 0xCC24
MENU_CURSOR_X = 0xCC25
MENU_SELECTED_ITEM_ID = 0xCC26
MENU_LAST_ITEM_ID = 0xCC28
MENU_KEY_BITMASK = 0xCC29
PARTY_ACTIVE_INDEX = 0xCC2F
CURSOR_TILE_PTR = 0xCC30

# -----------------------------------------------------------------------------#
# Experience / Progression
# -----------------------------------------------------------------------------#
EXP_FIRST_MON = 0xD179
BADGE_FLAGS = 0xD356

# -----------------------------------------------------------------------------#
# Sprite Data
# -----------------------------------------------------------------------------#
# wSpriteStateData1: 0xC100. 16 slots of 0x10 bytes each.
SPRITE_DATA_START = 0xC100
SPRITE_COUNT = 16
SPRITE_SIZE = 0x10

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _read_u16(memory, address: int) -> int:
    return (memory[address] << 8) | memory[address + 1]

def _read_rom_byte(rom: bytes, bank: int, addr: int) -> int:
    if bank <= 0: absolute = addr
    elif addr < 0x4000: absolute = addr
    else: absolute = (bank * 0x4000) + (addr - 0x4000)
    
    if absolute < 0 or absolute >= len(rom): return 0
    return rom[absolute]

def _read_rom_u16(rom: bytes, bank: int, addr: int) -> int:
    lo = _read_rom_byte(rom, bank, addr)
    hi = _read_rom_byte(rom, bank, addr + 1)
    return lo | (hi << 8)

def read_player_hp(memory) -> Tuple[int, int]:
    return _read_u16(memory, HP_CURRENT), _read_u16(memory, HP_MAX)

def read_enemy_hp(memory) -> Tuple[int, int]:
    current_hp = _read_u16(memory, ENEMY_HP_CURRENT)
    max_hp = _read_u16(memory, ENEMY_HP_MAX)
    return current_hp, max_hp

def read_player_position(memory) -> Tuple[int, int]:
    return memory[X_POS], memory[Y_POS]

def read_map_id(memory) -> int:
    return memory[MAP_N]

def read_map_width(memory) -> int:
    return memory[MAP_WIDTH]

def read_map_height(memory) -> int:
    return memory[MAP_HEIGHT]

def read_map_tileset(memory) -> int:
    return memory[MAP_TILESET]

def read_badge_count(memory) -> int:
    flags = memory[BADGE_FLAGS]
    return int(bin(flags).count("1"))

def read_party_size(memory) -> int:
    return memory[PARTY_SIZE]

def is_battle_active(memory) -> bool:
    return memory[BATTLE_TYPE] != 0

def is_menu_open(memory) -> bool:
    key_mask = memory[MENU_KEY_BITMASK]
    last_item = memory[MENU_LAST_ITEM_ID]
    return (key_mask != 0xFF) and (last_item > 0)

def read_first_mon_exp(memory) -> int:
    return (
        (memory[EXP_FIRST_MON] << 16)
        | (memory[EXP_FIRST_MON + 1] << 8)
        | memory[EXP_FIRST_MON + 2]
    )

def read_party_levels(memory) -> List[int]:
    size = read_party_size(memory)
    levels = []
    for idx in range(min(size, len(PARTY_LEVEL_ADDRESSES))):
        levels.append(int(memory[PARTY_LEVEL_ADDRESSES[idx]]))
    return levels

# --- Map & Connections ---

def read_map_connections(memory) -> dict[str, dict[str, object]]:
    flags = memory[MAP_CONNECTION_FLAGS]
    dirs = {
        "north": bool(flags & 0x01), "south": bool(flags & 0x02),
        "west": bool(flags & 0x04), "east": bool(flags & 0x08)
    }
    
    def _parse(start_addr):
        return {"dest_map": memory[start_addr], "raw": [int(memory[i]) for i in range(start_addr, start_addr+11)]}

    return {
        "north": {"exists": dirs["north"], **_parse(MAP_CONN1_START)},
        "south": {"exists": dirs["south"], **_parse(MAP_CONN2_START)},
        "west":  {"exists": dirs["west"],  **_parse(MAP_CONN3_START)},
        "east":  {"exists": dirs["east"],  **_parse(MAP_CONN4_START)},
    }

def read_map_warps(pyboy, map_id: int, rom_bytes: bytes | None = None) -> list[dict[str, int]]:
    rom = rom_bytes
    if rom is None: return []
    
    header_ptr = _read_rom_u16(rom, 0, MAP_HEADER_PTR_TABLE + (2 * map_id))
    header_bank = _read_rom_byte(rom, 0, MAP_HEADER_BANK_TABLE + map_id)
    if header_ptr == 0: return []
    
    offset = header_ptr
    conn_flags = _read_rom_byte(rom, header_bank, offset + 9)
    conn_count = sum(1 for bit in (0, 1, 2, 3) if conn_flags & (1 << bit))
    object_ptr_offset = 10 + (conn_count * MAP_CONN_BLOCK_SIZE)
    object_ptr = _read_rom_u16(rom, header_bank, offset + object_ptr_offset)
    
    warp_count = _read_rom_byte(rom, header_bank, object_ptr + 1)
    warps = []
    for i in range(warp_count):
        base = object_ptr + 2 + (i * 4)
        y = _read_rom_byte(rom, header_bank, base)
        x = _read_rom_byte(rom, header_bank, base + 1)
        dest_map = _read_rom_byte(rom, header_bank, base + 3)
        warps.append({"x": x, "y": y, "dest_map": dest_map})
    return warps

# --- Updated Sprite Logic ---

def read_sprites(memory) -> List[Dict[str, Any]]:
    """
    Reads active sprites, filters out the player (Index 0), and categorizes them.
    Returns: List of {'type': str, 'x': int, 'y': int, 'id': int}
    """
    sprites = []
    # FIX: Start range at 1 to skip Player (Sprite 0)
    for i in range(1, SPRITE_COUNT):
        base = SPRITE_DATA_START + (i * SPRITE_SIZE)
        
        # Byte 0: Picture ID (0=Hidden)
        pic_id = memory[base]
        if pic_id == 0: continue
            
        # Byte 1: Status (Bit 7=Hidden)
        if memory[base + 1] & 0x80: continue

        # Byte 4, 6: Y, X (Pixels, offset -4)
        y_raw = memory[base + 4]
        x_raw = memory[base + 6]
        grid_y = (y_raw + 4) // 16
        grid_x = (x_raw + 4) // 16
        
        # Byte 0xC: Text String ID (The Classifier)
        # Bit 6 set (0x40) -> Trainer
        # Bit 7 set (0x80) -> Item
        text_id = memory[base + 0x0C]
        
        s_type = "NPC"
        if text_id & 0x40:
            s_type = "TRAINER"
        elif text_id & 0x80:
            s_type = "ITEM"
            
        sprites.append({
            "index": i + 1,
            "type": s_type,
            "x": grid_x,
            "y": grid_y,
            "text_id": text_id
        })
    return sprites
