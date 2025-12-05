"""
src/env/ram_map.py
Memory addresses for Pokémon Red (US Version).
"""

# --- Player Info ---
HP_CURRENT      = 0xD16C  # 2 Bytes
HP_MAX          = 0xD16E  # 2 Bytes
PARTY_SIZE      = 0xD163  # 1 Byte
X_POS           = 0xD362  # 1 Byte
Y_POS           = 0xD361  # 1 Byte
MAP_N           = 0xD35E  # 1 Byte (Map ID)

# --- Battle Related ---
# 0xD057: 0=None, 1=Wild, 2=Trainer
BATTLE_TYPE     = 0xD057  
ENEMY_HP        = 0xCFE6  # 2 Bytes (Approximate, varies by slot)

# --- Experience (Sum of all party members is complex, tracking first mon for now) ---
# Note: In a full implementation, you'd sum 0xD179, 0xD1A5, etc.
EXP_FIRST_MON   = 0xD179  # 3 Bytes (Big Endian)

# --- Event Flags (For Rewards) ---
# These are bitmasks. 
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END   = 0xD886