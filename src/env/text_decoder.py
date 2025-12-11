"""
DEPRECATED: This module has moved to src.games.gb.text_decoder

TextDecoder is Pokemon Red-specific and has been moved to the gb/ folder
to support the new hot-swappable game architecture.

For new code, import from the new location:
    from src.games.gb.text_decoder import TextDecoder

Note: TextDecoder is game-specific. When using multiple games, check
which game is active before using game-specific features.
"""

# Backward compatibility - re-export from new location
from src.games.gb.text_decoder import TextDecoder

__all__ = ['TextDecoder']
