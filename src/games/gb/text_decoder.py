from typing import List, Dict, Tuple, Optional
from pyboy import PyBoy

class TextDecoder:
    """Reads Pokemon Game Boy internal text buffers and decodes them with cursor-based attention."""
    
    # Gen 1 Custom Charset (Expanded mapping for Pokemon GB games)
    CHAR_MAP = {
        # Uppercase letters
        0x80: 'A', 0x81: 'B', 0x82: 'C', 0x83: 'D', 0x84: 'E', 0x85: 'F', 0x86: 'G', 0x87: 'H',
        0x88: 'I', 0x89: 'J', 0x8A: 'K', 0x8B: 'L', 0x8C: 'M', 0x8D: 'N', 0x8E: 'O', 0x8F: 'P',
        0x90: 'Q', 0x91: 'R', 0x92: 'S', 0x93: 'T', 0x94: 'U', 0x95: 'V', 0x96: 'W', 0x97: 'X',
        0x98: 'Y', 0x99: 'Z', 0x9A: '(', 0x9B: ')', 0x9C: ':', 0x9D: ';', 0x9E: '[', 0x9F: ']',
        
        # Lowercase letters
        0xA0: 'a', 0xA1: 'b', 0xA2: 'c', 0xA3: 'd', 0xA4: 'e', 0xA5: 'f', 0xA6: 'g', 0xA7: 'h',
        0xA8: 'i', 0xA9: 'j', 0xAA: 'k', 0xAB: 'l', 0xAC: 'm', 0xAD: 'n', 0xAE: 'o', 0xAF: 'p',
        0xB0: 'q', 0xB1: 'r', 0xB2: 's', 0xB3: 't', 0xB4: 'u', 0xB5: 'v', 0xB6: 'w', 0xB7: 'x',
        0xB8: 'y', 0xB9: 'z',
        
        # Numbers
        0xF6: '0', 0xF7: '1', 0xF8: '2', 0xF9: '3', 0xFA: '4', 
        0xFB: '5', 0xFC: '6', 0xFD: '7', 0xFE: '8', 0xFF: '9',
        
        # Punctuation and symbols
        0x50: ' ',    # Space (common)
        0xE7: ' ',    # Alternative space encoding
        0xE8: '.',    # Period
        0xE9: ',',    # Comma
        0xF2: '?',    # Question mark
        0xF3: '!',    # Exclamation
        0x3B: '>',    # Arrow/pointer
        0xF4: '\'',   # Apostrophe
        0xE6: '\'',   # Alternative apostrophe encoding
        0xF5: '"',    # Quote
        0xF0: '-',    # Hyphen/dash
        0xF1: '/',    # Slash
        
        # Common empty/filler tiles that should be ignored  
        0x00: '',     # Null tile
        0x7F: '',     # Common border/empty tile (NOT a space in text)
        0x01: '',     # Another common empty tile
        0x02: '',     # Another common empty tile
    }
    
    # Cursor and UI Constants for Gen 1 Pokemon games
    CURSOR_TILE_ID = 0xED  # Cursor arrow tile in Gen 1
    BORDER_TILE_ID = 0x7F  # Common border/frame tile
    EMPTY_TILE_ID = 0x7F   # Empty/space tile
    
    # Screen dimensions in tiles
    SCREEN_WIDTH_TILES = 20   # Game Boy screen is 20 tiles wide
    SCREEN_HEIGHT_TILES = 18  # Game Boy screen is 18 tiles high
    
    # Dialogue box area (bottom portion of screen)
    DIALOGUE_START_ROW = 12   # Dialogue typically starts at row 12
    DIALOGUE_END_ROW = 17     # Dialogue typically ends at row 17

    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy
        # Background Tile Map 1 (0x9800-0x9BFF) - Main game screen
        self.bg_map_addr = 0x9800
        # Background Tile Map 2 (0x9C00-0x9FFF) - Alternative tilemap
        self.bg_map_addr_2 = 0x9C00
        # Window Tile Map (0x9C00-0x9FFF) - UI overlays, dialogue
        self.window_addr_start = 0x9C00 
        self.window_addr_end = 0x9FFF

    def _get_tile_map(self, use_window_layer=False) -> List[List[int]]:
        """
        Read the background tile map as a 2D array of tile IDs.
        
        Args:
            use_window_layer: If True, try to read from window layer (for UI/text)
        
        Returns:
            List of rows, each containing tile IDs for that row.
        """
        tile_map = []
        
        # Choose which tilemap to read from
        base_addr = self.window_addr_start if use_window_layer else self.bg_map_addr
        
        for row in range(self.SCREEN_HEIGHT_TILES):
            row_tiles = []
            for col in range(self.SCREEN_WIDTH_TILES):
                # Calculate VRAM address for this tile position
                # Game Boy tilemap is 32x32, but screen is only 20x18 visible
                addr = base_addr + (row * 32) + col
                tile_id = self.pyboy.memory[addr]
                row_tiles.append(tile_id)
            tile_map.append(row_tiles)
        return tile_map

    def find_cursor(self, tile_map: List[List[int]]) -> Optional[Tuple[int, int]]:
        """
        Scan the tile map to find the cursor arrow tile.
        
        Args:
            tile_map: 2D array of tile IDs (height x width)
            
        Returns:
            Tuple of (x, y) coordinates if cursor found, None otherwise
        """
        for y in range(len(tile_map)):
            for x in range(len(tile_map[y])):
                if tile_map[y][x] == self.CURSOR_TILE_ID:
                    return (x, y)
        return None

    def read_selection(self, tile_map: List[List[int]], cursor_pos: Tuple[int, int]) -> str:
        """
        Read text selection starting from cursor position.
        
        Args:
            tile_map: 2D array of tile IDs
            cursor_pos: (x, y) position of cursor
            
        Returns:
            Decoded text string of the selection (e.g., "EMBER", "BUY")
        """
        x, y = cursor_pos
        
        # Start reading from the tile immediately to the right of cursor
        selection_tiles = []
        read_x = x + 1
        
        # Read until we hit a border, empty tile, or edge of screen
        while read_x < self.SCREEN_WIDTH_TILES and read_x < len(tile_map[y]):
            tile_id = tile_map[y][read_x]
            
            # Stop reading if we hit a border or empty tile
            if tile_id == self.BORDER_TILE_ID or tile_id == self.EMPTY_TILE_ID:
                break
                
            # Stop if we hit another cursor (shouldn't happen but safety check)
            if tile_id == self.CURSOR_TILE_ID:
                break
                
            selection_tiles.append(tile_id)
            read_x += 1
        
        # Decode the tiles to text
        selection_text = self._decode_tiles(selection_tiles)
        return selection_text.strip()

    def _decode_tiles(self, tiles: List[int]) -> str:
        """
        Convert a list of tile IDs to a text string.
        
        Args:
            tiles: List of tile ID integers
            
        Returns:
            Decoded text string
        """
        text_chars = []
        unknown_count = 0
        total_count = 0
        
        for tile_id in tiles:
            total_count += 1
            if tile_id in self.CHAR_MAP:
                char = self.CHAR_MAP[tile_id]
                if char:  # Only add non-empty characters
                    text_chars.append(char)
            else:
                # Skip common empty/background tiles
                if tile_id not in [0x00, 0x7F, 0x01, 0x02, 0x03, 0x04, 0x05]:
                    unknown_count += 1
        
        result = ''.join(text_chars).strip()
        
        # If too many unknown characters, this is probably not real text
        if total_count > 0 and unknown_count / total_count > 0.5:
            return ''
        
        return result

    def _is_valid_text(self, text: str, debug=False) -> bool:
        """
        Validate if a decoded string looks like actual game text.
        
        Args:
            text: Decoded text string
            debug: If True, print debug info about validation
            
        Returns:
            True if text appears to be valid, False if it's likely garbage
        """
        if not text:
            if debug: print(f"DEBUG: Text validation failed - empty text")
            return False
        
        # Be less strict about minimum length for short menu items
        if len(text.strip()) < 1:
            if debug: print(f"DEBUG: Text validation failed - too short after strip: '{text}'")
            return False
        
        # Count valid characters (letters, numbers, common punctuation)
        valid_chars = 0
        total_chars = len(text)
        
        for char in text:
            if char.isalnum() or char in ' .,!?-\'":()[]>':
                valid_chars += 1
        
        # Be more lenient with character validation (50% instead of 70%)
        valid_ratio = valid_chars / total_chars if total_chars > 0 else 0
        if debug: print(f"DEBUG: Text validation - valid chars: {valid_chars}/{total_chars} ({valid_ratio:.2f})")
        
        if valid_ratio < 0.5:
            if debug: print(f"DEBUG: Text validation failed - low valid character ratio: {valid_ratio}")
            return False
        
        # Check for excessive repetition (like "?????" from unknown tiles) 
        for char in '?*#@$%^&':
            if char * 3 in text:
                if debug: print(f"DEBUG: Text validation failed - excessive repetition of '{char}'")
                return False
        
        if debug: print(f"DEBUG: Text validation passed for: '{text}'")
        return True

    def read_narrative(self, tile_map: List[List[int]]) -> str:
        """
        Read text from the dialogue box area (bottom of screen).
        
        Args:
            tile_map: 2D array of tile IDs
            
        Returns:
            Decoded narrative/dialogue text string
        """
        narrative_lines = []
        
        # Read each row in the dialogue box area
        for row in range(self.DIALOGUE_START_ROW, min(self.DIALOGUE_END_ROW + 1, len(tile_map))):
            row_tiles = []
            
            # Read each column in this row
            for col in range(self.SCREEN_WIDTH_TILES):
                tile_id = tile_map[row][col]
                
                # Skip border tiles at the edges
                if tile_id == self.BORDER_TILE_ID:
                    continue
                    
                row_tiles.append(tile_id)
            
            # Decode this row and add to narrative
            if row_tiles:  # Only process non-empty rows
                row_text = self._decode_tiles(row_tiles).strip()
                if row_text:  # Only add non-empty text
                    narrative_lines.append(row_text)
        
        # Join all lines with spaces
        return ' '.join(narrative_lines).strip()

    def decode(self, debug=False) -> Dict[str, str]:
        """
        Main decode method that returns structured text data.
        
        Args:
            debug: If True, print debug information about tile reading
        
        Returns:
            Dictionary with 'selection' and 'narrative' text:
            {
                'selection': 'EMBER',     # What the cursor is pointing to
                'narrative': 'Wild PIDGEY appeared!'  # Dialogue/story text
            }
        """
        # Initialize result
        result = {
            'selection': '',
            'narrative': ''
        }
        
        # Try both background and window layers
        for layer_name, use_window in [("background", False), ("window", True)]:
            tile_map = self._get_tile_map(use_window_layer=use_window)
            
            if debug:
                print(f"DEBUG: Trying {layer_name} layer (addr: 0x{(self.window_addr_start if use_window else self.bg_map_addr):04X})")
            
            # Look for cursor-based selection
            if not result['selection']:  # Only if we haven't found one yet
                cursor_pos = self.find_cursor(tile_map)
                if cursor_pos is not None:
                    selection = self.read_selection(tile_map, cursor_pos)
                    if debug:
                        print(f"DEBUG: Found cursor at {cursor_pos} in {layer_name} layer, raw selection: '{selection}'")
                    if self._is_valid_text(selection, debug=debug):
                        result['selection'] = selection
                    elif debug:
                        print(f"DEBUG: Selection failed validation: '{selection}'")
            
            # Read narrative/dialogue text  
            if not result['narrative']:  # Only if we haven't found one yet
                narrative = self.read_narrative(tile_map)
                if debug:
                    print(f"DEBUG: Raw narrative from {layer_name} layer: '{narrative}'")
                if self._is_valid_text(narrative, debug=debug):
                    result['narrative'] = narrative
                elif debug and narrative:
                    print(f"DEBUG: Narrative failed validation: '{narrative}'")
            
            # Debug: Show some tile samples
            if debug and len(tile_map) > 0:
                print(f"DEBUG: {layer_name} layer - Sample tiles from row 0: {[f'0x{tile:02X}' for tile in tile_map[0][:10]]}")
                if len(tile_map) > self.DIALOGUE_START_ROW:
                    print(f"DEBUG: {layer_name} layer - Sample tiles from dialogue row {self.DIALOGUE_START_ROW}: {[f'0x{tile:02X}' for tile in tile_map[self.DIALOGUE_START_ROW][:10]]}")
            
            # If we found both selection and narrative, no need to check other layer
            if result['selection'] and result['narrative']:
                break
        
        return result

    def read_current_text(self) -> str:
        """
        Legacy method: Scans the VRAM Window layer for visible text.
        
        DEPRECATED: Use decode() method instead for structured cursor-based attention.
        This method is kept for backwards compatibility.
        """
        # Simple heuristic: Read bytes, map to chars, join.
        text_content = []
        # Optimization: Only read the first few lines of the window
        for addr in range(self.window_addr_start, self.window_addr_start + 80): 
            val = self.pyboy.memory[addr]
            if val in self.CHAR_MAP:
                text_content.append(self.CHAR_MAP[val])
        
        full_str = "".join(text_content).strip()
        return full_str if len(full_str) > 3 else "" # Filter noise
