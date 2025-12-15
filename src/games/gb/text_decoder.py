from typing import List, Dict, Tuple, Optional
from pyboy import PyBoy

class TextDecoder:
    """
    Reads Pokemon Game Boy Gen 1 text from tile maps and decodes with cursor-based attention.

    Key Features:
    - Reads text directly from tile maps (0x9800/0x9C00 VRAM) instead of RAM buffer
    - Scans tile map for control codes (0x57=done, 0x58=prompt) to detect dialogue completion
    - Accumulates text fragments across multiple frames to handle scrolling text
    - Returns complete dialogues when text stabilizes for 20+ frames (0.33+ seconds)
    - Supports both dialogue and menu text with 5+ character minimum

    Character Encoding:
    - Per Bulbapedia Gen 1 specification with 256-byte character map
    - Control codes (0x49-0x5F) handled separately, not rendered as text
    - Special codes like 0xE1='pk', 0xE2='mn' form "POKéMON"

    Debug Output:
    - Raw hex tiles shown as 0xXX format (e.g., 0x80=A, 0xA0=a)
    - Hex→Char mapping shows each byte's translation
    - Quality metrics show undefined tile ratio
    """

    # Pokemon Gen 1 Text Control Codes
    CONTROL_CODES = {
        0x49: 'page',       # Begins a new Pokédex page
        0x4A: 'pkmn',       # Prints "PKMN"
        0x4B: '_cont',      # Stops and waits for confirmation before scrolling
        0x4C: 'autocont',   # Scroll dialogue down 1 without waiting
        0x4E: 'next_line',  # Move 1 line down in dialogue
        0x4F: 'bottom_line',# Write at the last line of dialogue
        0x50: 'end',        # Marks the end of a string
        0x51: 'paragraph',  # Begin a new dialogue page with button confirmation
        0x52: 'player_name',# Prints the player's name
        0x53: 'rival_name', # Prints the rival's name
        0x54: 'poke',       # Prints "POKé"
        0x55: 'cont',       # A variation of _cont and autocont
        0x56: 'ellipsis',   # Prints "……"
        0x57: 'done',       # Ends text box
        0x58: 'prompt',     # Prompts to end textbox
        0x59: 'target',     # Prints the target of a move
        0x5A: 'user',       # Prints the user of a move
        0x5B: 'pc',         # Prints "PC"
        0x5C: 'tm',         # Prints "TM"
        0x5D: 'trainer',    # Prints "TRAINER"
        0x5E: 'rocket',     # Prints "ROCKET"
        0x5F: 'dex',        # Prints "." and ends the Pokédex entry
    }

    # Gen 1 Pokemon English Character Encoding (from Bulbapedia)
    # Reference: https://bulbapedia.bulbagarden.net/wiki/Character_encoding_(Generation_I)
    CHAR_MAP = {
        # Uppercase letters (0x80-0x9F)
        0x80: 'A', 0x81: 'B', 0x82: 'C', 0x83: 'D', 0x84: 'E', 0x85: 'F', 0x86: 'G', 0x87: 'H',
        0x88: 'I', 0x89: 'J', 0x8A: 'K', 0x8B: 'L', 0x8C: 'M', 0x8D: 'N', 0x8E: 'O', 0x8F: 'P',
        0x90: 'Q', 0x91: 'R', 0x92: 'S', 0x93: 'T', 0x94: 'U', 0x95: 'V', 0x96: 'W', 0x97: 'X',
        0x98: 'Y', 0x99: 'Z', 0x9A: '(', 0x9B: ')', 0x9C: ':', 0x9D: ';', 0x9E: '[', 0x9F: ']',

        # Lowercase letters (0xA0-0xBF)
        0xA0: 'a', 0xA1: 'b', 0xA2: 'c', 0xA3: 'd', 0xA4: 'e', 0xA5: 'f', 0xA6: 'g', 0xA7: 'h',
        0xA8: 'i', 0xA9: 'j', 0xAA: 'k', 0xAB: 'l', 0xAC: 'm', 0xAD: 'n', 0xAE: 'o', 0xAF: 'p',
        0xB0: 'q', 0xB1: 'r', 0xB2: 's', 0xB3: 't', 0xB4: 'u', 0xB5: 'v', 0xB6: 'w', 0xB7: 'x',
        0xB8: 'y', 0xB9: 'z',
        # Accented characters
        0xBA: 'é', 0xBB: '\'d', 0xBC: '\'l', 0xBD: '\'s', 0xBE: '\'t', 0xBF: '\'v',

        # Blanks section (0xC0-0xDF) - all render as spaces
        # Reserved for umlauts in non-English Western versions, but in English render as spaces
        0xC0: ' ', 0xC1: ' ', 0xC2: ' ', 0xC3: ' ', 0xC4: ' ', 0xC5: ' ', 0xC6: ' ', 0xC7: ' ',
        0xC8: ' ', 0xC9: ' ', 0xCA: ' ', 0xCB: ' ', 0xCC: ' ', 0xCD: ' ', 0xCE: ' ', 0xCF: ' ',
        0xD0: ' ', 0xD1: ' ', 0xD2: ' ', 0xD3: ' ', 0xD4: ' ', 0xD5: ' ', 0xD6: ' ', 0xD7: ' ',
        0xD8: ' ', 0xD9: ' ', 0xDA: ' ', 0xDB: ' ', 0xDC: ' ', 0xDD: ' ', 0xDE: ' ', 0xDF: ' ',

        # Numbers (0xF6-0xFF)
        0xF6: '0', 0xF7: '1', 0xF8: '2', 0xF9: '3', 0xFA: '4',
        0xFB: '5', 0xFC: '6', 0xFD: '7', 0xFE: '8', 0xFF: '9',

        # Punctuation and symbols (0xE0-0xEF, 0xF0-0xF5)
        0xE0: '\'', 0xE1: 'pk', 0xE2: 'mn', 0xE3: '-', 0xE4: '\'r', 0xE5: '\'m',
        0xE6: '?', 0xE7: '!', 0xE8: '.', 0xE9: ' ', 0xEA: ' ', 0xEB: ' ', 0xEC: '▷',
        0xED: '▶', 0xEE: '▼', 0xEF: '♂',

        0xF0: '$',  # Pokémon Dollar
        0xF1: '*',  # Multiplication sign
        0xF2: '.',  # Period
        0xF3: '/',  # Slash
        0xF4: ',',  # Comma
        0xF5: '♀',  # Female symbol

        # Variable characters (0x60-0x7E) - context dependent
        # Most render as spaces when not in specific context (like HP bar)
        0x60: '\'', 0x61: '\'', 0x62: ' ', 0x63: ' ', 0x64: ' ', 0x65: ' ',
        0x66: ' ', 0x67: ' ', 0x68: ' ', 0x69: ' ', 0x6A: ' ', 0x6B: ' ',
        0x6C: ' ', 0x6D: ' ', 0x6E: ' ', 0x6F: ' ', 0x70: ' ', 0x71: ' ',
        0x72: ' ', 0x73: ' ', 0x74: ' ', 0x75: ' ', 0x76: ' ', 0x77: ' ',
        0x78: ' ', 0x79: ' ', 0x7A: ' ', 0x7B: ' ', 0x7C: ' ', 0x7D: ' ', 0x7E: ' ',

        # Space character (0x7F is explicitly space)
        0x7F: ' ',

        # Control codes (0x49-0x5F) - should be skipped in text, not rendered
        # These are handled separately in _decode_tiles()

        # Null/empty
        0x00: '',     # Null character - empty
    }

    # Cursor and UI Constants for Gen 1 Pokemon games
    CURSOR_TILE_ID = 0xED  # Cursor arrow tile in Gen 1
    BORDER_TILE_ID = 0x7F  # Border/frame tile in Gen 1 dialogue boxes
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
        # Text accumulation for detecting complete messages
        self._last_complete_narrative = ''
        self._accumulated_narrative = ''
        self._frames_without_change = 0

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

    def read_selection(self, tile_map: List[List[int]], cursor_pos: Tuple[int, int], debug=False) -> str:
        """
        Read text selection starting from cursor position.

        Args:
            tile_map: 2D array of tile IDs
            cursor_pos: (x, y) position of cursor
            debug: If True, print debug information

        Returns:
            Decoded text string of the selection (e.g., "EMBER", "BUY")
        """
        x, y = cursor_pos

        # Start reading from the tile immediately to the right of cursor
        selection_tiles = []
        read_x = x + 1
        consecutive_empty = 0

        # Read until we hit multiple consecutive empty tiles or edge of screen
        while read_x < self.SCREEN_WIDTH_TILES and read_x < len(tile_map[y]):
            tile_id = tile_map[y][read_x]

            # Stop if we hit another cursor (shouldn't happen but safety check)
            if tile_id == self.CURSOR_TILE_ID:
                break

            # Skip control codes
            if 0x49 <= tile_id <= 0x5F:
                read_x += 1
                continue

            # Check if this is a true empty tile
            is_true_empty = tile_id in [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]

            if is_true_empty:
                consecutive_empty += 1
                # Stop after seeing 2+ consecutive empty tiles
                if consecutive_empty > 1:
                    break
            else:
                consecutive_empty = 0
                selection_tiles.append(tile_id)

            read_x += 1

        # Decode the tiles to text
        if debug:
            print(f"DEBUG: Selection raw tiles: {[f'0x{tile:02X}' for tile in selection_tiles]}")
        selection_text = self._decode_tiles(selection_tiles, debug=debug)
        if debug:
            print(f"DEBUG: Selection decoded: '{selection_text}'")
        return selection_text.strip()

    def _decode_tiles(self, tiles: List[int], debug=False) -> str:
        """
        Convert a list of tile IDs to a text string.

        In Pokemon Gen 1:
        - Control codes (0x49-0x5F) don't render text, they control flow
        - Undefined codepoints print as spaces
        - All characters take the same space (monospaced)

        Reference: https://bulbapedia.bulbagarden.net/wiki/Character_encoding_(Generation_I)

        Args:
            tiles: List of tile ID integers
            debug: If True, print detailed decoding information

        Returns:
            Decoded text string
        """
        text_chars = []
        char_map_hits = []  # Track what was decoded
        unknown_count = 0
        total_count = 0

        for tile_id in tiles:
            # Don't count control codes in total - they don't render
            if 0x49 <= tile_id <= 0x5F:
                continue

            total_count += 1

            if tile_id in self.CHAR_MAP:
                char = self.CHAR_MAP[tile_id]
                # Skip empty entries in map
                if char:
                    text_chars.append(char)
                    char_map_hits.append(f"0x{tile_id:02X}→'{char}'")
                else:
                    char_map_hits.append(f"0x{tile_id:02X}→''")
            else:
                # Completely empty/null tiles - skip
                if tile_id in [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]:
                    char_map_hits.append(f"0x{tile_id:02X}→[SKIP:NULL]")
                    continue
                # Undefined characters render as spaces in Pokemon Gen 1
                text_chars.append(' ')
                unknown_count += 1
                char_map_hits.append(f"0x{tile_id:02X}→' '[UNDEFINED]")

        result = ''.join(text_chars).strip()
        # Normalize multiple spaces to single space
        result = ' '.join(result.split())

        # Post-process to fix missing spaces between words
        # In Pokemon Gen 1, if we see letter-letter patterns where a space should be,
        # try to infer it from character boundaries
        result = self._fix_missing_spaces(result)

        # Debug output - shows raw hex tiles and their translations
        if debug:
            print(f"  Raw hex tiles: {[f'0x{t:02X}' for t in tiles]}")
            print(f"  Hex→Char mapping: {' | '.join(char_map_hits)}")
            print(f"  After decoding: '{result}'")
            print(f"  After space normalization: '{result}'")
            print(f"  Quality: {unknown_count} undefined tiles out of {total_count} total")

        # If result is too much garbage, return empty
        if total_count > 0 and total_count > 3 and unknown_count / total_count > 0.8:
            return ''

        return result

    def _fix_missing_spaces(self, text: str) -> str:
        """
        Post-process text to fix missing spaces between words.

        If the tilemap doesn't contain space tiles, we can infer spaces based on
        character patterns. In Pokemon Gen 1 dialogue, certain character combinations
        are unlikely to be concatenated without a space:
        - Lowercase letters followed by uppercase letters (e.g., "iPlay" -> "i Play")
        - Common word endings followed by common word starts

        Args:
            text: Decoded text string potentially missing spaces

        Returns:
            Text with inferred spaces inserted
        """
        if not text or len(text) < 2:
            return text

        # Don't modify text that already has spaces
        if ' ' in text:
            return text

        # Check for patterns where spaces are likely missing
        result = []
        for i, char in enumerate(text):
            result.append(char)

            # Look ahead to next character
            if i < len(text) - 1:
                next_char = text[i + 1]

                # Insert space between lowercase and uppercase (e.g., "iPlay")
                # This is a strong indicator of a word boundary in English
                if char.islower() and next_char.isupper():
                    result.append(' ')

        return ''.join(result)

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

    def read_narrative(self, tile_map: List[List[int]], debug=False) -> str:
        """
        Read text from the dialogue box area (bottom of screen).

        Args:
            tile_map: 2D array of tile IDs
            debug: If True, print detailed decoding information

        Returns:
            Decoded narrative/dialogue text string
        """
        narrative_lines = []

        # Read each row in the dialogue box area
        for row in range(self.DIALOGUE_START_ROW, min(self.DIALOGUE_END_ROW + 1, len(tile_map))):
            row_tiles = []

            # Determine if this is a border row (top or bottom of dialogue box)
            is_border_row = (row == self.DIALOGUE_START_ROW or row == self.DIALOGUE_END_ROW)

            # Read each column in this row
            for col in range(self.SCREEN_WIDTH_TILES):
                tile_id = tile_map[row][col]

                # Skip border tiles:
                # - At the left and right edges (columns 0 and 19) in all rows
                # - Everywhere in top/bottom rows of the dialogue box
                if is_border_row or (col == 0 or col == 19):
                    if tile_id == self.BORDER_TILE_ID:
                        continue

                row_tiles.append(tile_id)

            # Decode this row and add to narrative
            if row_tiles:  # Only process non-empty rows
                if debug:
                    print(f"  Row {row} raw tiles: {[f'0x{t:02X}' for t in row_tiles]}")
                row_text = self._decode_tiles(row_tiles, debug=debug)
                # Don't strip individual rows - preserve spacing between rows
                if row_text and row_text.strip():  # Only add non-empty text
                    narrative_lines.append(row_text.strip())

        # Join all lines with spaces
        return ' '.join(narrative_lines).strip()

    def _scan_tilemap_for_control_codes(self, tile_map: List[List[int]], debug: bool = False) -> Tuple[bool, bool, bool]:
        """
        Scan tile map for control code markers that indicate dialogue state.

        In Pokémon Gen 1, control codes (0x49-0x5F) appear directly in the tile map as special
        tile IDs. This method replaces the old broken _read_text_buffer() that was reading from
        wrong memory address 0xC000 (ROM space). Now we read control codes from the actual tile
        map where the game renders them.

        Key markers:
        - 0x50 (page end): More text coming, press A to continue to next page
        - 0x57 (done): Dialogue completely finished, box will close
        - 0x58 (prompt): Wait for player input (appears at end of page content)

        Args:
            tile_map: 2D array of tile IDs from current screen (from VRAM 0x9800/0x9C00)
            debug: If True, print control code markers found

        Returns:
            Tuple of (has_page_end, has_done, has_prompt) booleans
        """
        has_page_end_marker = False
        has_done_marker = False
        has_prompt_marker = False

        # Scan the dialogue box area for control codes
        for row in range(self.DIALOGUE_START_ROW, min(self.DIALOGUE_END_ROW + 1, len(tile_map))):
            for col in range(self.SCREEN_WIDTH_TILES):
                tile_id = tile_map[row][col]

                # Check for control code markers
                if tile_id == 0x50:  # page end marker
                    has_page_end_marker = True
                elif tile_id == 0x57:  # done marker - dialogue finished
                    has_done_marker = True
                elif tile_id == 0x58:  # prompt marker - wait for player input
                    has_prompt_marker = True

        if debug and (has_page_end_marker or has_done_marker or has_prompt_marker):
            print(f"DEBUG: Tilemap control codes - page_end:{has_page_end_marker}, done:{has_done_marker}, prompt:{has_prompt_marker}")

        return has_page_end_marker, has_done_marker, has_prompt_marker

    def _get_complete_narrative(self, tile_map_narrative: str, tile_map: List[List[int]], debug=False) -> str:
        """
        Detect when a complete dialogue sequence has finished and return it once.

        PROBLEM SOLVED: The old implementation returned text multiple times per dialogue box.
        This caused "NARRATIVE_TEXT: Hello | NARRATIVE_TEXT: Hello world | NARRATIVE_TEXT: Hello world!"
        instead of a single complete message.

        SOLUTION: Only return when:
        1. Text stabilizes for 20+ frames (0.33+ seconds) with 5+ characters (tertiary - handles most cases)
        2. Text clears after 10+ frames of stability (primary - end of box)
        3. 0x57 (done) marker appears (secondary - explicit end signal)

        ACCUMULATION: Text fragments are merged using 4-case logic:
        - Case 1: Skip duplicates (exact same text seen before)
        - Case 2: Replace with fuller version (progressive text reveal)
        - Case 3: Merge overlapping text (e.g., "Hello w" + "world!" = "Hello world!")
        - Case 4: Append new unique content (different text sections)

        Args:
            tile_map_narrative: Current narrative text extracted from tile map
            tile_map: 2D array of tile IDs (scanned for control codes 0x57, 0x58, 0x50)
            debug: If True, prints dialogue state transitions and fragment merging details

        Returns:
            Complete dialogue text (non-empty) if dialogue just finished, empty string otherwise
        """
        # Initialize if needed
        if not hasattr(self, '_accumulated_narrative'):
            self._accumulated_narrative = ''
        if not hasattr(self, '_last_returned_narrative'):
            self._last_returned_narrative = ''
        if not hasattr(self, '_current_narrative'):
            self._current_narrative = ''
        if not hasattr(self, '_stable_frames'):
            self._stable_frames = 0
        if not hasattr(self, '_text_fragments'):
            self._text_fragments = []  # List of all text fragments seen
        if not hasattr(self, '_dialogue_active'):
            self._dialogue_active = False

        # Check tile map for control codes instead of reading from wrong memory address
        has_page_end, has_done, has_prompt = self._scan_tilemap_for_control_codes(tile_map, debug=debug)

        # Clean up the tile map narrative (remove continuation marker)
        # BUT track if the continuation marker was present - that's our return signal!
        has_continuation_marker = '▼' in (tile_map_narrative or '')
        clean_narrative = tile_map_narrative.replace('▼', '').strip() if tile_map_narrative else ''

        # Detect if we just transitioned to empty (dialogue box closed)
        # This happens when we had non-empty text before, now have empty
        had_text_before = self._current_narrative and len(self._current_narrative) > 0
        now_empty = not clean_narrative or len(clean_narrative) == 0
        text_just_became_empty = had_text_before and now_empty

        # Detect new dialogue session starting (transition from empty to has content)
        # Check BEFORE updating _current_narrative
        was_empty = not self._current_narrative or len(self._current_narrative) == 0
        now_has_content = clean_narrative and len(clean_narrative) > 0
        dialogue_just_started = was_empty and now_has_content

        if dialogue_just_started and not self._dialogue_active:
            # Starting dialogue from truly empty state
            self._dialogue_active = True
            self._accumulated_narrative = ''
            self._last_returned_narrative = ''
            if debug:
                print(f"DEBUG: New dialogue session detected (empty->content), starting fresh dialogue")

        # Track stability of current text
        if clean_narrative != self._current_narrative:
            self._current_narrative = clean_narrative
            self._stable_frames = 0
        else:
            self._stable_frames += 1

        # Accumulate text fragments when it's being displayed
        if clean_narrative:
            # Add this fragment if it's new/different
            if not self._text_fragments or clean_narrative != self._text_fragments[-1]:
                # Check if this fragment is COMPLETELY unrelated to accumulated text
                # This indicates a new dialogue source (e.g., SNES -> MOM transition)
                if self._accumulated_narrative and len(self._accumulated_narrative) >= 5:
                    is_new_dialogue_source = self._is_unrelated_content(clean_narrative, debug)
                    if is_new_dialogue_source:
                        # Return current accumulated content and start fresh
                        if debug:
                            print(f"DEBUG: Detected new dialogue source, returning accumulated: '{self._accumulated_narrative}'")
                        result = self._accumulated_narrative
                        self._last_returned_narrative = result
                        # Reset for new dialogue
                        self._accumulated_narrative = ''
                        self._text_fragments = [clean_narrative]
                        self._current_narrative = clean_narrative
                        self._stable_frames = 0
                        # Note: We'll return this result at the end of this call
                        # Store it for return after normal processing
                        if not hasattr(self, '_pending_return'):
                            self._pending_return = None
                        self._pending_return = result
                        if debug:
                            print(f"DEBUG: Starting fresh with new dialogue source: '{clean_narrative}'")
                    else:
                        self._text_fragments.append(clean_narrative)
                        if debug:
                            print(f"DEBUG: Added fragment: '{clean_narrative}' (total: {len(self._text_fragments)} fragments)")
                else:
                    self._text_fragments.append(clean_narrative)
                    if debug:
                        print(f"DEBUG: Added fragment: '{clean_narrative}' (total: {len(self._text_fragments)} fragments)")

            # Try to build accumulated narrative from all fragments
            if len(self._text_fragments) > 1:
                # Try to connect fragments by finding overlaps
                self._try_connect_fragments(debug)

        # Return narrative when dialogue box closes
        # Strategy: Return when text has been completely stable (not typing anymore)
        should_return = False
        end_reason = ''

        # Primary: Text is stable for SHORT time and box JUST cleared - dialogue line complete
        # The key: box must stay empty for at least 1 frame, AND text must be minimal (5+ chars to avoid noise)
        if (text_just_became_empty and self._stable_frames >= 10 and
            self._accumulated_narrative and len(self._accumulated_narrative) >= 5 and
            self._text_fragments and self._dialogue_active):
            # Box cleared after text was stable for 0.17+ seconds - dialogue line/page is complete
            if len(self._text_fragments) > 1:
                self._try_connect_fragments(debug)
            elif len(self._text_fragments) == 1:
                self._accumulated_narrative = self._text_fragments[0]

            should_return = True
            end_reason = 'text_stable_10f_then_cleared'
            if debug:
                print(f"DEBUG: Stable text ({self._stable_frames}f, {len(self._accumulated_narrative)} chars) then cleared, returning: '{self._accumulated_narrative}'")
        # Secondary: Check for 0x57 (done) marker in text buffer (definitive end)
        # Require minimal text (5+ chars) to avoid partial returns
        elif has_done and self._accumulated_narrative and len(self._accumulated_narrative) >= 5 and self._text_fragments:
            # Text buffer shows done marker AND we have text - dialogue is complete
            # Rebuild accumulated narrative before returning
            if len(self._text_fragments) > 1:
                self._try_connect_fragments(debug)
            elif len(self._text_fragments) == 1:
                self._accumulated_narrative = self._text_fragments[0]

            should_return = True
            end_reason = 'marker_0x57_with_text'
            if debug:
                print(f"DEBUG: Found done (0x57) marker with {len(self._accumulated_narrative)} chars, returning: '{self._accumulated_narrative}'")
        # Tertiary: Text is VERY stable for extended period (20+ frames = 0.33+ seconds)
        # This handles dialogues that don't have explicit end markers
        # When text stops changing for this long with minimal content (5+ chars), dialogue is likely complete
        elif (self._stable_frames >= 20 and
              self._accumulated_narrative and len(self._accumulated_narrative) >= 5 and
              self._text_fragments and self._dialogue_active):
            # Text has been completely stable for 0.33+ seconds - dialogue is complete
            if len(self._text_fragments) > 1:
                self._try_connect_fragments(debug)
            elif len(self._text_fragments) == 1:
                self._accumulated_narrative = self._text_fragments[0]

            should_return = True
            end_reason = 'text_stable_20f_no_change'
            if debug:
                print(f"DEBUG: Text stable for {self._stable_frames}f with {len(self._accumulated_narrative)} chars, returning: '{self._accumulated_narrative}'")

        if should_return and self._accumulated_narrative != self._last_returned_narrative:
            if debug:
                print(f"DEBUG: Dialogue ended ({end_reason}), returning: '{self._accumulated_narrative}'")
            self._last_returned_narrative = self._accumulated_narrative
            result = self._accumulated_narrative
            # Reset for next dialogue
            self._accumulated_narrative = ''
            self._text_fragments = []
            self._current_narrative = ''
            self._stable_frames = 0
            self._dialogue_active = False
            return result

        # Check for pending return from new dialogue source detection
        if hasattr(self, '_pending_return') and self._pending_return:
            result = self._pending_return
            self._pending_return = None
            return result

        return ''

    def _try_connect_fragments(self, debug=False) -> None:
        """
        Build complete narrative by accumulating all unique text from fragments.

        As the dialogue box scrolls, we see different portions of the narrative.
        This method collects all unique text that appears, preserving fragments
        that don't overlap and merging those that do.
        """
        if len(self._text_fragments) < 1:
            self._accumulated_narrative = ''
            return

        if len(self._text_fragments) == 1:
            self._accumulated_narrative = self._text_fragments[0]
            return

        # Start with the first fragment and accumulate all unique content
        accumulated = self._text_fragments[0]

        for i in range(1, len(self._text_fragments)):
            fragment = self._text_fragments[i]

            # Case 1: Fragment is already fully contained in accumulated text (duplicate)
            if fragment in accumulated:
                if debug:
                    print(f"DEBUG: Fragment '{fragment}' already in accumulated, skipping")
                continue

            # Case 2: Accumulated text is fully contained in fragment (fragment is fuller version)
            # This can happen when text is revealed progressively
            if accumulated in fragment:
                if debug:
                    print(f"DEBUG: Accumulated contained in fragment, replacing: '{accumulated}' -> '{fragment}'")
                accumulated = fragment
                continue

            # Case 3: Check for overlap between end of accumulated and beginning of fragment
            overlap_found = False
            for overlap_len in range(min(len(accumulated), len(fragment)), 0, -1):
                if accumulated[-overlap_len:] == fragment[:overlap_len]:
                    # Found overlap - extract and append only the new part
                    new_content = fragment[overlap_len:]
                    if new_content.strip():
                        accumulated += new_content
                        if debug:
                            print(f"DEBUG: Found {overlap_len}-char overlap, appending: '{new_content}'")
                    overlap_found = True
                    break

            # Case 4: No overlap found - append as new unique content
            if not overlap_found:
                accumulated += ' ' + fragment
                if debug:
                    print(f"DEBUG: No overlap with accumulated, appending new content: '{fragment}'")

        # Normalize the final result
        self._accumulated_narrative = accumulated.strip()
        # Normalize multiple spaces to single space
        self._accumulated_narrative = ' '.join(self._accumulated_narrative.split())

        if debug:
            print(f"DEBUG: Final accumulated narrative: '{self._accumulated_narrative}'")

    def _is_unrelated_content(self, new_fragment: str, debug: bool = False) -> bool:
        """
        Detect if a new fragment is completely unrelated to current accumulated text.

        This catches transitions between different dialogue sources (e.g., SNES -> MOM)
        where the game doesn't clear the text box but shows completely new content.

        Detection criteria:
        1. No overlap between end of accumulated and start of fragment
        2. Fragment doesn't contain any significant words from accumulated
        3. Accumulated doesn't contain any significant words from fragment

        Args:
            new_fragment: The new text fragment to check
            debug: If True, print detection details

        Returns:
            True if this appears to be from a completely different dialogue source
        """
        if not self._accumulated_narrative or not new_fragment:
            return False

        accumulated = self._accumulated_narrative

        # Check 1: Is there ANY overlap between end of accumulated and start of fragment?
        # If there's overlap, it's likely a continuation
        for overlap_len in range(min(len(accumulated), len(new_fragment), 20), 2, -1):
            if accumulated[-overlap_len:] == new_fragment[:overlap_len]:
                if debug:
                    print(f"DEBUG: Found {overlap_len}-char overlap, content is related")
                return False

        # Check 2: Is the fragment contained in accumulated or vice versa?
        if new_fragment in accumulated or accumulated in new_fragment:
            return False

        # Check 3: Do they share significant words? (3+ char words)
        accumulated_words = set(w.lower() for w in accumulated.split() if len(w) >= 3)
        fragment_words = set(w.lower() for w in new_fragment.split() if len(w) >= 3)

        # Remove very common words that might appear in any dialogue
        common_words = {'the', 'and', 'you', 'for', 'are', 'but', 'not', 'all', 'can', 'had',
                       'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its', 'let',
                       'may', 'new', 'now', 'old', 'see', 'way', 'who', 'boy', 'did', 'get',
                       'got', 'him', 'use', 'say', 'she', 'too', 'any', 'day', 'mom', 'red'}
        accumulated_words -= common_words
        fragment_words -= common_words

        shared_words = accumulated_words & fragment_words

        if shared_words:
            if debug:
                print(f"DEBUG: Shared significant words: {shared_words}, content is related")
            return False

        # No overlap, no containment, no shared significant words = unrelated content
        if debug:
            print(f"DEBUG: No relationship found between '{accumulated[:30]}...' and '{new_fragment[:30]}...'")
        return True

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
                    selection = self.read_selection(tile_map, cursor_pos, debug=debug)
                    if debug:
                        print(f"DEBUG: Found cursor at {cursor_pos} in {layer_name} layer, raw selection: '{selection}'")
                    if self._is_valid_text(selection, debug=debug):
                        result['selection'] = selection
                    elif debug:
                        print(f"DEBUG: Selection failed validation: '{selection}'")

            # Read narrative/dialogue text
            if not result['narrative']:  # Only if we haven't found one yet
                if debug:
                    print(f"DEBUG: Reading narrative from {layer_name} layer:")
                raw_narrative = self.read_narrative(tile_map, debug=debug)
                if debug:
                    print(f"DEBUG: Raw narrative from {layer_name} layer: '{raw_narrative}'")

                # Use tilemap accumulation method to detect when complete dialogue is ready
                complete_narrative = self._get_complete_narrative(raw_narrative, tile_map, debug=debug)
                if complete_narrative and self._is_valid_text(complete_narrative, debug=debug):
                    result['narrative'] = complete_narrative
                elif debug and complete_narrative:
                    print(f"DEBUG: Complete narrative failed validation: '{complete_narrative}'")

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
