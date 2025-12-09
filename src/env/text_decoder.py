from typing import List
from pyboy import PyBoy

class TextDecoder:
    """Reads Pokemon Red internal text buffers and decodes them."""
    
    # Gen 1 Custom Charset (Partial Mapping for critical chars)
    CHAR_MAP = {
        0x80: 'A', 0x81: 'B', 0x82: 'C', 0x83: 'D', 0x84: 'E', 0x85: 'F', 0x86: 'G', 0x87: 'H',
        0x88: 'I', 0x89: 'J', 0x8A: 'K', 0x8B: 'L', 0x8C: 'M', 0x8D: 'N', 0x8E: 'O', 0x8F: 'P',
        0x90: 'Q', 0x91: 'R', 0x92: 'S', 0x93: 'T', 0x94: 'U', 0x95: 'V', 0x96: 'W', 0x97: 'X',
        0x98: 'Y', 0x99: 'Z', 0x9A: '(', 0x9B: ')', 0x9C: ':', 0x9D: ';', 0x9E: '[', 0x9F: ']',
        0xA0: 'a', 0xA1: 'b', 0xA2: 'c', 0xA3: 'd', 0xA4: 'e', 0xA5: 'f', 0xA6: 'g', 0xA7: 'h',
        0xA8: 'i', 0xA9: 'j', 0xAA: 'k', 0xAB: 'l', 0xAC: 'm', 0xAD: 'n', 0xAE: 'o', 0xAF: 'p',
        0xB0: 'q', 0xB1: 'r', 0xB2: 's', 0xB3: 't', 0xB4: 'u', 0xB5: 'v', 0xB6: 'w', 0xB7: 'x',
        0xB8: 'y', 0xB9: 'z', 0xE8: '.', 0xF2: '?', 0xF3: '!', 0x50: ' '
    }

    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy
        # Common text buffer address for Gen 1 (often 0xCF4B or window map 0x9C00)
        # We scan the Window Tile Map area which is usually used for dialogue
        self.window_addr_start = 0x9C00 
        self.window_addr_end = 0x9FFF

    def read_current_text(self) -> str:
        """Scans the VRAM Window layer for visible text."""
        # Simple heuristic: Read bytes, map to chars, join.
        text_content = []
        # Optimization: Only read the first few lines of the window
        for addr in range(self.window_addr_start, self.window_addr_start + 80): 
            val = self.pyboy.memory[addr]
            if val in self.CHAR_MAP:
                text_content.append(self.CHAR_MAP[val])
        
        full_str = "".join(text_content).strip()
        return full_str if len(full_str) > 3 else "" # Filter noise