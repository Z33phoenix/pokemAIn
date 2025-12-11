"""
Unified visual observation processing for Game Boy and Game Boy Advance games.

This module implements the "Unified Canvas" strategy that normalizes all game 
observations to a fixed 240x160 resolution regardless of the original platform.
This allows training a single neural network model across GB and GBA games.

Key Features:
- Fixed output: (160, 240, 1) - Height, Width, Channels (Grayscale)
- GB (160x144): Centered with black padding (8px top, 40px left/right)
- GBA (240x160): Direct grayscale conversion
- Preserves text legibility for OCR by avoiding resize/stretch
"""

import numpy as np
from PIL import Image
from typing import Tuple


class UnifiedVisualProcessor:
    """
    Processes game screen observations into unified 240x160 grayscale format.
    
    Target output shape: (160, 240, 1) - Height, Width, Channels
    """
    
    TARGET_WIDTH = 240
    TARGET_HEIGHT = 160
    
    @classmethod
    def process_gb_screen(cls, screen_image: Image.Image) -> np.ndarray:
        """
        Process Game Boy screen (160x144) into unified canvas.
        
        Args:
            screen_image: PIL Image from PyBoy (160x144)
            
        Returns:
            np.ndarray: Unified observation (160, 240, 1)
        """
        # Convert to grayscale
        gray = screen_image.convert("L")
        
        # Verify GB dimensions
        width, height = gray.size
        if width != 160 or height != 144:
            raise ValueError(f"Expected GB screen 160x144, got {width}x{height}")
        
        # Create black canvas
        canvas = np.zeros((cls.TARGET_HEIGHT, cls.TARGET_WIDTH), dtype=np.uint8)
        
        # Calculate centering offsets
        # GB: 160x144 -> center in 240x160
        # Left/right padding: (240 - 160) / 2 = 40px each side  
        # Top/bottom padding: (160 - 144) / 2 = 8px each side
        left_pad = (cls.TARGET_WIDTH - width) // 2  # 40px
        top_pad = (cls.TARGET_HEIGHT - height) // 2  # 8px
        
        # Convert to numpy and place in center
        gb_array = np.array(gray)
        canvas[top_pad:top_pad + height, left_pad:left_pad + width] = gb_array
        
        # Return in PyTorch format: (channels, height, width)
        return canvas[np.newaxis, ...]
    
    @classmethod
    def process_gba_screen(cls, screen_image: Image.Image) -> np.ndarray:
        """
        Process Game Boy Advance screen (240x160) into unified canvas.
        
        Args:
            screen_image: PIL Image from mGBA (240x160)
            
        Returns:
            np.ndarray: Unified observation (160, 240, 1)
        """
        # Convert to grayscale
        gray = screen_image.convert("L")
        
        # Verify GBA dimensions
        width, height = gray.size
        if width != 240 or height != 160:
            raise ValueError(f"Expected GBA screen 240x160, got {width}x{height}")
        
        # Convert to numpy array
        gba_array = np.array(gray, dtype=np.uint8)
        
        # Return in PyTorch format: (channels, height, width)
        return gba_array[np.newaxis, ...]
    
    @classmethod
    def get_observation_space_shape(cls) -> Tuple[int, int, int]:
        """
        Get the standardized observation space shape.
        
        Returns:
            Tuple: (channels, height, width) = (1, 160, 240)
        """
        return (1, cls.TARGET_HEIGHT, cls.TARGET_WIDTH)


def create_unified_observation_space():
    """
    Create standardized Gymnasium observation space for unified visual processing.
    
    Returns:
        gym.spaces.Box: Observation space for 240x160 grayscale images in PyTorch format
    """
    from gymnasium import spaces
    channels, height, width = UnifiedVisualProcessor.get_observation_space_shape()
    return spaces.Box(
        low=0, 
        high=255, 
        shape=(channels, height, width), 
        dtype=np.uint8
    )