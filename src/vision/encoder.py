import torch
import torch.nn as nn
import numpy as np

class NatureCNN(nn.Module):
    """
    Standard DQN Feature Extractor (Mnih et al. 2015) - Unified Canvas Version.
    Input: (B, 1, 160, 240) - Supports both GB (centered) and GBA (native) games
    Output: (B, 512) feature vector
    """
    def __init__(self, input_channels=1, input_height=160, input_width=240, features_dim=512):
        """Initialize convolutional feature extractor and compute flatten size."""
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        self.cnn = nn.Sequential(
            # Layer 1: 8x8 kernel, stride 4
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: 4x4 kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: 3x3 kernel, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            nn.Flatten()
        )

        # Compute shape by passing a dummy tensor with actual input dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """Return a 512-dim feature vector from a (B, 1, 160, 240) input tensor."""
        if x.max() > 1.0:
             x = x / 255.0

        x = self.cnn(x)
        return self.linear(x)
