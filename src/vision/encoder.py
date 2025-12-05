import torch
import torch.nn as nn
import numpy as np

class NatureCNN(nn.Module):
    """
    Standard DQN Feature Extractor (Mnih et al. 2015).
    Input: (B, 1, 84, 84)
    Output: (B, 512) feature vector
    """
    def __init__(self, input_channels=1, features_dim=512):
        super().__init__()
        
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

        # Compute shape by passing a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Expects float input normalized [0,1] or [0, 255]
        # If input is 0-255, we normalize.
        if x.max() > 1.0:
             x = x / 255.0
             
        x = self.cnn(x)
        return self.linear(x)