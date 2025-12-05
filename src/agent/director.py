import torch
import torch.nn as nn
import numpy as np
from src.vision.encoder import NatureCNN
from src.agent.graph_memory import GraphMemory

class Director(nn.Module):
    """
    The High-Level Controller.
    
    Roles:
    1. Perception: Holds the Shared Vision Encoder (NatureCNN).
    2. Routing: Decides which Specialist (Nav vs Battle) should act.
    3. Memory: Maintains the GraphMemory for long-term topology.
    """
    def __init__(self, action_dim=2, load_path=None):
        super().__init__()
        
        # 1. SHARED VISION (The Eyes)
        # We output 512 features that will be fed to the Specialists too.
        self.encoder = NatureCNN()
        
        # 2. THE ROUTER (The Gating Network)
        # Input: 512 visual features
        # Output: 2 logits (Probability of [Nav_Specialist, Battle_Specialist])
        self.router = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # [0]=Nav, [1]=Battle
        )
        
        # 3. THE CARTOGRAPHER (Non-differentiable Memory)
        self.graph = GraphMemory()
        self.current_node = None
        
        # 4. STATE TRACKING
        self.backtracking_mode = False
        self.target_path = [] # Queue of actions to follow if backtracking

    def forward(self, x):
        """
        Forward pass for the Routing Network.
        Returns: logits, features
        """
        features = self.encoder(x)
        logits = self.router(features)
        return logits, features

    def select_specialist(self, obs, info, epsilon=0.1):
        """
        Decides WHO controls the body.
        
        Args:
            obs: Can be Tensor (B, C, H, W) OR Numpy (C, H, W)
        """
        # 1. PREPARE DATA
        # Neural Network needs Tensor (B, C, H, W)
        # Graph Memory needs Numpy (C, H, W)
        
        if isinstance(obs, torch.Tensor):
            # We have a Tensor (likely on GPU). 
            # Use it for the network, but detach/cpu for the graph.
            obs_t = obs
            # Remove batch dim [0] for the graph -> (1, 84, 84)
            obs_np = obs[0].detach().cpu().numpy().astype(np.uint8)
        else:
            # We have a Numpy array.
            obs_np = obs
            # Create tensor with batch dimension -> (1, 1, 84, 84)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                obs_t = obs_t.cuda()

        # 2. Update the Map (Mapping Phase)
        # Graph uses the CPU Numpy version
        current_hash = self.graph.update(obs_np, last_action=None) 
        
        # 3. Heuristic / Neural Selection
        with torch.no_grad():
            # Network uses the GPU Tensor version
            logits, features = self.forward(obs_t)
            
        # 4. Graph Override (The "Oak's Parcel" Logic)
        if self.backtracking_mode and len(self.target_path) > 0:
            next_move = self.target_path.pop(0)
            return 0, features, next_move 
            
        # 5. Epsilon-Greedy Selection
        if np.random.random() < epsilon:
            specialist_idx = np.random.randint(0, 2)
        else:
            specialist_idx = torch.argmax(logits).item()
            
        return specialist_idx, features, None

    def plan_backtrack(self, target_node_hash):
        """
        Triggers backtracking mode. Uses Dijkstra to find path in Graph.
        """
        path_actions = self.graph.find_path_to_start(target_node_hash)
        if path_actions:
            self.backtracking_mode = True
            self.target_path = path_actions
            return True
        return False