import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CrossQNavAgent(nn.Module):
    """
    Navigation Specialist using CrossQ (Batch-Norm DDPG/DQN style).
    
    Key Concept:
    - Removes the "Target Network" found in standard DQN.
    - Uses Batch Normalization to stabilize Q-values immediately.
    - Result: Learns simple tasks (like walking) much faster.
    """
    def __init__(self, input_dim=512, action_dim=8, lr=1e-4, gamma=0.99):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        
        # 1. The Q-Network (No Target Net!)
        # We use Batch Norm (BN) to allow aggressive updates
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), # The Secret Sauce of CrossQ
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, features):
        """
        Predict Q-values for all actions given visual features.
        """
        return self.q_net(features)

    def get_action(self, features, epsilon=0.1):
        """
        Selects action using Epsilon-Greedy policy.
        """
        # Exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Exploitation
        # CRITICAL FIX: Switch to eval mode to handle Batch Norm with size 1
        self.eval() 
        with torch.no_grad():
            q_values = self.forward(features)
            action = torch.argmax(q_values, dim=1).item()
        
        # Switch back to train mode so the network learns later
        self.train()
        
        return action

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        CrossQ Update Step.
        Note: We calculate target Q-values using the CURRENT network.
        """
        # 1. Predictions
        # Q(s, a)
        q_values = self.forward(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Targets (The CrossQ difference)
        # Standard DQN uses 'target_net'. We use 'self.q_net' directly.
        # This works because BatchNorm stabilizes the shifting distribution.
        with torch.no_grad():
            next_q_values = self.forward(next_states)
            max_next_q = next_q_values.max(1)[0]
            q_target = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # 3. Optimization
        loss = self.loss_fn(q_pred, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()