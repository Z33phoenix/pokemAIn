import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CrossQNavAgent(nn.Module):
    """
    Navigation Specialist using CrossQ (Batch-Norm DDPG/DQN style).
    
    UPDATED FOR END-TO-END TRAINING:
    - Input: Pre-processed Feature Vectors (512 dim), NOT Images.
    - Output: Loss Tensor (for external backward pass).
    """
    def __init__(self, input_dim=512, action_dim=8, lr=1e-4, gamma=0.99):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        
        # 1. The Q-Network
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
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
        Predict Q-values given visual features (from Director).
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
        self.eval() 
        with torch.no_grad():
            q_values = self.forward(features)
            action = torch.argmax(q_values, dim=1).item()
        
        self.train()
        return action

    def train_step_return_loss(self, features, actions, rewards, next_features, dones):
        """
        Calculates loss but DOES NOT optimize.
        Returns the Loss Tensor so the outer loop can backpropagate through 
        both this network AND the Director's Encoder.
        
        Args:
            features: Tensor (B, 512) - Output from Director.vision(state)
            next_features: Tensor (B, 512) - Output from Director.vision(next_state)
        """
        # 1. Predictions
        q_values = self.forward(features)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Targets
        # Note: We detach next_features because we don't need gradients for the target
        with torch.no_grad():
            next_q_values = self.forward(next_features)
            max_next_q = next_q_values.max(1)[0]
            q_target = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # 3. Calculate Loss
        loss = self.loss_fn(q_pred, q_target)
        
        # 4. Metrics
        td_error = (q_pred - q_target).detach()
        stats = {
            "nav/td_abs_mean": td_error.abs().mean().item(),
            "nav/q_pred_mean": q_pred.detach().mean().item(),
            "nav/loss": loss.item()
        }
        
        # CRITICAL: Return the Tensor, not the item.
        return loss, stats