from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for 'Beyond the Rainbow'.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowBattleAgent(nn.Module):
    """Distributional battle specialist with NoisyLinear exploration."""

    def __init__(self, config: Dict[str, float], input_dim: int = 512):
        super().__init__()
        self.action_dim = config.get("action_dim", 8)
        self.gamma = config.get("gamma", 0.99)
        self.atoms = config.get("atoms", 51)
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        self.register_buffer("supports", torch.linspace(self.v_min, self.v_max, self.atoms))

        self.feature_layer = nn.Sequential(NoisyLinear(input_dim, 512), nn.ReLU())
        self.value_stream = nn.Sequential(
            NoisyLinear(512, 128),
            nn.ReLU(),
            NoisyLinear(128, self.atoms),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 128),
            nn.ReLU(),
            NoisyLinear(128, self.action_dim * self.atoms),
        )

        self.optimizer = optim.AdamW(
            self.parameters(), lr=config.get("learning_rate", 1e-4)
        )

    def forward(self, features):
        """
        Returns: Distribution (Batch, Actions, Atoms)
        """
        x = self.feature_layer(features)
        
        val = self.value_stream(x).view(-1, 1, self.atoms)
        adv = self.advantage_stream(x).view(-1, self.action_dim, self.atoms)
        
        # Dueling Combination
        q_dist = val + adv - adv.mean(dim=1, keepdim=True)
        q_probs = F.softmax(q_dist, dim=2) 
        
        return q_probs

    def get_action(
        self, features: torch.Tensor, goal: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Selects action based on Expected Value (Mean of distribution).
        """
        if self.training:
            # Resample noise for exploration every step
            for m in self.modules():
                if isinstance(m, NoisyLinear): m.reset_noise()
                
        with torch.no_grad():
            q_probs = self.forward(features) # (1, Actions, Atoms)
            # Use self.supports (now a buffer)
            expected_value = (q_probs * self.supports).sum(dim=2)
            return torch.argmax(expected_value, dim=1).item()

    def train_step_return_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_features: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Distributional RL Loss (C51 / Categorical).
        Returns LOSS TENSOR (for End-to-End Backprop).
        """
        batch_size = features.size(0)

        with torch.no_grad():
            next_probs = self.forward(next_features)
            next_expected = (next_probs * self.supports).sum(dim=2)
            next_actions = next_expected.argmax(dim=1) 
            
            # Get distribution of the best next action
            next_dist = next_probs[range(batch_size), next_actions]
            
            # Project the distribution (Bellman Update)
            t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * self.supports.unsqueeze(0)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            
            # Projection Logic (Categorical Algorithm)
            delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            target_dist = torch.zeros_like(next_dist)
            for i in range(batch_size):
                for j in range(self.atoms):
                    b_val = b[i, j].item()
                    lower = int(torch.clamp(l[i, j], 0, self.atoms - 1).item())
                    upper = int(torch.clamp(u[i, j], 0, self.atoms - 1).item())
                    prob = next_dist[i, j]
                    if lower == upper:
                        target_dist[i, lower] += prob
                    else:
                        target_dist[i, lower] += prob * (upper - b_val)
                        target_dist[i, upper] += prob * (b_val - lower)

        # 2. Prediction
        dist = self.forward(features)
        action_dist = dist[range(batch_size), actions]
        
        # 3. Loss (KL Divergence / Cross Entropy)
        # We add 1e-6 to avoid log(0)
        loss = -torch.sum(target_dist * torch.log(action_dist + 1e-6)) / batch_size
        stats = {
            "battle/loss": loss.item(),
            "battle/entropy": -torch.sum(action_dist * torch.log(action_dist + 1e-6))
            .detach()
            .item()
            / batch_size,
        }
        return loss, stats
