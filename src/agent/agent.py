from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CrossQNavAgent(nn.Module):
    """Navigation specialist implementing CrossQ without a target network."""

    def __init__(self, config: Dict[str, float], input_dim: int = 512):
        """Initialize Q-network, optimizer, and hyperparameters for navigation."""
        super().__init__()
        self.action_dim = config.get("action_dim", 8)
        input_dim = int(config.get("input_dim", input_dim))
        self.gamma = config.get("gamma", 0.99)

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

        self.optimizer = optim.AdamW(
            self.q_net.parameters(), lr=config.get("learning_rate", 1e-4)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict Q-values for every action."""
        return self.q_net(features)

    def get_action(
        self,
        features: torch.Tensor,
        epsilon: float = 0.1,
        goal: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Epsilon-greedy discrete policy."""
        if goal:
            goal_type = goal.get("goal_type")
            if goal_type == "survive":
                epsilon = min(epsilon, 0.05)
            elif goal_type == "train":
                epsilon = min(epsilon, 0.2)
            elif goal_type and goal_type != "explore":
                epsilon = min(epsilon, 0.2)

        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        self.eval()
        with torch.no_grad():
            q_values = self.forward(features)
            action = torch.argmax(q_values, dim=1).item()
        self.train()
        return action

    def train_step_return_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_features: torch.Tensor,
        dones: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        next_goal: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the TD loss but leaves the optimizer step to the caller so the
        Director's encoder gradients can flow end-to-end.
        """
        q_values = self.forward(features)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.forward(next_features)
            max_next_q = next_q_values.max(1)[0]
            q_target = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(q_pred, q_target)
        td_error = (q_pred - q_target).detach()
        stats = {
            "agent/td_abs_mean": td_error.abs().mean().item(),
            "agent/q_pred_mean": q_pred.detach().mean().item(),
            "agent/loss": loss.item(),
        }
        return loss, stats
