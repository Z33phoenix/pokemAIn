from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CrossQBattleAgent(nn.Module):
    """Battle specialist matching the lightweight CrossQ architecture used for navigation."""

    def __init__(self, config: Dict[str, float], input_dim: int = 512):
        """Initialize Q-network, optimizer, and hyperparameters for battles."""
        super().__init__()
        self.action_dim = config.get("action_dim", 8)
        self.gamma = config.get("gamma", 0.99)

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

        self.optimizer = optim.AdamW(
            self.q_net.parameters(), lr=config.get("learning_rate", 1e-4)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predicts Q-values for every action."""
        return self.q_net(features)

    def get_action(
        self,
        features: torch.Tensor,
        goal: Optional[Dict[str, Any]] = None,
        epsilon: float = 0.1,
    ) -> int:
        """Epsilon-greedy discrete policy (goal currently unused)."""
        preferred_action = None
        if goal:
            goal_type = goal.get("goal_type")
            metadata = goal.get("metadata", {}) or {}
            preferred_action = metadata.get("preferred_action")
            if preferred_action is None:
                preferred_action = metadata.get("safe_action")
            if preferred_action is None:
                preferred_action = goal.get("target", {}).get("action")
            if goal_type == "survive":
                epsilon = min(epsilon, 0.05)
            elif goal_type == "train":
                epsilon = min(epsilon, 0.1)

        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        self.eval()
        with torch.no_grad():
            q_values = self.forward(features)
            if preferred_action is not None and 0 <= int(preferred_action) < self.action_dim:
                bias = torch.zeros_like(q_values)
                bias[:, int(preferred_action)] += 0.05
                q_values = q_values + bias
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
            "battle/td_abs_mean": td_error.abs().mean().item(),
            "battle/q_pred_mean": q_pred.detach().mean().item(),
            "battle/loss": loss.item(),
        }
        return loss, stats
