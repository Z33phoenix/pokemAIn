"""Tiny RAM-based DQN for menu control.

This module implements a minimal menu policy that operates purely on RAM-based
signals exposed via `PokemonRedGym._get_info` rather than full frames.

It is intended to be trained in a small micro-environment where episodes start
in (or just before) an interactive menu, and the goal is to move the cursor to
an arbitrary target index and confirm it.

Initially this is kept separate from the existing MenuBrain so we can iterate
quickly without breaking the hierarchical agent wiring. Once it is behaving
well, HierarchicalAgent can be updated to optionally use this policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TinyMenuConfig:
    """Configuration for the tiny menu DQN.

    This keeps hyperparameters local to the menu controller instead of
    overloading the global YAML config while we iterate.
    """

    state_dim: int = 4  # [menu_open, current_idx, target_idx, last_idx_norm]
    action_dim: int = 4  # [UP, DOWN, CONFIRM, CANCEL]
    hidden_dim: int = 64
    gamma: float = 0.95
    lr: float = 1e-3


class TinyMenuAgent(nn.Module):
    """Small MLP DQN operating on compact RAM features.

    State encoding (default):
        [0] menu_open      in {0,1}
        [1] current_idx    normalized to [0,1]
        [2] target_idx     normalized to [0,1]
        [3] last_idx_norm  (last_index / max_items) or 0 if unknown

    Actions are abstract indices; mapping to concrete PyBoy button presses is
    handled by the caller:
        0 -> move cursor up
        1 -> move cursor down
        2 -> confirm / A
        3 -> cancel / B
    """

    def __init__(self, cfg: TinyMenuConfig | None = None):
        super().__init__()
        self.cfg = cfg or TinyMenuConfig()

        self.net = nn.Sequential(
            nn.Linear(self.cfg.state_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.action_dim),
        )

        self.gamma = self.cfg.gamma
        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # State encoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def encode_state(info: Dict[str, Any], target_index: int, max_items: int | None = None) -> np.ndarray:
        """Return a compact state vector from the menu-related info dict.

        Args:
            info: env.info from PokemonRedGym._get_info.
            target_index: desired menu index [0, last_index].
            max_items: optional cap for normalization. If None, uses
                       (last_index + 1) from info when available.
        """
        menu_open = 1.0 if info.get("menu_open") and info.get("menu_has_options") else 0.0
        last_index = info.get("menu_last_index")
        if last_index is None or last_index < 0:
            last_index = 0
        num_items = last_index + 1
        if max_items is None:
            max_items = max(1, num_items)

        current_idx = info.get("menu_target", 0) or 0
        current_norm = float(current_idx) / float(max_items)
        target_norm = float(target_index) / float(max_items)
        last_idx_norm = float(last_index) / float(max_items)

        return np.array([menu_open, current_norm, target_norm, last_idx_norm], dtype=np.float32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    # ------------------------------------------------------------------
    # Policy & training
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, epsilon: float = 0.1, device: torch.device | None = None) -> int:
        """Epsilon-greedy action selection given a single state vector."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.cfg.action_dim)

        if device is None:
            device = next(self.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            q_values = self.forward(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        self.train()
        return action

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Standard DQN TD loss on a batch of transitions.

        This does the full optimizer step internally to keep it self-contained
        for now; if we later want to share optimizers/encoders we can refactor
        to mirror the specialists API.
        """
        # Q(s,a)
        q_values = self.forward(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.forward(next_states)
            max_next_q = next_q.max(1)[0]
            q_target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_error = (q_pred - q_target).detach()
        stats = {
            "menu_tiny/loss": loss.item(),
            "menu_tiny/td_abs_mean": td_error.abs().mean().item(),
            "menu_tiny/q_pred_mean": q_pred.detach().mean().item(),
        }
        return loss, stats
