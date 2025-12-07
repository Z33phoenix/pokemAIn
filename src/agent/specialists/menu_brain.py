from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MenuBrain(nn.Module):
    """
    DQN-style controller for deterministic menu navigation.

    Menus have far less visual variance than overworld scenes, so this model
    focuses on learning consistent cursor motions toward requested goals while
    allowing the Director to specify which submenu/item should be reached.
    """

    def __init__(self, config: Dict[str, Any], input_dim: int = 512):
        super().__init__()
        self.goal_dim = config.get("goal_dim", 8)
        self.gamma = config.get("gamma", 0.99)
        self.action_names = config.get(
            "actions",
            ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"],
        )
        self.action_dim = len(self.action_names)

        hidden_dim = config.get("hidden_dim", 256)
        self.goal_encoding_cfg = config.get("goal_encoding", {})
        self.goal_defaults = config.get("goal_defaults", {})
        self.q_net = nn.Sequential(
            nn.Linear(input_dim + self.goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

        self.optimizer = optim.AdamW(
            self.q_net.parameters(), lr=config.get("learning_rate", 1e-4)
        )
        self.loss_fn = nn.MSELoss()

    def encode_goal_batch(
        self, goals: list[Optional[Dict[str, Any]]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Convert a batch of goal context dictionaries into dense embeddings.

        Each vector packs the requested menu target, cursor position, and depth
        normalized by scales defined in config to avoid magic constants.
        """
        device = device or self.q_net[0].weight.device
        batch = len(goals)
        goal_vec = torch.zeros((batch, self.goal_dim), device=device)

        target_scale = self.goal_encoding_cfg.get("menu_target_scale", 1.0) or 1.0
        row_scale = self.goal_encoding_cfg.get("cursor_row_scale", 1.0) or 1.0
        col_scale = self.goal_encoding_cfg.get("cursor_col_scale", 1.0) or 1.0
        depth_scale = self.goal_encoding_cfg.get("depth_scale", 1.0) or 1.0
        open_value = self.goal_encoding_cfg.get("open_flag_value", 1.0)

        for i, ctx in enumerate(goals):
            if not ctx or ctx.get("goal_type") != "menu":
                continue
            target = ctx.get("target", {}) or {}
            menu_target = target.get("menu_target", self.goal_defaults.get("menu_target"))
            cursor = target.get("cursor")
            if cursor is None:
                cursor = (
                    self.goal_defaults.get("cursor_row"),
                    self.goal_defaults.get("cursor_col"),
                )
            menu_depth = target.get("menu_depth", self.goal_defaults.get("menu_depth"))
            if menu_target is not None:
                goal_vec[i, 0] = float(menu_target) / target_scale
            if self.goal_dim > 1 and cursor is not None and cursor[0] is not None:
                goal_vec[i, 1] = float(cursor[0]) / row_scale
            if self.goal_dim > 2 and cursor is not None and cursor[1] is not None:
                goal_vec[i, 2] = float(cursor[1]) / col_scale
            if self.goal_dim > 3 and menu_depth is not None:
                goal_vec[i, 3] = float(menu_depth) / depth_scale
            if self.goal_dim > 4 and ctx.get("metadata", {}).get("menu_open"):
                goal_vec[i, 4] = float(open_value)
        return goal_vec

    def encode_goal(
        self, goal_ctx: Optional[Dict[str, Any]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Single-sample convenience wrapper around encode_goal_batch."""
        return self.encode_goal_batch([goal_ctx], device=device)

    def _prepare_goal(self, goal_embedding: Optional[torch.Tensor], batch: int) -> torch.Tensor:
        if goal_embedding is None:
            return torch.zeros((batch, self.goal_dim), device=self.q_net[0].weight.device)
        return goal_embedding

    def forward(
        self, features: torch.Tensor, goal_embedding: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenates the latent frame features with a goal embedding."""
        goal_vec = self._prepare_goal(goal_embedding, features.size(0))
        q_input = torch.cat([features, goal_vec], dim=1)
        return self.q_net(q_input)

    def get_action(
        self,
        features: torch.Tensor,
        goal_embedding: Optional[torch.Tensor],
        epsilon: float = 0.1,
        menu_open: bool = True,
        open_action_index: Optional[int] = None,
    ) -> int:
        """
        Epsilon-greedy selection: deterministic when the Director demands a
        precise menu target, exploratory otherwise. If the menu is closed and
        an explicit open action is provided, force it so training stays inside
        menu contexts.
        """
        if not menu_open and open_action_index is not None:
            return open_action_index

        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        self.eval()
        with torch.no_grad():
            q_values = self.forward(features, goal_embedding)
            action = torch.argmax(q_values, dim=1).item()
        self.train()
        return action

    def train_step(
        self,
        features: torch.Tensor,
        goal_embedding: Optional[torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_features: torch.Tensor,
        next_goal_embedding: Optional[torch.Tensor],
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Standard DQN TD loss with goal-conditioned inputs.

        Note: This function no longer performs optimizer steps; callers should
        handle zero_grad/backward/step so the shared encoder can update
        alongside the menu network (consistent with other specialists).
        """
        q_values = self.forward(features, goal_embedding)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.forward(next_features, next_goal_embedding)
            max_next_q = next_q_values.max(1)[0]
            q_target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_pred, q_target)
        stats = {
            "menu/loss": loss.item(),
            "menu/q_pred_mean": q_pred.detach().mean().item(),
        }
        return loss, stats

    # ------------------------------------------------------------------ #
    # Placeholder helper methods to be implemented with OCR/template logic
    # ------------------------------------------------------------------ #
    def detect_cursor_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Returns the (x, y) position of the cursor in menu space.
        Placeholder: implement template matching / OCR later.
        """
        return None

    def read_menu_text(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts textual information (Bag, Pokémon, Save, etc.) from the menu.
        Placeholder: hook up OCR or glyph matching in a later pass.
        """
        return {}

    def compute_menu_reward(
        self,
        goal: Dict[str, Any],
        cursor_pos: Optional[Tuple[int, int]],
        menu_state: Dict[str, Any],
        stuck_steps: int,
    ) -> float:
        """
        Placeholder reward signal:
        - Positive when the requested submenu/item is highlighted or confirmed.
        - Negative when repeatedly toggling without progress or closing menus.
        """
        reward_cfg = self.goal_encoding_cfg.get("reward_weights", {})
        progress_reward = reward_cfg.get("correct_target", 0.0) if menu_state.get("current_menu") == goal.get("target") else 0.0
        cursor_reward = reward_cfg.get("cursor_on_target", 0.0) if cursor_pos == goal.get("cursor") else 0.0
        stale_threshold = goal.get("stale_threshold", reward_cfg.get("stale_threshold", 0))
        stuck_penalty = reward_cfg.get("stale_penalty", 0.0) if stale_threshold and stuck_steps > stale_threshold else 0.0
        premature_close = reward_cfg.get("close_penalty", 0.0) if menu_state.get("closed", False) else 0.0
        return progress_reward + cursor_reward + stuck_penalty + premature_close
