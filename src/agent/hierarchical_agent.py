from typing import Any, Dict, Tuple, Optional

import os
import torch

from src.agent.director import Director
from src.agent.specialists.nav_brain import CrossQNavAgent
from src.agent.specialists.battle_brain import RainbowBattleAgent
from src.agent.specialists.menu_brain import MenuBrain


class HierarchicalAgent:
    """Thin orchestrator that wires the Director and specialists together."""

    def __init__(
        self,
        action_dim: int,
        device: torch.device | str,
        director_cfg: Dict[str, Any],
        specialist_cfg: Dict[str, Any],
    ):
        self.device = torch.device(device)
        self.director = Director(director_cfg).to(self.device)

        nav_cfg = specialist_cfg["navigation"].copy()
        battle_cfg = specialist_cfg["battle"].copy()
        menu_cfg = specialist_cfg["menu"].copy()

        self.nav_actions = nav_cfg.pop("allowed_actions", list(range(action_dim)))
        nav_cfg["action_dim"] = len(self.nav_actions)
        self.nav_action_lookup = {act: idx for idx, act in enumerate(self.nav_actions)}

        self.battle_actions = battle_cfg.pop("allowed_actions", list(range(action_dim)))
        battle_cfg["action_dim"] = len(self.battle_actions)
        self.battle_action_lookup = {
            act: idx for idx, act in enumerate(self.battle_actions)
        }

        self.menu_actions = menu_cfg.pop("allowed_actions", list(range(action_dim)))
        menu_cfg["action_dim"] = len(self.menu_actions)
        self.menu_action_lookup = {
            act: idx for idx, act in enumerate(self.menu_actions)
        }

        self.nav_brain = CrossQNavAgent(nav_cfg).to(self.device)
        self.battle_brain = RainbowBattleAgent(battle_cfg).to(self.device)
        self.menu_brain = MenuBrain(menu_cfg).to(self.device)
        self.action_dim = action_dim

    def _zero_goal_embedding(self, dim: int) -> torch.Tensor:
        return torch.zeros((1, dim), device=self.device)

    def _encode_menu_goal(self, goal_ctx: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Normalize menu goal context into an embedding for the specialist."""
        if goal_ctx is None:
            return self._zero_goal_embedding(self.menu_brain.goal_dim)
        return self.menu_brain.encode_goal(goal_ctx, device=self.device)

    def get_action(
        self, obs, info: Dict[str, Any], epsilon: float
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Returns (action, specialist_index, metadata)."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        specialist_idx, features, forced_action, goal_ctx = self.director.select_specialist(
            obs_t, info, epsilon
        )
        action_meta = {"local_action": None, "goal": goal_ctx}

        if forced_action is not None:
            action_meta["local_action"] = self.nav_action_lookup.get(forced_action)
            if action_meta["local_action"] is None:
                raise ValueError(f"Forced action {forced_action} not allowed for navigation specialist")
            return forced_action, specialist_idx, action_meta

        if specialist_idx == 0:
            local_action = self.nav_brain.get_action(features, epsilon, goal_ctx)
            action = self.nav_actions[local_action]
            action_meta["local_action"] = local_action
        elif specialist_idx == 1:
            local_action = self.battle_brain.get_action(features, goal_ctx)
            action = self.battle_actions[local_action]
            action_meta["local_action"] = local_action
        else:
            if goal_ctx and goal_ctx.get("goal_type") == "menu":
                goal_ctx = goal_ctx.copy()
                goal_ctx.setdefault("metadata", {})
                goal_ctx["metadata"]["menu_open"] = info.get("menu_open", False)
                target = goal_ctx.get("target", {}) or {}
                if "cursor" not in target and info.get("menu_cursor") is not None:
                    target = target.copy()
                    target["cursor"] = info.get("menu_cursor")
                    goal_ctx["target"] = target
                action_meta["goal"] = goal_ctx
            goal_embedding = self._encode_menu_goal(goal_ctx)
            local_action = self.menu_brain.get_action(features, goal_embedding)
            action = self.menu_actions[local_action]
            action_meta["local_action"] = local_action
            action_meta["goal_embedding"] = goal_embedding.detach()
        return action, specialist_idx, action_meta

    def save(self, checkpoint_dir: str = "checkpoints", tag: str = "latest"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.director.state_dict(), os.path.join(checkpoint_dir, f"director_{tag}.pth"))
        torch.save(self.nav_brain.state_dict(), os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth"))
        torch.save(self.battle_brain.state_dict(), os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth"))
        torch.save(self.menu_brain.state_dict(), os.path.join(checkpoint_dir, f"menu_brain_{tag}.pth"))

    def load(self, checkpoint_dir: str = "checkpoints", tag: str = "latest"):
        dir_path = os.path.join(checkpoint_dir, f"director_{tag}.pth")
        nav_path = os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth")
        bat_path = os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth")
        menu_path = os.path.join(checkpoint_dir, f"menu_brain_{tag}.pth")
        if os.path.exists(dir_path):
            self.director.load_state_dict(torch.load(dir_path, map_location=self.device))
        if os.path.exists(nav_path):
            self.nav_brain.load_state_dict(torch.load(nav_path, map_location=self.device))
        if os.path.exists(bat_path):
            self.battle_brain.load_state_dict(torch.load(bat_path, map_location=self.device))
        if os.path.exists(menu_path):
            self.menu_brain.load_state_dict(torch.load(menu_path, map_location=self.device))

    def save_component(self, component: str, checkpoint_dir: str = "checkpoints", tag: str = "latest"):
        """Save a single module so we can cherry-pick the best brains across runs."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        component = component.lower()
        if component == "director":
            torch.save(self.director.state_dict(), os.path.join(checkpoint_dir, f"director_{tag}.pth"))
        elif component == "nav":
            torch.save(self.nav_brain.state_dict(), os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth"))
        elif component == "battle":
            torch.save(self.battle_brain.state_dict(), os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth"))
        elif component == "menu":
            torch.save(self.menu_brain.state_dict(), os.path.join(checkpoint_dir, f"menu_brain_{tag}.pth"))
        else:
            raise ValueError(f"Unknown component '{component}'. Expected director/nav/battle/menu.")
