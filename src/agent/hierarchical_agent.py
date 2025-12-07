from typing import Any, Dict, Tuple, Optional

import os
import torch
from pyboy.utils import WindowEvent

from src.agent.director import Director
from src.agent.specialists.nav_brain import CrossQNavAgent
from src.agent.specialists.battle_brain import CrossQBattleAgent
from src.agent.specialists.menu_brain import MenuBrain


WINDOW_EVENT_TO_ACTION_ID = {
    WindowEvent.PRESS_ARROW_DOWN: 0,
    WindowEvent.PRESS_ARROW_LEFT: 1,
    WindowEvent.PRESS_ARROW_RIGHT: 2,
    WindowEvent.PRESS_ARROW_UP: 3,
    WindowEvent.PRESS_BUTTON_A: 4,
    WindowEvent.PRESS_BUTTON_B: 5,
    WindowEvent.PRESS_BUTTON_START: 6,
    WindowEvent.PRESS_BUTTON_SELECT: 7,
}


class HierarchicalAgent:
    """Thin orchestrator that wires the Director and specialists together."""

    def __init__(
        self,
        action_dim: int,
        device: torch.device | str,
        director_cfg: Dict[str, Any],
        specialist_cfg: Dict[str, Any],
    ):
        """Wire up director and specialist heads with allowed action mappings."""
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
        self.battle_brain = CrossQBattleAgent(battle_cfg).to(self.device)
        self.menu_brain = MenuBrain(menu_cfg).to(self.device)
        self.menu_open_action = self._resolve_menu_open_action()
        self.action_dim = action_dim

    def _zero_goal_embedding(self, dim: int) -> torch.Tensor:
        """Return a zero goal embedding of the requested dimension on the agent device."""
        return torch.zeros((1, dim), device=self.device)

    def _encode_menu_goal(self, goal_ctx: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Normalize menu goal context into an embedding for the specialist."""
        if goal_ctx is None:
            return self._zero_goal_embedding(self.menu_brain.goal_dim)
        return self.menu_brain.encode_goal(goal_ctx, device=self.device)

    @staticmethod
    def _info_menu_active(info: Dict[str, Any]) -> bool:
        """Return True when info reports an interactive menu with options."""
        menu_open = info.get("menu_open")
        has_options = info.get("menu_has_options")
        if menu_open is None and has_options is None:
            return False
        if menu_open is None:
            return bool(has_options)
        if has_options is None:
            return bool(menu_open)
        return bool(menu_open and has_options)

    def _resolve_menu_open_action(self) -> int | None:
        """Return the local menu action index that maps to pressing START, if available."""
        start_action_id = WINDOW_EVENT_TO_ACTION_ID.get(WindowEvent.PRESS_BUTTON_START)
        if start_action_id is None:
            return None
        return self.menu_action_lookup.get(start_action_id)

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
                goal_ctx["metadata"]["menu_open"] = self._info_menu_active(info)
                target = goal_ctx.get("target", {}) or {}
                if "cursor" not in target and info.get("menu_cursor") is not None:
                    target = target.copy()
                    target["cursor"] = info.get("menu_cursor")
                    goal_ctx["target"] = target
                action_meta["goal"] = goal_ctx
            goal_embedding = self._encode_menu_goal(goal_ctx)
            menu_open = self._info_menu_active(info)
            open_idx = self.menu_open_action
            local_action = self.menu_brain.get_action(
                features,
                goal_embedding,
                epsilon=min(epsilon, 0.2),
                menu_open=menu_open,
                open_action_index=open_idx,
            )
            action = self.menu_actions[local_action]
            action_meta["local_action"] = local_action
            action_meta["goal_embedding"] = goal_embedding.detach()
        return action, specialist_idx, action_meta

    def save(self, checkpoint_dir: str = "checkpoints", tag: str = "latest"):
        """Persist all components to disk under the provided tag."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.director.state_dict(), os.path.join(checkpoint_dir, f"director_{tag}.pth"))
        torch.save(self.nav_brain.state_dict(), os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth"))
        torch.save(self.battle_brain.state_dict(), os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth"))
        torch.save(self.menu_brain.state_dict(), os.path.join(checkpoint_dir, f"menu_brain_{tag}.pth"))

    def load(self, checkpoint_dir: str = "checkpoints", tag: str = "latest"):
        """Load components from disk if checkpoint files exist."""
        def _safe_load(path: str):
            """Safely load a state_dict if the checkpoint file exists."""
            if not os.path.exists(path):
                return None
            load_kwargs = {"map_location": self.device}
            try:
                return torch.load(path, weights_only=True, **load_kwargs)
            except TypeError:
                return torch.load(path, **load_kwargs)

        dir_path = os.path.join(checkpoint_dir, f"director_{tag}.pth")
        nav_path = os.path.join(checkpoint_dir, f"nav_brain_{tag}.pth")
        bat_path = os.path.join(checkpoint_dir, f"battle_brain_{tag}.pth")
        menu_path = os.path.join(checkpoint_dir, f"menu_brain_{tag}.pth")
        director_state = _safe_load(dir_path)
        if director_state is not None:
            self.director.load_state_dict(director_state)
        nav_state = _safe_load(nav_path)
        if nav_state is not None:
            self.nav_brain.load_state_dict(nav_state)
        battle_state = _safe_load(bat_path)
        if battle_state is not None:
            self.battle_brain.load_state_dict(battle_state)
        menu_state = _safe_load(menu_path)
        if menu_state is not None:
            self.menu_brain.load_state_dict(menu_state)

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
