import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agent.graph_memory import GraphMemory
from src.vision.encoder import NatureCNN


@dataclass
class Goal:
    """High-level directive chosen by the Director."""

    name: str
    goal_type: str
    priority: int
    target: Dict[str, Any]
    metadata: Dict[str, Any]
    max_steps: int
    goal_vector: Optional[list[float]] = None
    steps_spent: int = 0
    status: str = "pending"
    progress: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation of the goal."""
        return {
            "name": self.name,
            "goal_type": self.goal_type,
            "priority": self.priority,
            "target": self.target,
            "metadata": self.metadata,
            "max_steps": self.max_steps,
            "goal_vector": self.goal_vector,
            "steps_spent": self.steps_spent,
            "status": self.status,
            "progress": self.progress,
        }


class Director(nn.Module):
    """
    Shared vision encoder + routing logic that decides which specialist acts.

    Responsibilities:
    - Encode visual observations into shared latent features.
    - Maintain a lightweight graph of explored states for novelty estimation.
    - Manage exploration/battle goals and bias the router accordingly.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize shared encoder, router heads, goal tracking, and graph memory."""
        super().__init__()
        self.config = config
        hidden_dim = config.get("router_hidden_dim", 64)
        num_specialists = config.get("num_specialists", 2)

        self.encoder = NatureCNN()
        self.router = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_specialists),
        )

        graph_cfg = config.get("graph", {})
        self.graph = GraphMemory(
            max_nodes=graph_cfg.get("max_nodes", 5000),
            downsample_size=graph_cfg.get("downsample_size", 8),
            quantization_step=graph_cfg.get("quantization_step", 32),
        )

        self.goal_types = config.get("goal_head_types", ["explore", "train", "survive", "menu"])
        self.goal_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.goal_types)),
        )
        self.goal_specialist_map = config.get(
            "goal_specialist_map", {"explore": 0, "train": 1, "survive": 2, "menu": 2}
        )
        self.goal_llm_cfg = config.get("goal_llm", {})
        self.llm_enabled = bool(self.goal_llm_cfg.get("enabled", False))

        self.active_goal: Optional[Goal] = None
        self.goal_queue: List[Goal] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.goal_counter = 0
        self.backtracking_mode = False
        self.target_path: List[int] = []
        self.prev_battle_active = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns router logits alongside encoded features."""
        features = self.encoder(obs)
        logits = self.router(features)
        return logits, features

    def select_specialist(
        self, obs: torch.Tensor, info: Dict[str, Any], epsilon: float
    ) -> Tuple[int, torch.Tensor, Optional[int], Optional[Dict[str, Any]]]:
        """
        Returns (specialist_index, latent_features, forced_action).
        Forced action is populated when backtracking needs deterministic moves.
        """
        obs_cpu = obs.detach().cpu().numpy()
        state_hash, is_new_state = self.graph.update(obs_cpu, info, last_action=None)

        self._update_goal_progress(info, is_new_state)
        if self.active_goal is None and self.goal_queue:
            self._activate_goal(self.goal_queue.pop(0))

        features = self.encoder(obs)
        goal_logits = self.goal_head(features)
        desired_goal_type = self._determine_goal_type(info, goal_logits, epsilon)
        if self.active_goal is None and not self.llm_enabled:
            self._activate_goal(self._make_goal(desired_goal_type, info))
        elif self.active_goal is not None and not self.llm_enabled and self.active_goal.goal_type != desired_goal_type:
            self._activate_goal(self._make_goal(desired_goal_type, info))

        logits = self.router(features)
        logits = self._apply_goal_bias(logits)

        # If we already have an active goal, route directly to the mapped specialist
        # to keep control aligned with the current goal instead of re-deciding from
        # the frame alone.
        if self.active_goal:
            mapped_idx = self.goal_specialist_map.get(self.active_goal.goal_type)
            if mapped_idx is not None:
                return (
                    mapped_idx,
                    features,
                    None if not self.backtracking_mode else None,
                    self.active_goal.as_dict(),
                )

        if self.backtracking_mode and self.target_path:
            forced_action = self.target_path.pop(0)
            if not self.target_path:
                self.backtracking_mode = False
            return (
                0,
                features,
                forced_action,
                self.active_goal.as_dict() if self.active_goal else None,
            )

        if np.random.random() < epsilon:
            specialist_idx = np.random.randint(0, logits.shape[-1])
        else:
            specialist_idx = torch.argmax(logits, dim=1).item()

        return (
            specialist_idx,
            features,
            None,
            self.active_goal.as_dict() if self.active_goal else None,
        )

    # ------------------------------------------------------------------ #
    # Goal management
    # ------------------------------------------------------------------ #
    def _determine_goal_type(
        self, info: Dict[str, Any], goal_logits: torch.Tensor, epsilon: float
    ) -> str:
        """Pick a goal type using logits, epsilon-greedy exploration, and menu/battle guards."""
        if info.get("battle_active", False):
            return "train"
        if self._menu_active(info):
            return "menu"

        menu_open = self._menu_active(info)
        goal_type_list = self.goal_types or []

        if np.random.random() < epsilon and goal_type_list:
            candidates = [g for g in goal_type_list if g != "menu"] if not menu_open else goal_type_list
            return np.random.choice(candidates or goal_type_list)

        if goal_logits.numel() == 0 or not goal_type_list:
            return "explore"

        masked_logits = goal_logits.clone()
        if not menu_open and "menu" in goal_type_list:
            menu_idx = goal_type_list.index("menu")
            masked_logits[:, menu_idx] = float("-inf")

        goal_idx = torch.argmax(masked_logits, dim=1).item()
        if masked_logits[0, goal_idx].isinf():
            return "explore"

        goal_idx = max(0, min(goal_idx, len(goal_type_list) - 1))
        return goal_type_list[goal_idx]

    def _make_goal(self, goal_type: str, info: Dict[str, Any]) -> Goal:
        """Dispatch to a goal-construction helper based on goal_type."""
        if goal_type == "train":
            return self._make_train_goal(info)
        if goal_type == "survive":
            return self._make_survive_goal(info)
        if goal_type == "menu":
            return self._make_menu_goal(info)
        return self._make_exploration_goal(info)

    def _make_exploration_goal(self, info: Dict[str, Any]) -> Goal:
        """Build a novelty-seeking exploration goal."""
        goal_cfg = self.config.get("goals", {}).get("explore", {})
        goal = Goal(
            name=f"explore-{self.goal_counter}",
            goal_type="explore",
            priority=goal_cfg.get("priority", 0),
            target={"novel_states": goal_cfg.get("novel_states", 0)},
            metadata={"start_map": info.get("map_id"), "behavior": "explore"},
            max_steps=goal_cfg.get("max_steps", 0),
            goal_vector=None,
        )
        self.goal_counter += 1
        return goal

    def _make_train_goal(self, info: Dict[str, Any]) -> Goal:
        """Build a training goal oriented around battles won or XP gained."""
        goal_cfg = self.config.get("goals", {}).get("train", {})
        goal = Goal(
            name=f"train-{self.goal_counter}",
            goal_type="train",
            priority=goal_cfg.get("priority", 0),
            target={
                "xp_gained": goal_cfg.get("xp_gained", 0),
                "battles_won": goal_cfg.get("battles_won", 0),
            },
            metadata={"behavior": "train"},
            max_steps=goal_cfg.get("max_steps", 0),
            goal_vector=None,
        )
        self.goal_counter += 1
        return goal

    def _make_survive_goal(self, info: Dict[str, Any]) -> Goal:
        """Build a survivability goal targeting higher HP."""
        goal_cfg = self.config.get("goals", {}).get("survive", {})
        goal = Goal(
            name=f"survive-{self.goal_counter}",
            goal_type="survive",
            priority=goal_cfg.get("priority", 0),
            target={"hp_target": goal_cfg.get("hp_target", 0.0)},
            metadata={
                "hp_threshold": goal_cfg.get("hp_threshold", 0.0),
                "behavior": "survive",
            },
            max_steps=goal_cfg.get("max_steps", 0),
            goal_vector=None,
        )
        self.goal_counter += 1
        return goal

    def _make_menu_goal(self, info: Dict[str, Any]) -> Goal:
        """Build a deterministic menu-navigation goal using RAM cursor/target hints."""
        goal_cfg = self.config.get("goals", {}).get("menu", {})
        default_cursor = (
            goal_cfg.get("cursor_row"),
            goal_cfg.get("cursor_col"),
        )
        target_cursor = info.get("menu_cursor") or default_cursor
        goal = Goal(
            name=f"menu-{self.goal_counter}",
            goal_type="menu",
            priority=goal_cfg.get("priority", 0),
            target={
                "menu_target": info.get("menu_target", goal_cfg.get("menu_target")),
                "cursor": target_cursor,
                "menu_depth": info.get("menu_depth", goal_cfg.get("menu_depth")),
            },
            metadata={"behavior": "menu"},
            max_steps=goal_cfg.get("max_steps", 0),
            goal_vector=None,
        )
        self.goal_counter += 1
        return goal

    def _activate_goal(self, goal: Optional[Goal]):
        """Promote a goal to active status and reset its bookkeeping."""
        if goal is None:
            return
        goal.status = "active"
        goal.steps_spent = 0
        goal.progress = {}
        # Remove queue-only metadata when the goal is promoted to active.
        goal.metadata.pop("queue_expires_at", None)
        goal.metadata.pop("queued_at_step", None)
        self.active_goal = goal

    def _complete_active_goal(self, status: str):
        """Mark the active goal complete with a status and clear routing state."""
        if not self.active_goal:
            return
        self.active_goal.status = status
        self.completed_goals.append(self.active_goal.as_dict())
        self.active_goal = None
        self.backtracking_mode = False
        self.target_path.clear()

    def _update_goal_progress(self, info: Dict[str, Any], is_new_state: bool):
        """Update progress for the active goal and complete it when satisfied."""
        if not self.active_goal:
            return

        goal = self.active_goal
        goal.steps_spent += 1

        if goal.goal_type == "train":
            battle_active = info.get("battle_active", False)
            if self.prev_battle_active and not battle_active:
                goal.progress["battles_won"] = goal.progress.get("battles_won", 0) + 1
            target_battles = goal.target.get("battles_won")
            if target_battles and goal.progress.get("battles_won", 0) >= target_battles:
                self._complete_active_goal("succeeded")
                self.prev_battle_active = battle_active
                return
            if goal.steps_spent >= goal.max_steps:
                self._complete_active_goal("timeout")
                self.prev_battle_active = battle_active
                return
            self.prev_battle_active = battle_active
            return

        if goal.goal_type == "survive":
            hp_percent = info.get("hp_percent")
            target_hp = goal.target.get("hp_target")
            if hp_percent is not None and target_hp is not None and hp_percent >= target_hp:
                self._complete_active_goal("succeeded")
                return
            if goal.steps_spent >= goal.max_steps:
                self._complete_active_goal("timeout")
                return

        if goal.goal_type == "menu":
            current_menu = info.get("menu_target")
            desired = goal.target.get("menu_target")
            if desired is not None and current_menu == desired:
                self._complete_active_goal("succeeded")
                return
            if goal.steps_spent >= goal.max_steps:
                self._complete_active_goal("timeout")
                return

        if goal.goal_type == "explore":
            if is_new_state:
                goal.progress["novel_states"] = goal.progress.get("novel_states", 0) + 1
            if info.get("map_id") != goal.metadata.get("start_map"):
                goal.progress["new_map"] = True

            if goal.progress.get("novel_states", 0) >= goal.target.get("novel_states", 0):
                self._complete_active_goal("succeeded")
                return
            if goal.progress.get("new_map"):
                self._complete_active_goal("succeeded")
                return
            if goal.steps_spent >= goal.max_steps:
                self._complete_active_goal("timeout")
        self.prev_battle_active = info.get("battle_active", False)

    def _menu_active(self, info: Dict[str, Any]) -> bool:
        """Return True when RAM exposes an interactive (non-dialogue) menu."""
        menu_open = info.get("menu_open")
        has_options = info.get("menu_has_options")
        if menu_open is None and has_options is None:
            return False
        if menu_open is None:
            return bool(has_options)
        if has_options is None:
            return bool(menu_open)
        return bool(menu_open and has_options)

    def _apply_goal_bias(self, logits: torch.Tensor) -> torch.Tensor:
        """Boost router logits toward the specialist mapped to the active goal."""
        if self.active_goal is None:
            return logits
        bias_cfg = self.config.get("goal_bias", {})
        bias = torch.zeros_like(logits)
        goal_type = self.active_goal.goal_type
        target_idx = self.goal_specialist_map.get(goal_type)
        if target_idx is not None and target_idx < bias.shape[1]:
            bias[:, target_idx] += bias_cfg.get(goal_type, 0.0)
        return logits + bias

    def enqueue_goal(self, goal: Goal, current_step: Optional[int] = None):
        """Add a goal to the priority queue with an optional expiration step."""
        if current_step is not None and goal.max_steps:
            goal.metadata["queue_expires_at"] = current_step + int(goal.max_steps)
            goal.metadata["queued_at_step"] = current_step
        self.goal_queue.append(goal)
        self.goal_queue.sort(key=lambda g: g.priority, reverse=True)

    def prune_expired_goals(self, current_step: int) -> int:
        """Remove queued goals whose queue TTL has elapsed; returns count removed."""
        before = len(self.goal_queue)
        self.goal_queue = [
            g
            for g in self.goal_queue
            if g.metadata.get("queue_expires_at") is None or g.metadata["queue_expires_at"] > current_step
        ]
        return before - len(self.goal_queue)

    def get_last_completed_goal(self) -> Optional[Dict[str, Any]]:
        """Return the most recently completed goal dict, if any."""
        if not self.completed_goals:
            return None
        return self.completed_goals[-1]

    def get_goal_metrics(self) -> Dict[str, float]:
        """Expose metrics about the active/queued goal state for logging."""
        metrics = {
            "goal/has_active": 1.0 if self.active_goal else 0.0,
            "goal/type_explore": 0.0,
            "goal/type_train": 0.0,
            "goal/type_survive": 0.0,
            "goal/type_menu": 0.0,
            "goal/steps": 0.0,
            "goal/priority": 0.0,
            "goal/queue_len": float(len(self.goal_queue)),
        }
        if self.active_goal:
            goal_type = self.active_goal.goal_type
            key = f"goal/type_{goal_type}"
            if key in metrics:
                metrics[key] = 1.0
            metrics["goal/steps"] = float(self.active_goal.steps_spent)
            metrics["goal/priority"] = float(self.active_goal.priority)
        return metrics
