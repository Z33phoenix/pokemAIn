import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyboy.utils import WindowEvent

from src.agent.graph_memory import GraphMemory
from src.utils.game_data import map_id_to_name, map_name_to_id, pokemon_id_to_name


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


class LLMGoalCoordinator:
    """Utility helpers for building/parsing text-only LLM goal messages."""

    @staticmethod
    def build_state_summary(info: Dict[str, Any], reward_sys) -> Dict[str, Any]:
        map_id = info.get("map_id")
        quest_flags = info.get("quest_flags", {}) or {}
        badges = info.get("badges", {}) or {}

        key_items = list(info.get("key_items", []) or [])
        if quest_flags.get("oak_parcel") and "OAK'S PARCEL" not in key_items:
            key_items.append("OAK'S PARCEL")
        if quest_flags.get("town_map") and "TOWN MAP" not in key_items:
            key_items.append("TOWN MAP")
        if quest_flags.get("ss_anne") and "S.S. TICKET" not in key_items:
            key_items.append("S.S. TICKET")
        if quest_flags.get("mewtwo") and "MYSTERIOUS ITEM" not in key_items:
            key_items.append("MYSTERIOUS ITEM")

        # Prefer a full party summary if provided; otherwise derive from active mon.
        party = []
        if info.get("party"):
            party = info.get("party")
        else:
            hp_cur = info.get("hp_current")
            hp_max = info.get("hp_max")
            species_id = info.get("active_pokemon_id") or info.get("party_species_id") or info.get("species_id")
            species_name = pokemon_id_to_name(species_id)
            status = "FAINTED" if hp_cur is not None and hp_cur <= 0 else "OK"
            if hp_cur is not None and hp_max is not None and hp_max > 0:
                party.append(
                    {
                        "species": species_name,
                        "hp": int(hp_cur),
                        "max_hp": int(hp_max),
                        "level": info.get("active_pokemon_level"),
                        "status": status,
                    }
                )

        battle_status = "Battle" if info.get("battle_active") else "Overworld"
        last_goal_entry = info.get("last_goal")
        if not last_goal_entry:
            last_goal = getattr(reward_sys, "last_completed_goal", None)
            if last_goal:
                last_goal_entry = {
                    "target": last_goal.get("target") or last_goal.get("name"),
                    "status": last_goal.get("status"),
                }
            else:
                # Seed the very first request with an explicit starting goal for clarity.
                last_goal_entry = {"target": "Start Game", "status": "New"}

        return {
            "location": {
                "map_name": map_id_to_name(map_id),
                "map_id": map_id,
                "x": info.get("x"),
                "y": info.get("y"),
                "nearby_sprites": info.get("nearby_sprites", []),
            },
            "party": party,
            "inventory": {
                "key_items": key_items,
                "hms_owned": info.get("hms_owned", []),
                "items": info.get("items", []),
            },
            "game_state": {
                "badges": info.get("badge_count", 0),
                "money": info.get("money", 0),
                "battle_status": battle_status,
            },
            "last_goal": last_goal_entry,
        }

    @staticmethod
    def sanitize_goal_response(
        goal_json: Dict[str, Any], goal_defaults: Dict[str, Any], current_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        goal_type = goal_json.get("goal_type") or "NAVIGATE"
        target_map_name = goal_json.get("target_map_name") or goal_json.get("target_location_name")
        target_map_id = goal_json.get("target_map_id")
        if target_map_id is None:
            target_map_id = map_name_to_id(target_map_name)

        target = {
            "map_id": target_map_id,
            "map_name": target_map_name,
            "required_item": goal_json.get("required_item"),
            "description": goal_json.get("action_description"),
        }
        metadata = {"reason": goal_json.get("thought_process")}
        if current_info.get("map_id") is not None:
            metadata["start_map"] = current_info.get("map_id")

        defaults = goal_defaults.get("explore", {})
        return {
            "goal_type": goal_type,
            "priority": int(defaults.get("priority", 1) or 1),
            "target": target,
            "metadata": metadata,
            "max_steps": int(defaults.get("max_steps", 256) or 256),
            "name": goal_json.get("name") or target_map_name or "llm-goal",
        }


class Director:
    """Lightweight, non-neural coordinator that always routes to the navigation brain."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        graph_cfg = config.get("graph", {})
        self.graph = GraphMemory(
            max_nodes=graph_cfg.get("max_nodes", 5000),
            downsample_size=graph_cfg.get("downsample_size", 8),
            quantization_step=graph_cfg.get("quantization_step", 32),
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

    def encode_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Flatten normalized observation tensor into a feature vector."""
        if obs.dim() == 4:
            obs = obs.squeeze(0)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs.view(1, -1).to(torch.float32)

    def encode_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Flatten a batch of observations."""
        if obs_batch.max() > 1.0:
            obs_batch = obs_batch / 255.0
        return obs_batch.view(obs_batch.size(0), -1).to(torch.float32)

    def select_specialist(
        self, obs: torch.Tensor, info: Dict[str, Any], epsilon: float
    ) -> Tuple[int, torch.Tensor, Optional[int], Optional[Dict[str, Any]]]:
        """Returns (specialist_index, features, forced_action, goal_ctx)."""
        obs_cpu = obs.detach().cpu().numpy()
        self.graph.update(obs_cpu, info, last_action=None)

        if self.active_goal is None and self.goal_queue:
            self._activate_goal(self.goal_queue.pop(0))

        features = self.encode_features(obs)
        return (
            0,  # always navigation specialist
            features,
            None,
            self.active_goal.as_dict() if self.active_goal else None,
        )

    # Goal management (kept minimal for compatibility)
    def _activate_goal(self, goal: Goal):
        self.active_goal = goal
        self.active_goal.status = "active"

    def enqueue_goal(self, goal: Goal, current_step: int = 0):
        self.goal_queue.append(goal)
        self.goal_queue.sort(key=lambda g: g.priority, reverse=True)

    def complete_goal(self, status: str = "done"):
        if self.active_goal:
            self.active_goal.status = status
            self.completed_goals.append(self.active_goal.as_dict())
            self.active_goal = None

    def prune_expired_goals(self, current_step: int = 0):
        # No-op placeholder; can be expanded to drop stale goals.
        return

    def get_last_completed_goal(self) -> Optional[Dict[str, Any]]:
        if not self.completed_goals:
            return None
        return self.completed_goals[-1]

    def get_goal_metrics(self) -> Dict[str, float]:
        return {
            "goal/active": 1 if self.active_goal else 0,
            "goal/queue_len": len(self.goal_queue),
            "goal/completed": len(self.completed_goals),
        }


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
