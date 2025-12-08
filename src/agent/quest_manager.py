import json
import os
from typing import Any, Dict, List


class QuestManager:
    """
    Tracks high-level walkthrough "stages" and validates completion against
    simple RAM-derived conditions.
    """

    def __init__(self, walkthrough_path: str = "config/walkthrough_steps.json", save_dir: str = "saves"):
        self.steps = self._load_walkthrough(walkthrough_path)
        self.current_step_index = 0
        self.active_goal: Dict[str, Any] | None = None
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_walkthrough(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            # Fallback minimal progression if no walkthrough is provided.
            return [
                {"name": "1. START", "condition": {"type": "map_id", "value": 12}},  # Route 1
                {"name": "2. RETURN TO PALLET", "condition": {"type": "map_id", "value": 0}},  # Pallet Town
                {"name": "3. GET POKEDEX", "condition": {"type": "item", "value": "POKEDEX"}},
                {"name": "4. TO VIRIDIAN", "condition": {"type": "map_id", "value": 1}},  # Viridian City
            ]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_current_stage_name(self) -> str:
        if not self.steps:
            return "No Steps"
        return self.steps[self.current_step_index]["name"]

    def get_checkpoint_path(self) -> str:
        return os.path.join(self.save_dir, f"stage_{self.current_step_index}.state")

    def check_completion(self, state: Dict[str, Any]) -> bool:
        if not self.steps:
            return False
        current_step = self.steps[self.current_step_index]
        condition = current_step.get("condition", {})
        if self._satisfies_condition(state, condition):
            self.current_step_index = min(self.current_step_index + 1, len(self.steps) - 1)
            return True
        return False

    def _satisfies_condition(self, state: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        ctype = condition.get("type")
        value = condition.get("value")
        if ctype == "map_id":
            return state.get("location", {}).get("map_id") == value
        if ctype == "item":
            return value in (state.get("inventory", {}).get("key_items") or [])
        if ctype == "badge":
            return (state.get("game_state", {}).get("badges") or 0) >= value
        if ctype == "party_count":
            return len(state.get("party") or []) >= value
        return False
