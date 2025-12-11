"""
Goal Strategy - Abstract interface for goal-setting strategies.

This module defines the pluggable goal-setting architecture, allowing you to easily
swap between different goal-setting approaches (LLM-based, heuristic, none, etc.).

Architecture:
    GoalStrategy (abstract) - defines the interface
    ├── LLMGoalStrategy - uses LLM to set intelligent goals
    ├── HeuristicGoalStrategy - uses hand-crafted heuristics
    └── NoGoalStrategy - disables goal-setting entirely

Usage:
    from src.agent.goal_strategy import create_goal_strategy

    # Create LLM-based goal setter
    goal_strategy = create_goal_strategy("llm", config)

    # Or disable goals completely
    goal_strategy = create_goal_strategy("none", config)

    # Query for new goal
    goal = goal_strategy.generate_goal(state_summary, director)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Goal:
    """High-level directive chosen by a GoalStrategy."""

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


class GoalStrategy(ABC):
    """Abstract base class for goal-setting strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate_goal(
        self,
        state_summary: Dict[str, Any],
        director: Any  # Forward reference to avoid circular import
    ) -> Optional[Goal]:
        """
        Generate a new goal based on the current state.

        Args:
            state_summary: Dictionary containing current game state
            director: Reference to the Director for accessing active goals, memory, etc.

        Returns:
            Goal object if a new goal should be set, None otherwise
        """
        pass

    @abstractmethod
    def should_generate_goal(self, step: int, update_frequency: int) -> bool:
        """
        Determine if a new goal should be generated at this step.

        Args:
            step: Current training step
            update_frequency: Frequency to poll for new goals

        Returns:
            True if a new goal should be generated, False otherwise
        """
        pass

    def enabled(self) -> bool:
        """Returns whether this strategy is enabled."""
        return True


class LLMGoalStrategy(GoalStrategy):
    """Uses an LLM to generate intelligent, context-aware goals."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from src.agent.goal_llm import PokemonGoalLLM

        llm_cfg = config.get("goal_llm", {})
        self.llm = PokemonGoalLLM(
            api_url=llm_cfg.get("api_url", "http://localhost:11434/api/chat"),
            model=llm_cfg.get("model", "pokemon-goal"),
            enabled=llm_cfg.get("enabled", True),
            timeout=llm_cfg.get("timeout", 50.0)
        )
        self.allowed_goal_types = config.get("allowed_goal_types", {"NAVIGATE", "INTERACT", "BATTLE", "MENU", "SEARCH"})

    def enabled(self) -> bool:
        return self.llm.enabled

    def should_generate_goal(self, step: int, update_frequency: int) -> bool:
        """Only generate if LLM is enabled and it's time to poll."""
        return self.enabled() and (step % update_frequency == 0)

    def generate_goal(self, state_summary: Dict[str, Any], director: Any) -> Optional[Goal]:
        """Query the LLM for a new goal."""
        if not self.enabled():
            return None

        goal_json = self.llm.generate_goal(state_summary)

        if not goal_json or goal_json.get("goal_type") not in self.allowed_goal_types:
            return None

        # Fill in defaults
        if "priority" not in goal_json:
            goal_json["priority"] = 1
        if "max_steps" not in goal_json:
            goal_json["max_steps"] = 300

        target = goal_json.get("target", {})
        metadata = {"reasoning": goal_json.get("thought_process", "")}

        # Compute goal vector if target has coordinates
        goal_vector = None
        if "x" in target and "y" in target:
            metadata["target_x"] = target["x"]
            metadata["target_y"] = target["y"]
            metadata["target_action"] = target.get("action", "move")

            current_info = state_summary.get("current_info", {})
            curr_x, curr_y = current_info.get("x", 0), current_info.get("y", 0)
            dx = target["x"] - curr_x
            dy = target["y"] - curr_y
            mag = (dx**2 + dy**2)**0.5
            if mag > 0:
                goal_vector = [dx / mag, dy / mag]
            else:
                goal_vector = [0.0, 0.0]

        return Goal(
            name=goal_json.get("name", "llm-goal"),
            goal_type=goal_json.get("goal_type"),
            target=target,
            max_steps=int(goal_json.get("max_steps")),
            priority=int(goal_json.get("priority")),
            metadata=metadata,
            goal_vector=goal_vector
        )


class HeuristicGoalStrategy(GoalStrategy):
    """Uses hand-crafted rules to generate exploration goals."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.update_frequency = config.get("heuristic_goal_frequency", 500)

    def should_generate_goal(self, step: int, update_frequency: int) -> bool:
        """Generate goals at configured frequency."""
        return step % self.update_frequency == 0

    def generate_goal(self, state_summary: Dict[str, Any], director: Any) -> Optional[Goal]:
        """Create a simple exploration goal based on current location."""
        current_info = state_summary.get("current_info", {})
        current_map = current_info.get("map_id", 0)

        # Simple heuristic: explore the current map
        return Goal(
            name=f"heuristic-explore-{current_map}",
            goal_type="NAVIGATE",
            priority=0,
            target={"map_id": current_map, "description": "Explore current area"},
            max_steps=200,
            metadata={"strategy": "heuristic", "reason": "Systematic exploration"},
            goal_vector=None
        )


class NoGoalStrategy(GoalStrategy):
    """Disables goal-setting entirely - pure reactive RL."""

    def enabled(self) -> bool:
        return False

    def should_generate_goal(self, step: int, update_frequency: int) -> bool:
        """Never generate goals."""
        return False

    def generate_goal(self, state_summary: Dict[str, Any], director: Any) -> Optional[Goal]:
        """Always return None - no goals."""
        return None


def create_goal_strategy(strategy_type: str, config: Dict[str, Any]) -> GoalStrategy:
    """
    Factory function to create a goal strategy.

    Args:
        strategy_type: "llm", "heuristic", or "none"
        config: Configuration dictionary

    Returns:
        GoalStrategy instance

    Example:
        >>> config = load_config()
        >>> goal_strategy = create_goal_strategy("llm", config)
        >>> goal = goal_strategy.generate_goal(state_summary, director)
    """
    strategy_type = strategy_type.lower()

    if strategy_type == "llm":
        return LLMGoalStrategy(config)
    elif strategy_type == "heuristic":
        return HeuristicGoalStrategy(config)
    elif strategy_type == "none":
        return NoGoalStrategy(config)
    else:
        raise ValueError(f"Unknown goal strategy type: {strategy_type}. Choose 'llm', 'heuristic', or 'none'.")
