"""
Reward Strategy - Abstract interface for reward calculation strategies.

This module defines the pluggable reward calculation architecture, allowing you to
easily control how goal-based rewards are computed based on whether goal-setting is enabled.

Architecture:
    RewardStrategy (abstract) - defines the interface
    ├── GoalAwareRewardStrategy - full rewards when goals are active
    ├── BaseRewardStrategy - no goal bonuses, pure environment rewards
    └── CustomRewardStrategy - user-defined reward logic

Usage:
    from src.env.reward_strategy import create_reward_strategy

    # Create goal-aware rewards (when LLM is enabled)
    reward_strategy = create_reward_strategy("goal_aware", config, reward_sys)

    # Create base rewards only (when LLM is disabled)
    reward_strategy = create_reward_strategy("base", config, reward_sys)

    # Compute rewards
    reward_data = reward_strategy.compute_rewards(info, next_info, next_obs, action_data)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import numpy as np


class RewardStrategy(ABC):
    """Abstract base class for reward calculation strategies."""

    def __init__(self, config: Dict[str, Any], reward_system: Any):
        """
        Initialize reward strategy.

        Args:
            config: Configuration dictionary
            reward_system: RewardSystem instance for base reward calculations
        """
        self.config = config
        self.reward_sys = reward_system

    @abstractmethod
    def compute_rewards(
        self,
        info: Dict[str, Any],
        next_info: Dict[str, Any],
        next_obs: Any,
        action_data: Dict[str, Any],
        director: Any = None
    ) -> Dict[str, float]:
        """
        Compute reward components based on the strategy.

        Args:
            info: Previous game state info
            next_info: Current game state info
            next_obs: Current observation
            action_data: Dictionary containing action and goal context
            director: Reference to Director for goal completion checks

        Returns:
            Dictionary with keys:
                - total: Final clipped reward
                - goal_bonus: Goal-related rewards (0.0 if goals disabled)
                - goal_success: Boolean indicating goal completion
                - components: Dict of individual reward components
        """
        pass

    @abstractmethod
    def should_include_goal_rewards(self) -> bool:
        """Returns whether this strategy includes goal-based rewards."""
        pass


class GoalAwareRewardStrategy(RewardStrategy):
    """
    Full reward calculation including goal bonuses and completion rewards.

    Use this when goal-setting is ENABLED (LLM or heuristic goals).
    """

    def __init__(self, config: Dict[str, Any], reward_system: Any):
        super().__init__(config, reward_system)
        self.current_badges = 0
        self.goal_step_count = 0

    def should_include_goal_rewards(self) -> bool:
        return True

    def compute_rewards(
        self,
        info: Dict[str, Any],
        next_info: Dict[str, Any],
        next_obs: Any,
        action_data: Dict[str, Any],
        director: Any = None
    ) -> Dict[str, float]:
        """Compute full rewards including goal bonuses."""
        # Get base components from reward system
        goal_ctx = action_data.get("goal_ctx")
        comps = self.reward_sys.compute_components(
            next_info,
            next_obs,
            action_data["env_action"],
            goal_ctx=goal_ctx
        )

        # Compute goal completion rewards
        completion_reward, success = self._check_goal_completion(next_info, director)

        # Goal timeout penalty
        self.goal_step_count += 1
        timeout = False
        if director and director.active_goal:
            max_steps = director.active_goal.max_steps
            if self.goal_step_count >= max_steps:
                completion_reward -= 5.0
                timeout = True
                self.goal_step_count = 0

        # Combine all rewards
        total = (
            comps.get("global_reward", 0.0) +
            comps.get("nav_reward", 0.0) +
            comps.get("battle_reward", 0.0) +
            comps.get("menu_reward", 0.0) +
            comps.get("goal_bonus", 0.0) +
            completion_reward
        )

        clipped = float(np.clip(total, -10, 10))

        return {
            "total": clipped,
            "goal_bonus": comps.get("goal_bonus", 0.0) + completion_reward,
            "goal_success": success,
            "goal_timeout": timeout,
            "components": comps
        }

    def _check_goal_completion(self, next_info: Dict[str, Any], director: Any) -> Tuple[float, bool]:
        """
        Check if a goal was completed and return appropriate reward.

        Args:
            next_info: Current game state
            director: Director instance with active goal

        Returns:
            (completion_reward, success_flag)
        """
        # Badge reward (always given, not goal-specific)
        new_badges = next_info.get("badges", 0)
        if new_badges > self.current_badges:
            self.current_badges = new_badges
            return 50.0, False

        # No active goal
        if not director or not director.active_goal:
            return 0.0, False

        target = director.active_goal.target or {}
        reached = False

        # Check coordinate-based goals
        if "x" in target and "y" in target:
            curr_x, curr_y = next_info.get("x", -999), next_info.get("y", -999)
            dist = ((curr_x - target["x"])**2 + (curr_y - target["y"])**2)**0.5
            if dist < 1.0:
                reached = True

        # Check map-based goals
        elif target.get("map_id") is not None:
            if next_info.get("map_id") == target.get("map_id"):
                reached = True

        elif target.get("map_name"):
            if next_info.get("map_name") == target.get("map_name"):
                reached = True

        if reached:
            director.complete_goal(status="done")
            self.goal_step_count = 0
            return 10.0, True

        return 0.0, False


class BaseRewardStrategy(RewardStrategy):
    """
    Pure environment rewards - NO goal bonuses or completion rewards.

    Use this when goal-setting is DISABLED (pure reactive RL).
    This fixes the bug where goal completion rewards are given even when LLM is disabled.
    """

    def __init__(self, config: Dict[str, Any], reward_system: Any):
        super().__init__(config, reward_system)
        self.current_badges = 0

    def should_include_goal_rewards(self) -> bool:
        return False

    def compute_rewards(
        self,
        info: Dict[str, Any],
        next_info: Dict[str, Any],
        next_obs: Any,
        action_data: Dict[str, Any],
        director: Any = None
    ) -> Dict[str, float]:
        """Compute only base environment rewards, no goal bonuses."""
        # Get base components WITHOUT goal_ctx
        comps = self.reward_sys.compute_components(
            next_info,
            next_obs,
            action_data["env_action"],
            goal_ctx=None  # CRITICAL: Force no goal context
        )

        # Only include badge rewards (environment-driven, not goal-driven)
        badge_reward = self._check_badge_completion(next_info)

        # Combine ONLY environment rewards (no goal bonuses)
        total = (
            comps.get("global_reward", 0.0) +
            comps.get("nav_reward", 0.0) +
            comps.get("battle_reward", 0.0) +
            comps.get("menu_reward", 0.0) +
            badge_reward
        )

        clipped = float(np.clip(total, -10, 10))

        return {
            "total": clipped,
            "goal_bonus": 0.0,  # Always 0 for base strategy
            "goal_success": False,
            "goal_timeout": False,
            "components": comps
        }

    def _check_badge_completion(self, next_info: Dict[str, Any]) -> float:
        """Badge rewards are environment-based, not goal-based."""
        new_badges = next_info.get("badges", 0)
        if new_badges > self.current_badges:
            self.current_badges = new_badges
            return 50.0
        return 0.0


class HybridRewardStrategy(RewardStrategy):
    """
    Combines base rewards with optional goal shaping.

    Use this when you want directional shaping from goals but not completion bonuses.
    """

    def __init__(self, config: Dict[str, Any], reward_system: Any):
        super().__init__(config, reward_system)
        self.current_badges = 0
        self.include_direction_shaping = config.get("include_direction_shaping", True)

    def should_include_goal_rewards(self) -> bool:
        return self.include_direction_shaping

    def compute_rewards(
        self,
        info: Dict[str, Any],
        next_info: Dict[str, Any],
        next_obs: Any,
        action_data: Dict[str, Any],
        director: Any = None
    ) -> Dict[str, float]:
        """Include base rewards + directional shaping, but NO completion bonuses."""
        # Include goal_ctx only if direction shaping enabled
        goal_ctx = action_data.get("goal_ctx") if self.include_direction_shaping else None

        comps = self.reward_sys.compute_components(
            next_info,
            next_obs,
            action_data["env_action"],
            goal_ctx=goal_ctx
        )

        badge_reward = self._check_badge_completion(next_info)

        total = (
            comps.get("global_reward", 0.0) +
            comps.get("nav_reward", 0.0) +
            comps.get("battle_reward", 0.0) +
            comps.get("menu_reward", 0.0) +
            badge_reward
        )

        clipped = float(np.clip(total, -10, 10))

        return {
            "total": clipped,
            "goal_bonus": 0.0,  # No completion bonuses
            "goal_success": False,
            "goal_timeout": False,
            "components": comps
        }

    def _check_badge_completion(self, next_info: Dict[str, Any]) -> float:
        new_badges = next_info.get("badges", 0)
        if new_badges > self.current_badges:
            self.current_badges = new_badges
            return 50.0
        return 0.0


def create_reward_strategy(
    strategy_type: str,
    config: Dict[str, Any],
    reward_system: Any
) -> RewardStrategy:
    """
    Factory function to create a reward strategy.

    Args:
        strategy_type: "goal_aware", "base", or "hybrid"
        config: Configuration dictionary
        reward_system: RewardSystem instance

    Returns:
        RewardStrategy instance

    Example:
        >>> reward_sys = RewardSystem(config["rewards"])
        >>> strategy = create_reward_strategy("goal_aware", config, reward_sys)
        >>> reward_data = strategy.compute_rewards(info, next_info, obs, action_data, director)
    """
    strategy_type = strategy_type.lower()

    if strategy_type == "goal_aware":
        return GoalAwareRewardStrategy(config, reward_system)
    elif strategy_type == "base":
        return BaseRewardStrategy(config, reward_system)
    elif strategy_type == "hybrid":
        return HybridRewardStrategy(config, reward_system)
    else:
        raise ValueError(
            f"Unknown reward strategy type: {strategy_type}. "
            f"Choose 'goal_aware', 'base', or 'hybrid'."
        )
