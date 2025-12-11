"""
Strategy Configuration Loader

Loads and configures goal-setting and reward strategies based on a unified configuration.
Automatically selects the correct strategy combination based on whether goals are enabled.

Usage:
    from src.utils.strategy_config_loader import StrategyConfigLoader

    loader = StrategyConfigLoader(config)
    goal_strategy = loader.create_goal_strategy()
    reward_strategy = loader.create_reward_strategy(reward_sys)

    # The loader automatically selects:
    # - LLM goal strategy + goal_aware rewards if LLM enabled
    # - No goal strategy + base rewards if LLM disabled
"""

from typing import Dict, Any, Tuple


class StrategyConfigLoader:
    """Manages strategy selection based on configuration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy loader.

        Args:
            config: Full configuration dictionary (from hyperparameters.yaml)
        """
        self.config = config
        self._detect_strategy_mode()

    def _detect_strategy_mode(self):
        """Detect which strategies to use based on configuration."""
        goal_llm_cfg = self.config.get("goal_llm", {})
        llm_enabled = bool(goal_llm_cfg.get("enabled", False))

        # Explicit strategy override (for advanced users)
        explicit_goal_strategy = self.config.get("goal_strategy")
        explicit_reward_strategy = self.config.get("reward_strategy")

        if explicit_goal_strategy:
            self.goal_strategy_type = explicit_goal_strategy
        else:
            # Auto-detect based on LLM enabled status
            self.goal_strategy_type = "llm" if llm_enabled else "none"

        if explicit_reward_strategy:
            self.reward_strategy_type = explicit_reward_strategy
        else:
            # Auto-detect: goal-aware if ANY goals enabled, base otherwise
            if self.goal_strategy_type == "none":
                self.reward_strategy_type = "base"
            else:
                self.reward_strategy_type = "goal_aware"

        # Store for logging
        self.llm_enabled = llm_enabled

    def create_goal_strategy(self):
        """
        Create the appropriate goal strategy.

        Returns:
            GoalStrategy instance
        """
        from src.agent.goal_strategy import create_goal_strategy
        return create_goal_strategy(self.goal_strategy_type, self.config)

    def create_reward_strategy(self, reward_system: Any):
        """
        Create the appropriate reward strategy.

        Args:
            reward_system: RewardSystem instance

        Returns:
            RewardStrategy instance
        """
        from src.env.reward_strategy import create_reward_strategy
        return create_reward_strategy(self.reward_strategy_type, self.config, reward_system)

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about selected strategies.

        Returns:
            Dictionary with strategy configuration details
        """
        return {
            "goal_strategy": self.goal_strategy_type,
            "reward_strategy": self.reward_strategy_type,
            "llm_enabled": self.llm_enabled,
            "goal_rewards_enabled": self.reward_strategy_type in ["goal_aware", "hybrid"],
        }

    def print_strategy_info(self):
        """Print strategy configuration to console."""
        info = self.get_strategy_info()
        print(f"\n{'='*60}")
        print(f"Strategy Configuration")
        print(f"{'='*60}")
        print(f"Goal Strategy:    {info['goal_strategy'].upper()}")
        print(f"Reward Strategy:  {info['reward_strategy'].upper()}")
        print(f"LLM Enabled:      {info['llm_enabled']}")
        print(f"Goal Rewards:     {info['goal_rewards_enabled']}")
        print(f"{'='*60}\n")


def get_strategy_presets() -> Dict[str, Dict[str, str]]:
    """
    Get predefined strategy combinations.

    Returns:
        Dictionary of preset configurations
    """
    return {
        "llm": {
            "description": "Full LLM-based goal-setting with completion rewards",
            "goal_strategy": "llm",
            "reward_strategy": "goal_aware"
        },
        "heuristic": {
            "description": "Hand-crafted heuristic goals with completion rewards",
            "goal_strategy": "heuristic",
            "reward_strategy": "goal_aware"
        },
        "reactive": {
            "description": "Pure reactive RL - no goals, no goal rewards (fastest)",
            "goal_strategy": "none",
            "reward_strategy": "base"
        },
        "hybrid": {
            "description": "Heuristic goals with directional shaping only (no completion bonuses)",
            "goal_strategy": "heuristic",
            "reward_strategy": "hybrid"
        }
    }


def load_strategy_preset(preset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a strategy preset into the configuration.

    Args:
        preset_name: Name of the preset ("llm", "heuristic", "reactive", "hybrid")
        config: Configuration dictionary to modify

    Returns:
        Modified configuration dictionary
    """
    presets = get_strategy_presets()
    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {list(presets.keys())}"
        )

    preset = presets[preset_name]
    config = config.copy()
    config["goal_strategy"] = preset["goal_strategy"]
    config["reward_strategy"] = preset["reward_strategy"]

    return config
