"""
TRAINING SCRIPT - Pluggable Architecture

This training script uses pluggable architectures for:
1. RL Algorithm (Brain) - CrossQ, BBF, Rainbow, etc.
2. Goal-Setting Strategy - LLM, Heuristic, or None
3. Reward Strategy - Goal-Aware, Base, or Hybrid

Example usage:
    # LLM-based training with CrossQ
    python train.py --brain crossq --strategy llm

    # Pure reactive RL (no goals, no goal rewards)
    python train.py --brain crossq --strategy reactive

    # Heuristic goals with CrossQ
    python train.py --brain crossq --strategy heuristic

    # Override individual settings
    python train.py --brain crossq --strategy llm --memory-preset low
"""

import argparse
import os
import glob
import random
from typing import Any, Dict, Tuple
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.agent.director import Director, EpisodicMemory
from src.agent.pokemon_agent import create_agent
from src.core.env_factory import create_environment
from src.env.rewards import RewardSystem
from src.utils.logger import Logger
from src.utils.brain_config_loader import BrainConfigLoader
from src.utils.strategy_config_loader import StrategyConfigLoader, load_strategy_preset


class TrainingConfig:
    """Handles loading and overriding configuration dictionaries."""

    @staticmethod
    def load(config_path: str | None = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config", "hyperparameters.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def apply_overrides(
        cfg: Dict[str, Any],
        headless: bool | None,
        total_steps: int | None,
        state_path: str | None = None
    ) -> Dict[str, Any]:
        cfg = cfg.copy()
        env_cfg = cfg.get("environment", {}).copy()
        if headless is not None:
            env_cfg["headless"] = headless
        if env_cfg.get("headless") is True:
            env_cfg["emulation_speed"] = 0
        elif env_cfg.get("headless") is False:
            env_cfg["emulation_speed"] = 4
        if state_path:
            env_cfg["state_path"] = state_path
        cfg["training"] = cfg.get("training", {}).copy()
        if total_steps is not None:
            cfg["training"]["total_steps"] = total_steps
        cfg["environment"] = env_cfg
        return cfg


class PokemonTrainer:
    """
    Fully refactored trainer using pluggable architectures.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        brain_type: str,
        brain_config: Dict[str, Any],
        run_name: str | None,
        checkpoint_root: str,
        save_tag: str,
        device: str,
        seed: int | None
    ):
        self.cfg = cfg
        self.brain_type = brain_type
        self.brain_config = brain_config
        self.run_name = run_name
        self.save_tag = save_tag
        self.device = torch.device(device)
        self.seed = self._seed_everything(seed)

        self.checkpoint_dir, self.log_dir = self._prepare_dirs(checkpoint_root)
        self.logger = Logger(log_dir=self.log_dir, run_name=run_name)

        # Training params
        self.llm_update_freq = 300
        self.total_steps = cfg["training"]["total_steps"]
        self.steps_done = 0
        self.current_badges = 0

        # Tracking
        self.best_reward = float("-inf")
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.save_dir = cfg.get("saves_dir", "saves")
        os.makedirs(self.save_dir, exist_ok=True)

    def _seed_everything(self, seed: int | None) -> int:
        if seed is None:
            seed = int(torch.randint(0, 10**6, (1,)).item())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def _prepare_dirs(self, base_root: str) -> Tuple[str, str]:
        ckpt_dir = os.path.join(base_root, self.run_name) if self.run_name else base_root
        logs_root = os.path.join(base_root, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(logs_root, exist_ok=True)
        return ckpt_dir, logs_root

    def setup(self):
        """Setup environment, agent, strategies, and memory."""
        # Create environment using factory (supports hot-swapping between games)
        game_id = self.cfg.get("game", "pokemon_red")
        self.env = create_environment(game_id, self.cfg["environment"])

        # Get game data provider and memory interface from environment
        self.game_data = self.env.get_game_data()
        memory_interface = self.env.get_memory_interface()

        # Create reward system with memory interface
        self.reward_sys = RewardSystem(self.cfg["rewards"], memory_interface=memory_interface)

        # Create strategies
        self.strategy_loader = StrategyConfigLoader(self.cfg)
        self.strategy_loader.print_strategy_info()

        self.goal_strategy = self.strategy_loader.create_goal_strategy()
        self.reward_strategy = self.strategy_loader.create_reward_strategy(self.reward_sys)

        # Create director with goal strategy
        self.director = Director(self.cfg["director"], self.goal_strategy)

        # Memory for episodic events
        self.memory = EpisodicMemory()

        self._setup_agent()

    def _setup_agent(self):
        """Create agent using the brain architecture."""
        agent_cfg = self.cfg.get("agent", {}).copy()
        allowed_actions = agent_cfg.get("allowed_actions", list(range(self.env.action_space.n)))

        self.agent = create_agent(
            brain_type=self.brain_type,
            brain_config=self.brain_config,
            allowed_actions=allowed_actions,
            device=str(self.device)
        )

        # Load checkpoint if exists
        path = os.path.join(self.checkpoint_dir, f"agent_{self.brain_type}_{self.save_tag}.pth")
        if os.path.exists(path):
            try:
                self.agent.load_checkpoint(path)
                print(f"✓ Loaded {self.brain_type} agent from {path}")
            except Exception as e:
                print(f"⚠ Could not load agent weights: {e}")

        print(f"✓ Created {self.brain_type.upper()} agent")
        print(f"  Memory: {self.brain_config.get('buffer_capacity', 'N/A')} capacity, "
              f"{self.brain_config.get('batch_size', 'N/A')} batch size")

    def run(self):
        """Main training loop."""
        obs, info = self._reset_env()
        self._poll_goal_strategy(info, force=True)

        pbar = tqdm(total=self.total_steps)
        self.goal_step_count = 0

        while self.steps_done < self.total_steps:
            self.steps_done += 1

            # 1. Sense: Read text & update memory
            self._update_memory(info)

            # 2. Think: Poll goal strategy periodically
            if self.steps_done % self.llm_update_freq == 0:
                self._poll_goal_strategy(info)

            # 3. Act: Select & execute action
            action_data = self._select_action(obs, info)
            next_obs, _, terminated, truncated, next_info = self.env.step(action_data["env_action"])

            # 4. Learn: Compute rewards using reward strategy
            reward_data = self._compute_rewards(info, next_info, next_obs, action_data)

            # Store experience and train
            self.agent.store_experience(
                obs=obs,
                action=action_data["local_action"],
                reward=reward_data["total"],
                next_obs=next_obs,
                done=terminated or truncated
            )

            loss, metrics = self.agent.train_step()
            if loss is not None:
                self._periodically_log_metrics(reward_data, metrics)
            else:
                self._periodically_log_metrics(reward_data, {})

            self._log_progress(info, reward_data, pbar)

            # Loop upkeep
            obs, info = next_obs, next_info

            # Handle map changes (clear goals)
            if next_info.get("map_id") != info.get("map_id"):
                if self.director.active_goal:
                    self.director.complete_goal(status="map_changed")
                self.director.clear_goals()
                self.goal_step_count = 0
                self._poll_goal_strategy(next_info)

            # Episode end handling
            if terminated or truncated:
                obs, info = self._handle_episode_end(reward_data["total"])
                self.goal_step_count = 0

            # Checkpoint saving
            if self.steps_done % self.cfg["training"]["save_frequency"] == 0:
                self._save_agent(self.save_tag)

            pbar.update(1)

        self._cleanup()

    def _reset_env(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and load latest badge checkpoint."""
        obs, info = self.env.reset()
        self.reward_sys.reset()

        # Load latest badge save
        ckpt_path = self._get_latest_checkpoint()
        if ckpt_path and os.path.exists(ckpt_path):
            self.env.load_state(ckpt_path)
            obs, info = self.env._get_obs(), self.env._get_info()

        self.current_badges = info.get("badges", 0)
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.goal_step_count = 0
        return obs, info

    def _get_latest_checkpoint(self) -> str | None:
        """Find save with highest badge count."""
        saves = glob.glob(os.path.join(self.save_dir, "badge_*.state"))
        if not saves:
            return None
        try:
            latest = max(saves, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
            return latest
        except:
            return None

    def _update_memory(self, info: Dict[str, Any]):
        """Check for sensory inputs and log them."""
        current_map_id = info.get("map_id")
        current_map_name = self.game_data.map_id_to_name(current_map_id)

        if not hasattr(self, "_last_map"):
            self._last_map = current_map_name
        if current_map_name != self._last_map:
            self.memory.log_event(self.steps_done, "MAP_CHANGE", f"Entered {current_map_name}")
            self._last_map = current_map_name

        # Text decoding with cursor-based attention (works for all Pokemon GB games)
        if hasattr(self.env, 'text_decoder'):
            try:
                # Debug mode disabled after initial diagnosis
                debug_mode = False
                
                decoded_data = self.env.text_decoder.decode(debug=debug_mode)
                selection = decoded_data.get('selection', '').strip()
                narrative = decoded_data.get('narrative', '').strip()
                
                # Track last seen text to avoid spam
                last_selection = getattr(self, '_last_selection', '')
                last_narrative = getattr(self, '_last_narrative', '')
                
                # Always try legacy method for comparison
                legacy_text = self.env.text_decoder.read_current_text()
                if debug_mode and legacy_text:
                    print(f"DEBUG: Legacy method found text: '{legacy_text}'")
                
                # Only output when text changes (avoid repetitive logging)
                selection_changed = selection and selection != last_selection
                narrative_changed = narrative and narrative != last_narrative
                
                if selection_changed or narrative_changed:
                    debug_parts = []
                    if selection_changed:
                        debug_parts.append(f"[CURSOR→{selection}]")
                        self.memory.log_event(self.steps_done, "CURSOR_SELECTION", selection)
                        self._last_selection = selection
                    if narrative_changed:
                        debug_parts.append(f"[TEXT: {narrative}]")
                        self.memory.log_event(self.steps_done, "NARRATIVE_TEXT", narrative)
                        self._last_narrative = narrative
                    
                    # Include current map and step context for better debugging
                    map_context = f"[{current_map_name}]" if current_map_name else "[Unknown Map]"
                    print(f"🎮 AI Reading {map_context} Step {self.steps_done}: {' '.join(debug_parts)}")
                
                # Clear cached text when both are empty (dialogue closed)
                if not selection and not narrative:
                    self._last_selection = ''
                    self._last_narrative = ''
                elif debug_mode and not (selection_changed or narrative_changed):
                    print(f"DEBUG: Same text as before at step {self.steps_done} - not logging")
                    
            except Exception as e:
                if debug_mode:
                    print(f"DEBUG: Text decoder error: {e}")
                # Don't disrupt training for text errors

    def _poll_goal_strategy(self, info: Dict[str, Any], force: bool = False):
        """Query the goal strategy for a new goal."""
        # Build state summary
        context_log = self.memory.consume_history()

        valid_moves = []
        conns = info.get("map_connections", {})
        for direction, details in conns.items():
            if details.get("exists"):
                valid_moves.append(f"Exit {direction.title()}")

        warps = info.get("map_warps", [])
        for w in warps:
            valid_moves.append(f"Warp at ({w['x']},{w['y']})")

        sprites = info.get("sprites", [])
        for s in sprites:
            valid_moves.append(f"{s['type']} at ({s['x']},{s['y']})")

        nearby_entities_str = "; ".join(valid_moves) if valid_moves else "None visible"

        state_summary = {
            "history": context_log,
            "current_location": self.game_data.map_id_to_name(info.get("map_id")),
            "nearby_entities": nearby_entities_str,
            "party_size": info.get("party_size"),
            "badges": info.get("badges"),
            "last_goal": self.director.active_goal.as_dict() if self.director.active_goal else None,
            "current_info": info  # For goal vector calculation
        }

        # Poll the strategy
        new_goal = self.director.poll_for_goal(state_summary, self.steps_done, self.llm_update_freq)

        if new_goal:
            self.director.clear_goals()
            self.director.enqueue_goal(new_goal, self.steps_done)
            self.goal_step_count = 0

    def _select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using agent."""
        # Get goal context from director
        _, _, _, goal_ctx = self.director.select_specialist(
            torch.from_numpy(obs).unsqueeze(0).unsqueeze(0),
            info,
            0.0  # Epsilon managed by brain
        )

        # Get action from agent
        env_action, local_action, features = self.agent.get_action_with_features(
            obs,
            deterministic=False,
            goal=goal_ctx
        )

        return {
            "env_action": env_action,
            "local_action": local_action,
            "features": features,
            "goal_ctx": goal_ctx
        }

    def _compute_rewards(
        self,
        info: Dict[str, Any],
        next_info: Dict[str, Any],
        next_obs: np.ndarray,
        action_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute rewards using the reward strategy."""
        # CRITICAL: Reward strategy handles goal enablement checking
        reward_data = self.reward_strategy.compute_rewards(
            info, next_info, next_obs, action_data, self.director
        )

        # Badge checkpoint saving
        new_badges = next_info.get("badges", 0)
        if new_badges > self.current_badges:
            self.current_badges = new_badges
            save_path = os.path.join(self.save_dir, f"badge_{new_badges}.state")
            self.env.save_state(save_path)
            self.memory.log_event(self.steps_done, "SYSTEM", f"Badge Earned! Saved to {save_path}")

        return reward_data

    def _periodically_log_metrics(self, reward_data, brain_metrics):
        if self.steps_done % self.cfg["training"]["fast_update_frequency"] != 0:
            return

        metrics = {"reward/total": reward_data["total"]}
        metrics.update(self.director.get_goal_metrics())
        metrics.update(brain_metrics)
        self.logger.log_step(metrics, self.steps_done)

    def _log_progress(self, info, reward_data, pbar):
        self.episode_reward += reward_data["total"]
        self.episode_steps += 1

        if self.director.active_goal:
            g_name = self.director.active_goal.name
            g_type = self.director.active_goal.goal_type
        else:
            g_name = "none"
            g_type = "none"

        brain_metrics = self.agent.get_metrics()
        epsilon = brain_metrics.get("brain/epsilon", 0.0)

        strategy_info = self.strategy_loader.get_strategy_info()
        goal_strat = strategy_info["goal_strategy"][:3].upper()  # LLM, HEU, NON

        pbar.set_description(
            f"[{self.brain_type.upper()}|{goal_strat}] "
            f"Rew:{self.episode_reward:.1f} | "
            f"ε:{epsilon:.3f} | "
            f"Badges:{self.current_badges} | "
            f"{g_type}:{g_name}"
        )

    def _handle_episode_end(self, final_reward):
        self.logger.log_step({
            "episode/reward": self.episode_reward,
            "episode/badges": self.current_badges
        }, self.steps_done)
        self.episode_count += 1

        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self._save_agent("best_reward")

        ckpt = self._get_latest_checkpoint()
        if ckpt:
            self.env.load_state(ckpt)
            return self.env._get_obs(), self.env._get_info()
        return self.env.reset()

    def _save_agent(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"agent_{self.brain_type}_{tag}.pth")
        self.agent.save_checkpoint(path)

    def _cleanup(self):
        self._save_agent(self.save_tag)
        self.logger.close()
        self.env.close()


def train(
    config_path=None,
    brain_type="crossq",
    strategy_preset="llm",
    memory_preset="low",
    run_name=None,
    checkpoint_root="experiments",
    save_tag="latest",
    total_steps_override=None,
    headless=None,
    state_path_override=None,
    device_override=None,
    seed=None,
    **brain_overrides
):
    """
    Train Pokemon agent with fully pluggable architecture.

    Args:
        brain_type: "crossq", "bbf", or "rainbow"
        strategy_preset: "llm", "heuristic", "reactive", or "hybrid"
        memory_preset: "minimal", "low", "medium", or "high"
        brain_overrides: Additional brain config overrides
    """
    # Load base config
    cfg = TrainingConfig.load(config_path)
    cfg = TrainingConfig.apply_overrides(cfg, headless, total_steps_override, state_path_override)

    # Apply strategy preset
    cfg = load_strategy_preset(strategy_preset, cfg)

    # Load brain config
    brain_loader = BrainConfigLoader()
    brain_config = brain_loader.get_brain_config(brain_type, memory_preset, brain_overrides)

    # Print configuration
    vram_estimate = brain_loader.estimate_vram_usage(brain_config)
    print(f"\n{'='*60}")
    print(f"REFACTORED TRAINING - Pluggable Architecture")
    print(f"{'='*60}")
    print(f"Brain: {brain_type.upper()} | Memory: {memory_preset.upper()}")
    print(f"Strategy: {strategy_preset.upper()}")
    print(f"Estimated VRAM: {vram_estimate['total_mb']:.0f} MB")
    print(f"{'='*60}\n")

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")

    trainer = PokemonTrainer(
        cfg, brain_type, brain_config, run_name,
        checkpoint_root, save_tag, device, seed
    )
    trainer.setup()
    trainer.run()
    return {"done": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Pokemon Red AI with fully pluggable architecture"
    )

    # Main args
    parser.add_argument("--config", type=str, help="Path to hyperparameters.yaml")
    parser.add_argument("--brain", type=str, default="crossq",
                        choices=["crossq", "bbf", "rainbow"],
                        help="RL algorithm")
    parser.add_argument("--strategy", type=str, default="llm",
                        choices=["llm", "heuristic", "reactive", "hybrid"],
                        help="Goal-setting strategy preset")
    parser.add_argument("--memory-preset", type=str, default="low",
                        choices=["minimal", "low", "medium", "high"],
                        help="Memory usage preset")

    # Standard training args
    parser.add_argument("--run-name", type=str, help="Experiment run name")
    parser.add_argument("--checkpoint-root", type=str, default="experiments")
    parser.add_argument("--save-tag", type=str, default="latest")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--windowed", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--state-path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)

    # Brain overrides
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-capacity", type=int)
    parser.add_argument("--gamma", type=float)

    args = parser.parse_args()

    # Collect brain overrides
    brain_overrides = {}
    if args.learning_rate is not None:
        brain_overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        brain_overrides["batch_size"] = args.batch_size
    if args.buffer_capacity is not None:
        brain_overrides["buffer_capacity"] = args.buffer_capacity
    if args.gamma is not None:
        brain_overrides["gamma"] = args.gamma

    train(
        config_path=args.config,
        brain_type=args.brain,
        strategy_preset=args.strategy,
        memory_preset=args.memory_preset,
        run_name=args.run_name,
        checkpoint_root=args.checkpoint_root,
        save_tag=args.save_tag,
        total_steps_override=args.total_steps,
        headless=args.headless,
        state_path_override=args.state_path,
        device_override=args.device,
        seed=args.seed,
        **brain_overrides
    )
