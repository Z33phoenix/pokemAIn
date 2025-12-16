# -*- coding: utf-8 -*-
"""
TRAINING SCRIPT - Pluggable Architecture + Human Brain

This script trains Pokemon agents through a unified training pipeline.
You can substitute different "brains" - either RL algorithms OR a human player.

Brain Types:
    - crossq: CrossQ RL algorithm (stable dual Q-networks)
    - bbf: BBF (coming soon)
    - rainbow: Rainbow DQN (coming soon)
    - human: You! Keyboard-controlled player in the training loop

When using --brain human:
    - You control the game via keyboard
    - All training infrastructure still runs (rewards, memory, battle logic, etc.)
    - Perfect for debugging and verifying game integration without waiting for RL

Example usage:

    # RL Training with LLM-based goal setting
    python train.py --brain crossq --strategy llm

    # Pure reactive RL (no goals)
    python train.py --brain crossq --strategy reactive

    # HUMAN PLAYER MODE - Full training pipeline with manual keyboard control
    python train.py --brain human --strategy reactive

    # RL with custom overrides
    python train.py --brain crossq --strategy llm --learning-rate 0.0001
"""

import argparse
import sys
import io

# Set UTF-8 encoding for Windows console output with line buffering for real-time output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
import os
import random
from typing import Any, Dict, Tuple, Optional
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
from src.utils.human_buffer import HumanExperienceRecorder


BADGE_SEQUENCE = [
    "Boulder Badge",
    "Cascade Badge",
    "Thunder Badge",
    "Rainbow Badge",
    "Soul Badge",
    "Marsh Badge",
    "Volcano Badge",
    "Earth Badge",
]

REGION_BY_GAME = {
    "pokemon_red": "Kanto",
    "pokemon_blue": "Kanto",
    "pokemon_yellow": "Kanto",
    "pokemon_emerald": "Hoenn",
}

TEXT_EMBED_DIM = 17


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
        seed: int | None,
        human_recorder: Optional[HumanExperienceRecorder] = None
    ):
        self.cfg = cfg
        self.brain_type = brain_type
        self.brain_config = brain_config
        self.run_name = run_name
        self.save_tag = save_tag
        self.device = torch.device(device)
        self.seed = self._seed_everything(seed)
        self.human_recorder = human_recorder

        self.checkpoint_dir, self.log_dir = self._prepare_dirs(checkpoint_root)
        self.logger = Logger(log_dir=self.log_dir, run_name=run_name)

        # Training params
        self.llm_update_freq = 300
        self.total_steps = cfg["training"]["total_steps"]
        self.steps_done = 0
        self.current_badges = 0

        # Tracking
        self.session_reward = 0.0
        self.session_steps = 0
        self._text_embedding_dim = TEXT_EMBED_DIM
        self._zero_text_embedding = np.zeros(self._text_embedding_dim, dtype=np.float32)
        self._current_cursor_embedding = self._zero_text_embedding.copy()

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
        action_names = self.env.get_action_names()

        if isinstance(self.brain_config, dict):
            self.brain_config["text_feature_dim"] = self._text_embedding_dim

        self.agent = create_agent(
            brain_type=self.brain_type,
            brain_config=self.brain_config,
            allowed_actions=allowed_actions,
            device=str(self.device),
            action_names=action_names
        )

        # Load checkpoint if exists (skip for human brain)
        if self.brain_type.lower() != "human":
            path = os.path.join(self.checkpoint_dir, f"agent_{self.brain_type}_{self.save_tag}.pth")
            if os.path.exists(path):
                try:
                    self.agent.load_checkpoint(path)
                    print(f"✓ Loaded {self.brain_type} agent from {path}")
                except Exception as e:
                    print(f"⚠ Could not load agent weights: {e}")

        print(f"✓ Created {self.brain_type.upper()} agent")
        if self.brain_type.lower() != "human":
            print(f"  Memory: {self.brain_config.get('buffer_capacity', 'N/A')} capacity, "
                  f"{self.brain_config.get('batch_size', 'N/A')} batch size")

    def run(self):
        """Main training loop."""
        obs, info = self._boot_env()
        self._update_memory(info)
        self._poll_goal_strategy(info, force=True)

        pbar = tqdm(total=self.total_steps, file=sys.stderr, dynamic_ncols=True)
        self.goal_step_count = 0

        while self.steps_done < self.total_steps:
            self.steps_done += 1

            # 2. Think: Poll goal strategy periodically
            if self.steps_done % self.llm_update_freq == 0:
                self._poll_goal_strategy(info)

            # 3. Act: Select & execute action
            current_cursor_embedding = getattr(self, "_current_cursor_embedding", self._zero_text_embedding)
            action_data = self._select_action(obs, info, current_cursor_embedding)
            next_obs, _, terminated, truncated, next_info = self.env.step(action_data["env_action"])

            # 4. Learn: Compute rewards and train
            next_cursor_embedding = self._update_memory(next_info)
            reward_data = self._compute_rewards(info, next_info, next_obs, action_data)

            # Store experience and train
            self.agent.store_experience(
                obs=obs,
                action=action_data["local_action"],
                reward=reward_data["total"],
                next_obs=next_obs,
                done=False,
                text_embedding=current_cursor_embedding,
                next_text_embedding=next_cursor_embedding
            )

            loss, metrics = self.agent.train_step()
            if loss is not None:
                self._periodically_log_metrics(reward_data, metrics)
            else:
                self._periodically_log_metrics(reward_data, {})

            if self._maybe_record_human_demo(
                obs=obs,
                next_obs=next_obs,
                local_action=action_data["local_action"],
                env_action=action_data["env_action"],
                reward=reward_data["total"],
                done=False,
                text_embedding=current_cursor_embedding,
                next_text_embedding=next_cursor_embedding
            ):
                print(f"\nCaptured {self.human_recorder.size} human transitions. Stopping early.")
                break

            self._log_progress(info, reward_data, pbar)

            # Loop upkeep
            obs, info = next_obs, next_info

            if terminated or next_info.get("window_closed"):
                print("\nEnvironment requested termination. Ending training loop.")
                break

            # Handle map changes (clear goals)
            if next_info.get("map_id") != info.get("map_id"):
                if self.director.active_goal:
                    self.director.complete_goal(status="map_changed")
                self.director.clear_goals()
                self.goal_step_count = 0
                self._poll_goal_strategy(next_info)

            # Checkpoint saving
            if self.steps_done % self.cfg["training"]["save_frequency"] == 0:
                self._save_agent(self.save_tag)

            pbar.update(1)

        self._cleanup()

    def _boot_env(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Boot environment once and initialize tracking state."""
        obs, info = self.env.reset()
        self.reward_sys.reset()

        self.current_badges = info.get("badges", 0)
        self.session_reward = 0.0
        self.session_steps = 0
        self.goal_step_count = 0
        return obs, info

    def _update_memory(self, info: Dict[str, Any]) -> np.ndarray:
        """Check for sensory inputs, log them, and update text embeddings."""
        current_map_id = info.get("map_id")
        current_map_name = self.game_data.map_id_to_name(current_map_id)
        selection = ''

        if not hasattr(self, "_last_map"):
            self._last_map = current_map_name
        if current_map_name != self._last_map:
            self.memory.log_event(self.steps_done, "MAP_CHANGE", f"Entered {current_map_name}")
            self._last_map = current_map_name

        # Text decoding with cursor-based attention (works for all Pokemon GB games)
        if hasattr(self.env, 'text_decoder'):
            try:
                # Enable debug mode to see raw tiles and decoded text
                debug_mode = False  # Set to True for detailed debugging

                decoded_data = self.env.text_decoder.decode(debug=debug_mode)
                selection = decoded_data.get('selection', '').strip()
                narrative = decoded_data.get('narrative', '').strip()

                # Track last seen selection and narrative to avoid duplicate logging
                last_selection = getattr(self, '_last_selection', '')
                last_narrative = getattr(self, '_last_narrative', '')

                # Selection changes should still be tracked
                selection_changed = selection and selection != last_selection

                # Only log narrative when it's NEW (changed from last time)
                # The decoder now handles completion detection, so any narrative here is ready to log
                narrative_changed = narrative and narrative != last_narrative

                # Track selection and narrative changes for progress bar display
                if selection_changed or narrative_changed:
                    if selection_changed:
                        self.memory.log_event(self.steps_done, "CURSOR_SELECTION", selection)
                        self._last_selection = selection
                        # Store for progress bar display (truncate if too long)
                        self._current_cursor_display = selection[:30] if len(selection) > 30 else selection
                    if narrative_changed:
                        self.memory.log_event(self.steps_done, "NARRATIVE_TEXT", narrative)
                        self._last_narrative = narrative
                        # Important narratives get printed (battle messages, etc.)
                        if any(kw in narrative.lower() for kw in ['damage', 'fainted', 'won', 'lost', 'caught']):
                            print(f"📜 {narrative}")

                # Clear cached selection when empty (dialogue closed)
                if not selection:
                    self._last_selection = ''
                    self._current_cursor_display = ''

            except Exception as e:
                if debug_mode:
                    print(f"DEBUG: Text decoder error: {e}")
                # Don't disrupt training for text errors

        self._current_cursor_embedding = self._vectorize_selection(selection)
        return self._current_cursor_embedding

    def _infer_region(self) -> str:
        """Map configured game id to a region label for LLM context."""
        game_id = (self.cfg.get("game") or "").lower()
        return REGION_BY_GAME.get(game_id, "Unknown")

    def _derive_screen_mode(self, info: Dict[str, Any]) -> str:
        """Coarse screen mode classification sent to the LLM."""
        if info.get("battle_active"):
            return "BATTLE"
        if info.get("menu_open"):
            return "MENU"
        return "OVERWORLD"

    def _extract_nearby_entities(self, info: Dict[str, Any]) -> list[str]:
        """Build a compact list of actionable entities/warps near the player."""
        entries: list[str] = []
        conns = info.get("map_connections", {}) or {}
        for direction, details in conns.items():
            if not details or not details.get("exists"):
                continue
            dest_map = details.get("dest_map")
            dest_name = self.game_data.map_id_to_name(dest_map)
            entries.append(f"Exit {direction.title()} -> {dest_name}")

        for warp in info.get("map_warps", []) or []:
            dest_name = self.game_data.map_id_to_name(warp.get("dest_map"))
            entries.append(f"Warp at ({warp.get('x')},{warp.get('y')}) -> {dest_name}")

        for sprite in info.get("sprites", []) or []:
            sprite_type = sprite.get("type", "NPC")
            entries.append(f"{sprite_type} at ({sprite.get('x')},{sprite.get('y')})")
        return entries

    def _format_memory_log(self, context_log: str) -> list[str]:
        """Convert memory history into a list while dropping placeholder text."""
        if not context_log or context_log.strip().lower() == "no recent events.":
            return []
        return [line.strip() for line in context_log.splitlines() if line.strip()]

    def _vectorize_selection(self, selection: str) -> np.ndarray:
        """Convert the current cursor selection into a fixed-length embedding."""
        selection = selection or ''
        vec = np.zeros(self._text_embedding_dim, dtype=np.float32)
        if not selection:
            return vec

        max_chars = self._text_embedding_dim - 1
        trimmed = selection.upper()[:max_chars]
        for idx, char in enumerate(trimmed):
            ascii_code = max(32, min(126, ord(char)))
            vec[idx] = (ascii_code - 32) / 94.0  # Normalize printable ASCII range
        vec[-1] = 1.0  # Presence flag
        return vec

    def _build_party_knowledge(self, info: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Summarize party condition with the limited telemetry currently available."""
        party_size = info.get("party_size") or 0
        if party_size <= 0:
            return []

        hp_current = info.get("hp_current", 0)
        hp_max = max(info.get("hp_max", 1), 1)
        hp_percent = info.get("hp_percent", 0.0) or 0.0
        condition = "OK"
        if hp_percent < 0.25:
            condition = "CRITICAL"
        elif hp_percent < 0.5:
            condition = "HURT"

        party_summary = {
            "species": "Unknown Lead",
            "level": None,
            "condition": condition,
            "hp": f"{hp_current}/{hp_max}",
            "notes": f"Party size: {party_size}",
        }
        return [party_summary]

    def _build_inventory_knowledge(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key badges and items available to the agent."""
        badge_count = int(info.get("badges", 0) or 0)
        badges = BADGE_SEQUENCE[:badge_count]
        return {
            "key_items": [],
            "badges": badges,
        }

    def _build_llm_payload(
        self,
        info: Dict[str, Any],
        context_log: str,
        nearby_entities: list[str]
    ) -> Dict[str, Any]:
        """Create the structured JSON payload sent to goal LLMs."""
        location_name = self.game_data.map_id_to_name(info.get("map_id"))
        payload = {
            "meta": {
                "game": getattr(self.game_data, "get_game_name", lambda: self.cfg.get("game", "pokemon_red"))(),
                "region": self._infer_region(),
            },
            "current_state": {
                "location": {
                    "name": location_name,
                    "map_id": info.get("map_id"),
                    "coordinates": {"x": info.get("x"), "y": info.get("y")},
                    "nearby_entities": nearby_entities,
                },
                "screen_mode": self._derive_screen_mode(info),
                "hp": {
                    "current": info.get("hp_current"),
                    "max": info.get("hp_max"),
                    "percent": round((info.get("hp_percent") or 0.0) * 100, 1),
                },
            },
            "party_knowledge": self._build_party_knowledge(info),
            "inventory_knowledge": self._build_inventory_knowledge(info),
            "memory_log": self._format_memory_log(context_log),
        }

        active_goal = self.director.active_goal.as_dict() if self.director.active_goal else None
        last_goal = self.director.get_last_completed_goal()
        if active_goal or last_goal:
            payload["goal_context"] = {
                "active_goal": active_goal,
                "last_completed_goal": last_goal,
            }
        return payload

    def _poll_goal_strategy(self, info: Dict[str, Any], force: bool = False):
        """Query the goal strategy for a new goal."""
        # Build state summary
        context_log = self.memory.consume_history()

        nearby_entities = self._extract_nearby_entities(info)

        state_summary = {
            "llm_payload": self._build_llm_payload(info, context_log, nearby_entities),
            "current_info": info,  # For goal vector calculation
            "last_goal": self.director.active_goal.as_dict() if self.director.active_goal else None,
        }

        # Poll the strategy
        new_goal = self.director.poll_for_goal(state_summary, self.steps_done, self.llm_update_freq)

        if new_goal:
            self.director.clear_goals()
            self.director.enqueue_goal(new_goal, self.steps_done)
            self.goal_step_count = 0

    def _select_action(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        text_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
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
            goal=goal_ctx,
            text_embedding=text_embedding
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
        
        # DEBUG: Track massive reward drops after battle
        prev_battle = info.get("battle_active", False)
        curr_battle = next_info.get("battle_active", False)
        total_reward = reward_data["total"]
        
        if prev_battle and not curr_battle and total_reward < -100:  # Just exited battle with huge penalty
            print(f"🔥 POST-BATTLE MASSIVE PENALTY: {total_reward:.1f}")
            print(f"   Reward breakdown from reward_strategy:")
            for key, value in reward_data.items():
                if abs(value) > 1.0:
                    print(f"     {key}: {value:.1f}")

        # Badge checkpoint saving
        new_badges = next_info.get("badges", 0)
        if new_badges > self.current_badges:
            self.current_badges = new_badges
            self.memory.log_event(self.steps_done, "SYSTEM", f"Badge Earned! Total badges: {new_badges}")

        return reward_data

    def _periodically_log_metrics(self, reward_data, brain_metrics):
        if self.steps_done % self.cfg["training"]["fast_update_frequency"] != 0:
            return

        metrics = {"reward/total": reward_data["total"]}

        # Log individual reward components for debugging
        components = reward_data.get("components", {})
        if components:
            for comp_name, comp_value in components.items():
                metrics[f"reward/{comp_name}"] = comp_value

        metrics.update(self.director.get_goal_metrics())
        metrics.update(brain_metrics)
        self.logger.log_step(metrics, self.steps_done)

    def _log_progress(self, info, reward_data, pbar):
        self.session_reward += reward_data["total"]
        self.session_steps += 1

        brain_metrics = self.agent.get_metrics()
        epsilon = brain_metrics.get("brain/epsilon", 0.0)

        strategy_info = self.strategy_loader.get_strategy_info()
        goal_strat = strategy_info["goal_strategy"][:3].upper()  # LLM, HEU, NON

        # Add battle/enemy health info
        battle_active = info.get("battle_active", False)
        if battle_active:
            enemy_hp_current = info.get("enemy_hp_current", 0)
            enemy_hp_max = info.get("enemy_hp_max", 1)
            if enemy_hp_max > 0:
                enemy_hp_percent = (enemy_hp_current / enemy_hp_max) * 100
                location_info = f"⚔️ Enemy:{enemy_hp_current}/{enemy_hp_max}({enemy_hp_percent:.0f}%)"
            else:
                location_info = "⚔️ Battle"
        else:
            map_id = info.get("map_id")
            map_name = self.game_data.map_id_to_name(map_id) if map_id is not None else "Unknown"
            location_info = f"🌍 {map_name}"

        # Add cursor info if available
        cursor_display = getattr(self, '_current_cursor_display', '')
        cursor_info = f" | 📍{cursor_display}" if cursor_display else ""

        pbar.set_description(
            f"[{self.brain_type.upper()}|{goal_strat}] "
            f"Rew:{self.session_reward:.1f} | "
            f"ε:{epsilon:.3f} | "
            f"Badges:{self.current_badges} | "
            f"{location_info}{cursor_info}"
        )

    def _save_agent(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"agent_{self.brain_type}_{tag}.pth")
        self.agent.save_checkpoint(path)

    def _cleanup(self):
        self._save_agent(self.save_tag)
        if self.human_recorder:
            path = self.human_recorder.save()
            print(f"\nƒo\" Human buffer saved to {path}")
        self.logger.close()
        if hasattr(self, "director"):
            self.director.shutdown()
        self.env.close()

    def _maybe_record_human_demo(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        local_action: int,
        env_action: int,
        reward: float,
        done: bool,
        text_embedding: np.ndarray,
        next_text_embedding: np.ndarray
    ) -> bool:
        if not self.human_recorder:
            return False
        self.human_recorder.record(
            obs=obs,
            action=local_action,
            env_action=env_action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            text_embedding=text_embedding,
            next_text_embedding=next_text_embedding
        )
        return self.human_recorder.is_full


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
    record_human_buffer_path: Optional[str] = None,
    human_buffer_steps: int = 2000,
    emulation_speed_override: Optional[int] = None,
    **brain_overrides
):
    """
    Train Pokemon agent through unified training pipeline.

    Args:
        brain_type: "crossq", "bbf", "rainbow", or "human" (keyboard control)
        strategy_preset: "llm", "heuristic", "reactive", or "hybrid"
        memory_preset: "minimal", "low", "medium", or "high"
        brain_overrides: Additional brain config overrides

    When brain_type="human":
        - You control the game via keyboard
        - All training infrastructure runs normally (rewards, memory, etc.)
        - Perfect for debugging game integration
    """
    # Load base config
    cfg = TrainingConfig.load(config_path)
    total_override = human_buffer_steps if record_human_buffer_path else total_steps_override
    cfg = TrainingConfig.apply_overrides(cfg, headless, total_override, state_path_override)
    if emulation_speed_override is not None:
        env_cfg = cfg.get("environment", {}).copy()
        env_cfg["emulation_speed"] = emulation_speed_override
        cfg["environment"] = env_cfg

    # Apply strategy preset
    cfg = load_strategy_preset(strategy_preset, cfg)

    # Load brain config (skip for human brain)
    if brain_type.lower() == "human":
        brain_config = {}  # No config needed for human
    else:
        brain_loader = BrainConfigLoader()
        brain_config = brain_loader.get_brain_config(brain_type, memory_preset, brain_overrides)
        vram_estimate = brain_loader.estimate_vram_usage(brain_config)
        print(f"\n{'='*70}")
        print(f"RL TRAINING - Pluggable Architecture")
        print(f"{'='*70}")
        print(f"Brain: {brain_type.upper()} | Memory: {memory_preset.upper()}")
        print(f"Strategy: {strategy_preset.upper()}")
        print(f"Estimated VRAM: {vram_estimate['total_mb']:.0f} MB")
        print(f"{'='*70}\n")

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")

    human_recorder = None
    if record_human_buffer_path:
        if brain_type.lower() != "human":
            raise ValueError("--record-human-buffer requires --brain human")
        human_recorder = HumanExperienceRecorder(
            capacity=human_buffer_steps,
            output_path=record_human_buffer_path,
            text_feature_dim=TEXT_EMBED_DIM
        )
        print(f"ƒo\" Recording up to {human_buffer_steps} human steps at {record_human_buffer_path}")

    trainer = PokemonTrainer(
        cfg, brain_type, brain_config, run_name,
        checkpoint_root, save_tag, device, seed, human_recorder=human_recorder
    )
    trainer.setup()
    trainer.run()
    return {"done": True, "brain": brain_type}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Pokemon Red AI with fully pluggable architecture"
    )

    # Main args
    parser.add_argument("--config", type=str, help="Path to hyperparameters.yaml")
    parser.add_argument("--brain", type=str, default="crossq",
                        choices=["crossq", "bbf", "rainbow", "human"],
                        help="RL algorithm or 'human' for manual keyboard control")
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
    parser.add_argument("--emulation-speed", type=int,
                        help="Override environment emulation speed (1=normal, higher=faster).")

    # Brain overrides
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-capacity", type=int)
    parser.add_argument("--gamma", type=float)

    # Human buffer collection
    parser.add_argument("--record-human-buffer", type=str,
                        help="Path to save a human demonstration buffer (forces human brain)")
    parser.add_argument("--human-buffer-steps", type=int, default=2000,
                        help="Number of steps to record for the human buffer")

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
        record_human_buffer_path=args.record_human_buffer,
        human_buffer_steps=args.human_buffer_steps,
        emulation_speed_override=args.emulation_speed,
        **brain_overrides
    )
