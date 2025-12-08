import argparse
import os
import random
import time
from typing import Any, Dict, Tuple, Optional, Set
from collections import deque
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.agent.director import Goal, Director
from src.agent.agent import CrossQNavAgent
from src.agent.goal_llm import PokemonGoalLLM
from src.agent.quest_manager import QuestManager
from src.agent.director import LLMGoalCoordinator
from src.env.pokemon_red_gym import PokemonRedGym
from src.env.rewards import RewardSystem
from src.utils.logger import Logger


class TrainingConfig:
    """Handles loading and overriding configuration dictionaries."""

    @staticmethod
    def load(config_path: str | None = None) -> Dict[str, Any]:
        """Load hyperparameters YAML from the provided or default path."""
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
        """Apply command-line overrides to the configuration."""
        cfg = cfg.copy()
        env_cfg = cfg.get("environment", {}).copy()
        
        if headless is not None:
            env_cfg["headless"] = headless
        
        # Adjust emulation speed based on headless state
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
    Main training controller for the Pokemon Red Reinforcement Learning agent.
    Encapsulates environment, agent, director, and training loop state.
    """
    
    # Strict list of allowed goal types as defined by the user
    ALLOWED_GOAL_TYPES: Set[str] = {
        "NAVIGATE", "INTERACT", "BATTLE", "MENU", "SEARCH", "PUZZLE"
    }

    def __init__(self, cfg: Dict[str, Any], run_name: str | None, 
                 checkpoint_root: str, save_tag: str, device: str, seed: int | None):
        self.cfg = cfg
        self.run_name = run_name
        self.save_tag = save_tag
        self.device = torch.device(device)
        self.seed = self._seed_everything(seed)
        
        self.checkpoint_dir, self.log_dir = self._prepare_dirs(checkpoint_root)
        self.logger = Logger(log_dir=self.log_dir, run_name=run_name)
        
        # Training State
        self.epsilon = cfg["training"]["epsilon"]["start"]
        self.best_reward = float("-inf")
        self.best_loss = None
        self.episode_count = 0
        self.steps_done = 0
        self.total_steps = cfg["training"]["total_steps"]
        
        # LLM/Goal State
        self.llm_goal_counter = 0
        self.last_completed_goal_count = 0
        self.goal_defaults = cfg["director"].get("goals", {})

    def _seed_everything(self, seed: int | None) -> int:
        """Seed Python, NumPy, and Torch RNGs."""
        if seed is None:
            seed = int(torch.randint(0, 10**6, (1,)).item())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"--- STARTING TRAINING ON {self.device} [seed={seed}] ---")
        return seed

    def _prepare_dirs(self, base_root: str) -> Tuple[str, str]:
        """Create and return checkpoint and log directories."""
        ckpt_dir = os.path.join(base_root, self.run_name) if self.run_name else base_root
        logs_root = os.path.join(base_root, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(logs_root, exist_ok=True)
        return ckpt_dir, logs_root

    def setup(self):
        """Initialize all major components (Env, Agent, Director, LLM)."""
        self.env = PokemonRedGym(self.cfg["environment"])
        self.reward_sys = RewardSystem(self.cfg["rewards"])
        self.director = Director(self.cfg["director"])
        self._setup_quest_manager()
        self._setup_agent()
        self._setup_llm()
        self.agent_buffer = deque(maxlen=50)

    def _setup_quest_manager(self):
        """Initialize QuestManager with config paths."""
        llm_cfg = self.cfg.get("goal_llm", {})
        self.quest_manager = QuestManager(
            walkthrough_path=llm_cfg.get("walkthrough_path", "config/walkthrough_steps.json"),
            save_dir=self.cfg.get("saves_dir", "saves"),
        )

    def _setup_agent(self):
        """Initialize the CrossQNavAgent and load weights if available."""
        agent_cfg = self.cfg.get("agent", {}).copy()
        agent_actions = agent_cfg.get("allowed_actions", list(range(self.env.action_space.n)))
        agent_cfg["action_dim"] = len(agent_actions)
        agent_cfg["input_dim"] = agent_cfg.get("input_dim", 96 * 96)
        
        self.agent = CrossQNavAgent(agent_cfg).to(self.device)
        self.agent_actions = agent_actions
        
        path = os.path.join(self.checkpoint_dir, f"agent_brain_{self.save_tag}.pth")
        if os.path.exists(path):
            self.agent.load_state_dict(torch.load(path, map_location=self.device))

    def _setup_llm(self):
        """Initialize the PokemonGoalLLM."""
        d_cfg = self.cfg.get("director", {})
        g_cfg = self.cfg.get("goal_llm") or d_cfg.get("goal_llm") or {}
        enabled = g_cfg.get("enabled", True)
        self.goal_llm = PokemonGoalLLM(
            api_url=g_cfg.get("api_url", "http://localhost:11434/api/chat"),
            model=g_cfg.get("model", "pokemon-goal"),
            enabled=bool(enabled),
            timeout=g_cfg.get("timeout", 25.0),
            debug=bool(g_cfg.get("debug", self.cfg.get("debug", False))),
        )

    def run(self):
        """Execute the main training loop."""
        obs, info = self._reset_env()
        self._initial_goal_request(info)
        
        pbar = tqdm(total=self.total_steps)
        while self.steps_done < self.total_steps:
            self.steps_done += 1
            metrics = self._training_step(obs, info)
            
            # Update loop state
            obs, info = metrics["next_obs"], metrics["next_info"]
            self._log_progress(metrics, pbar)
            
            if metrics["done"]:
                obs, info = self._handle_episode_end(metrics["reward_total"], metrics["goal_success"])
                
            if self.steps_done % self.cfg["training"]["save_frequency"] == 0:
                self._save_agent(self.save_tag)
                
            pbar.update(1)
        
        self._cleanup()

    def _reset_env(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and load the latest quest checkpoint."""
        obs, info = self.env.reset()
        self.reward_sys.reset()
        
        ckpt_path = self.quest_manager.get_checkpoint_path()
        if os.path.exists(ckpt_path):
            self.env.load_state(ckpt_path)
            obs, info = self.env._get_obs(), self.env._get_info()
            
        self.last_stage_save = ckpt_path
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.goal_step_count = 0
        return obs, info

    def _initial_goal_request(self, info: Dict[str, Any]):
        """Request the first goal before the loop starts."""
        self._manage_goals(info, step=0)

    def _training_step(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a single step: Goal check -> Action -> Env Step -> Train."""
        self._manage_goals(info, self.steps_done)
        self._ensure_active_goal(info)
        
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_data = self._select_action(obs_t, info)
        
        next_obs, _, terminated, truncated, next_info = self.env.step(action_data["env_action"])
        
        reward_data = self._compute_rewards(info, next_info, next_obs, action_data)
        done = terminated or truncated or reward_data["goal_success"] or reward_data["goal_timeout"]
        
        loss_data = self._update_agent(obs_t, action_data, reward_data, next_obs, done)
        
        self._update_epsilon()
        self._periodically_log_metrics(reward_data, loss_data)
        
        return {
            "next_obs": next_obs, "next_info": next_info, "done": done,
            "reward_total": reward_data["total"], "goal_success": reward_data["goal_success"]
        }

    def _manage_goals(self, info: Dict[str, Any], step: int):
        """Check logic to request a new goal from the LLM."""
        if not self.goal_llm.enabled: return

        self.director.prune_expired_goals(step)
        completed = len(self.director.completed_goals)
        queue_empty = len(self.director.goal_queue) == 0
        no_active = self.director.active_goal is None
        
        # Only request if we have no goals and have completed something new (or it's start)
        should_request = queue_empty and no_active and (completed > self.last_completed_goal_count or self.llm_goal_counter == 0)
        
        if should_request:
            self._request_goal_from_llm(info, step)

    def _request_goal_from_llm(self, info: Dict[str, Any], step: int):
        """Build state summary, query LLM, and enqueue the result if valid."""
        state = self._build_llm_state(info)
        goal_json = self.goal_llm.generate_goal(state)
        
        if not goal_json or not isinstance(goal_json, dict): return

        try:
            goal_json = LLMGoalCoordinator.sanitize_goal_response(goal_json, self.goal_defaults, info)
            if self._validate_goal_type(goal_json):
                self._enqueue_sanitized_goal(goal_json, info, step)
                self.last_completed_goal_count = len(self.director.completed_goals)
            else:
                if self.cfg.get("debug"): 
                    print(f"[LLM][WARN] Invalid goal type received: {goal_json.get('goal_type')}")
        except Exception as e:
            if self.cfg.get("debug"): print(f"[LLM][ERROR] Goal sanitization failed: {e}")

    def _validate_goal_type(self, goal_json: Dict[str, Any]) -> bool:
        """Strictly enforce that the goal is one of the 6 allowed types."""
        g_type = goal_json.get("goal_type")
        return g_type in self.ALLOWED_GOAL_TYPES

    def _build_llm_state(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Construct the state payload for the LLM."""
        if self.llm_goal_counter == 0 and len(self.director.completed_goals) == 0:
            # Canonical startup payload
            return {
                "location": {"map_name": "Red's house 2F", "map_id": 38, "x": 3, "y": 6, "nearby_sprites": []},
                "party": [], "inventory": {"key_items": [], "hms_owned": [], "items": []},
                "game_state": {"badges": 0, "money": 3000, "battle_status": "Overworld"},
                "last_goal": {"target": "Start Game", "status": "New"},
            }
        return LLMGoalCoordinator.build_state_summary(info, self.reward_sys)

    def _enqueue_sanitized_goal(self, goal_json: Dict[str, Any], info: Dict[str, Any], step: int):
        """Convert JSON goal to Goal object and enqueue."""
        self.llm_goal_counter += 1
        goal = Goal(
            name=goal_json.get("name") or f"llm-{self.llm_goal_counter}",
            goal_type=goal_json.get("goal_type"),  # Already validated
            priority=int(goal_json.get("priority", 0) or 0),
            target=goal_json.get("target") or {},
            metadata=goal_json.get("metadata") or {},
            max_steps=int(goal_json.get("max_steps", 0) or 0),
            goal_vector=goal_json.get("goal_vector"),
        )
        self.director.enqueue_goal(goal, current_step=step)

    def _ensure_active_goal(self, info: Dict[str, Any]):
        """Ensure Director has an active goal; fallback to NAVIGATE if needed."""
        if self.director.active_goal is not None: return

        if self.director.goal_queue:
            self.director._activate_goal(self.director.goal_queue.pop(0))
            self.goal_step_count = 0
        elif not self.goal_llm.enabled:
            # Fallback only when LLM is OFF. Default to NAVIGATE (Explore).
            fallback = self._create_fallback_goal(info)
            self.director.enqueue_goal(fallback)
            self.director._activate_goal(self.director.goal_queue.pop(0))
            self.goal_step_count = 0

    def _create_fallback_goal(self, info: Dict[str, Any]) -> Goal:
        """Create a default NAVIGATE goal when LLM is disabled."""
        max_steps = self.goal_defaults.get("explore", {}).get("max_steps", 256)
        return Goal(
            name="fallback-navigate", 
            goal_type="NAVIGATE", 
            priority=0,
            target={"map_id": info.get("map_id"), "map_name": info.get("map_name")},
            metadata={}, 
            max_steps=max_steps, 
            goal_vector=None
        )

    def _select_action(self, obs_t: torch.Tensor, info: Dict[str, Any]) -> Dict[str, Any]:
        """Consult Director and Agent to determine the next action."""
        specialist_idx, features, forced_action, goal_ctx = self.director.select_specialist(obs_t, info, self.epsilon)
        encoded_features = self.director.encode_features(obs_t).to(self.device)
        
        if forced_action is not None:
            return {"local": 0, "env_action": forced_action, "goal_ctx": goal_ctx, "encoded": encoded_features}
            
        local_action = self.agent.get_action(encoded_features, self.epsilon, goal_ctx)
        return {
            "local": local_action,
            "env_action": self.agent_actions[local_action],
            "goal_ctx": goal_ctx,
            "encoded": encoded_features
        }

    def _compute_rewards(self, info: Dict[str, Any], next_info: Dict[str, Any], 
                         next_obs: np.ndarray, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate rewards, handle map changes, and check goal completion."""
        # 1. Base Rewards
        comps = self.reward_sys.compute_components(
            next_info, self.env.pyboy.memory, next_obs, action_data["env_action"], goal_ctx=action_data["goal_ctx"]
        )
        
        # 2. Map Change Handling
        if next_info.get("map_id") != info.get("map_id"):
            if self.director.active_goal: self.director.complete_goal(status="map_changed")
            self.director.goal_queue.clear() # clear micro-goals
            self.goal_step_count = 0
            self._manage_goals(next_info, self.steps_done) # replan immediately
        
        # 3. Goal Completion
        completion_reward, success = self._check_goal_completion(next_info)
        
        # 4. Timeout Check
        self.goal_step_count += 1
        max_steps = (self.director.active_goal.max_steps if self.director.active_goal else 256)
        timeout = self.goal_step_count >= max_steps
        if timeout: completion_reward -= 10.0

        total = comps.get("global_reward", 0.0) + comps.get("agent_reward", 0.0) + comps.get("goal_bonus", 0.0) + completion_reward
        clipped = float(np.clip(total, -self.cfg["training"].get("reward_clip", np.inf), self.cfg["training"].get("reward_clip", np.inf)))
        
        return {
            "total": clipped, "goal_bonus": comps.get("goal_bonus", 0.0) + completion_reward,
            "goal_success": success, "goal_timeout": timeout, "components": comps
        }

    def _check_goal_completion(self, next_info: Dict[str, Any]) -> Tuple[float, bool]:
        """Check if active goal criteria are met by the new state."""
        if not self.director.active_goal: return 0.0, False
        
        target = self.director.active_goal.target or {}
        reached_id = target.get("map_id") is not None and next_info.get("map_id") == target.get("map_id")
        reached_name = target.get("map_name") and next_info.get("map_name") == target.get("map_name")
        
        if reached_id or reached_name:
            self.director.complete_goal(status="done")
            self.goal_step_count = 0
            # Check for Stage Save (Ratchet)
            if self.quest_manager.check_completion(self._build_llm_state(next_info)):
                self.env.save_state(self.quest_manager.get_checkpoint_path())
                self.last_stage_save = self.quest_manager.get_checkpoint_path()
            return self.cfg["rewards"].get("goal_bonus", {}).get("explore", {}).get("map_match", 1.0), True
            
        return 0.0, False

    def _update_agent(self, obs_t: torch.Tensor, action_data: Dict[str, Any], 
                      reward_data: Dict[str, Any], next_obs: np.ndarray, done: bool) -> Dict[str, Any]:
        """Perform a single Backprop step on the Agent."""
        next_feat = self.director.encode_features(torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)).to(self.device)
        rewards_t = torch.tensor([reward_data["total"]], dtype=torch.float32, device=self.device)
        dones_t = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=self.device)
        actions_t = torch.tensor([action_data["local"]], dtype=torch.int64, device=self.device)
        
        loss, stats = self.agent.train_step_return_loss(
            action_data["encoded"], actions_t, rewards_t, next_feat, dones_t
        )
        
        self.agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.agent.optimizer.step()
        
        if self.best_loss is None or loss.item() < self.best_loss:
            self.best_loss = loss.item()
            
        return {"loss": loss.item(), "stats": stats}

    def _update_epsilon(self):
        """Decay epsilon."""
        cfg = self.cfg["training"]["epsilon"]
        if self.epsilon > cfg["end"]:
            self.epsilon -= (cfg["start"] - cfg["end"]) / cfg["decay"]

    def _periodically_log_metrics(self, reward_data: Dict[str, Any], loss_data: Dict[str, Any]):
        """Log granular step metrics at configured frequency."""
        if self.steps_done % self.cfg["training"]["fast_update_frequency"] != 0: return
        
        metrics = {
            "policy/epsilon": self.epsilon,
            "reward/total": reward_data["total"],
            "reward/goal_bonus": reward_data["goal_bonus"],
            "loss/agent": loss_data["loss"]
        }
        metrics.update(loss_data["stats"])
        metrics.update(self.director.get_goal_metrics())
        self.logger.log_step(metrics, self.steps_done)

    def _log_progress(self, metrics: Dict[str, Any], pbar: tqdm):
        """Update progress bar description."""
        self.episode_reward += metrics["reward_total"]
        self.episode_steps += 1
        
        g_name = self.director.active_goal.name if self.director.active_goal else "none"
        pbar.set_description(
            f"[{self.run_name or 'solo'}] Rew: {self.episode_reward:.2f} | Ep: {self.episode_count} | {self.director.active_goal.goal_type}: {g_name}"
        )

    def _handle_episode_end(self, final_step_reward: float, goal_success: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Log episode summary, save best model, and reset/ratchet environment."""
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self._save_agent("best_reward")
            
        self.logger.log_step({
            "episode/reward": self.episode_reward,
            "episode/length": float(self.episode_steps),
        }, self.steps_done)
        
        self.episode_count += 1
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Ratchet Logic: If goal success, keep going. If fail, reload last stage save.
        if goal_success:
            return self.env._get_obs(), self.env._get_info()
        
        if self.last_stage_save and os.path.exists(self.last_stage_save):
            self.env.load_state(self.last_stage_save)
            return self.env._get_obs(), self.env._get_info()
            
        return self.env.reset()

    def _save_agent(self, tag: str):
        """Save agent weights to disk."""
        torch.save(self.agent.state_dict(), os.path.join(self.checkpoint_dir, f"agent_brain_{tag}.pth"))

    def _cleanup(self):
        """Close resources."""
        self._save_agent(self.save_tag)
        self.logger.close()
        self.env.close()


def train(
    config_path: str | None = None,
    run_name: str | None = None,
    checkpoint_root: str = "experiments",
    save_tag: str = "latest",
    total_steps_override: int | None = None,
    headless: bool | None = None,
    state_path_override: str | None = None,
    device_override: str | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Entry point to initialize and run the Trainer."""
    cfg = TrainingConfig.load(config_path)
    cfg = TrainingConfig.apply_overrides(cfg, headless, total_steps_override, state_path_override)
    
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = PokemonTrainer(cfg, run_name, checkpoint_root, save_tag, device, seed)
    trainer.setup()
    trainer.run()
    
    return {
        "run_name": run_name or "solo",
        "checkpoint_dir": trainer.checkpoint_dir,
        "best_reward": trainer.best_reward,
        "seed": trainer.seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single hierarchical agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameters YAML.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging.")
    parser.add_argument("--checkpoint-root", type=str, default="experiments", help="Base dir for checkpoints.")
    parser.add_argument("--save-tag", type=str, default="latest", help="Tag for checkpoints.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--headless", dest="headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--windowed", dest="headless", action="store_false", help="Force windowed mode.")
    parser.set_defaults(headless=None)
    parser.add_argument("--state-path", type=str, default=None, help="Override environment state path.")
    parser.add_argument("--device", type=str, default=None, help="Override torch device.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    train(
        config_path=args.config,
        run_name=args.run_name,
        checkpoint_root=args.checkpoint_root,
        save_tag=args.save_tag,
        total_steps_override=args.total_steps,
        headless=args.headless,
        state_path_override=args.state_path,
        device_override=args.device,
        seed=args.seed,
    )