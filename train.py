import argparse
import os
import glob
import random
from typing import Any, Dict, Tuple, Optional, Set
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.agent.director import Goal, Director, LLMGoalCoordinator, EpisodicMemory
from src.agent.agent import CrossQNavAgent
from src.agent.goal_llm import PokemonGoalLLM
from src.env.pokemon_red_gym import PokemonRedGym
from src.env.rewards import RewardSystem
from src.env.text_decoder import TextDecoder
from src.utils.logger import Logger
from src.utils.game_data import map_id_to_name  # <--- NEW IMPORT

class TrainingConfig:
    """Handles loading and overriding configuration dictionaries."""

    @staticmethod
    def load(config_path: str | None = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config", "hyperparameters.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def apply_overrides(cfg: Dict[str, Any], headless: bool | None, total_steps: int | None, state_path: str | None = None) -> Dict[str, Any]:
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
    Context-Aware Trainer. 
    Uses RAM text decoding and episodic memory to guide the agent without a walkthrough.
    """

    ALLOWED_GOAL_TYPES: Set[str] = {"NAVIGATE", "INTERACT", "BATTLE", "MENU", "SEARCH"}

    def __init__(self, cfg: Dict[str, Any], run_name: str | None, 
                 checkpoint_root: str, save_tag: str, device: str, seed: int | None):
        self.cfg = cfg
        self.run_name = run_name
        self.save_tag = save_tag
        self.device = torch.device(device)
        self.seed = self._seed_everything(seed)
        
        self.checkpoint_dir, self.log_dir = self._prepare_dirs(checkpoint_root)
        self.logger = Logger(log_dir=self.log_dir, run_name=run_name)
        
        self.epsilon = cfg["training"]["epsilon"]["start"]
        self.llm_update_freq = 300
        self.total_steps = cfg["training"]["total_steps"]
        self.steps_done = 0
        self.current_badges = 0
        self.goal_defaults = cfg["director"].get("goals", {})
        
        # Tracking
        self.best_reward = float("-inf")
        self.best_loss = None
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.llm_goal_counter = 0
        self.last_completed_goal_count = 0
        self.save_dir = cfg.get("saves_dir", "saves")
        os.makedirs(self.save_dir, exist_ok=True)

    def _seed_everything(self, seed: int | None) -> int:
        if seed is None: seed = int(torch.randint(0, 10**6, (1,)).item())
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
        self.env = PokemonRedGym(self.cfg["environment"])
        self.reward_sys = RewardSystem(self.cfg["rewards"])
        self.director = Director(self.cfg["director"])
        
        # New "Senses"
        self.text_decoder = TextDecoder(self.env.pyboy)
        self.memory = EpisodicMemory() # Corrected: No capacity arg
        
        self._setup_agent()
        self._setup_llm()

    def _setup_agent(self):
        agent_cfg = self.cfg.get("agent", {}).copy()
        agent_actions = agent_cfg.get("allowed_actions", list(range(self.env.action_space.n)))
        agent_cfg["action_dim"] = len(agent_actions)
        
        # Explicitly set input_dim to 9216 (96x96 pixels)
        agent_cfg["input_dim"] = 96 * 96 
        
        self.agent = CrossQNavAgent(agent_cfg).to(self.device)
        self.agent_actions = agent_actions
        
        path = os.path.join(self.checkpoint_dir, f"agent_brain_{self.save_tag}.pth")
        if os.path.exists(path):
            try:
                self.agent.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Loaded agent from {path}")
            except RuntimeError as e:
                print(f"Could not load agent weights due to shape mismatch (ignoring old weights): {e}")

    def _setup_llm(self):
        g_cfg = self.cfg.get("goal_llm", {})
        self.goal_llm = PokemonGoalLLM(
            api_url=g_cfg.get("api_url", "http://localhost:11434/api/chat"),
            model=g_cfg.get("model", "pokemon-goal"),
            enabled=g_cfg.get("enabled", True),
            timeout=50.0
        )

    def run(self):
        obs, info = self._reset_env()
        self._poll_llm(info, force=True)

        pbar = tqdm(total=self.total_steps)
        while self.steps_done < self.total_steps:
            self.steps_done += 1
            
            # 1. Sense: Read Text & Update Memory
            self._update_memory(info)
            
            # 2. Think: Poll LLM Periodically
            if self.steps_done % self.llm_update_freq == 0:
                self._poll_llm(info)

            # 3. Act: Select & Execute Action
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_data = self._select_action(obs_t, info)
            next_obs, _, terminated, truncated, next_info = self.env.step(action_data["env_action"])
            
            # 4. Learn
            reward_data = self._compute_rewards(info, next_info, next_obs, action_data)
            
            # --- FIX: Correct Method Name Call ---
            self._update_agent(obs_t, action_data, reward_data, next_obs, terminated or truncated)
            
            self._update_epsilon()
            self._periodically_log_metrics(reward_data)
            self._log_progress(info, reward_data, pbar)

            # Loop upkeep
            obs, info = next_obs, next_info
            
            # Episode End Handling
            if terminated or truncated or reward_data["goal_success"]:
                if terminated or truncated:
                     obs, info = self._handle_episode_end(reward_data["total"], False)
                elif reward_data["goal_success"]:
                     pass 

            if self.steps_done % self.cfg["training"]["save_frequency"] == 0:
                self._save_agent(self.save_tag)
                
            pbar.update(1)
            
        self._cleanup()

    def _reset_env(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and load the latest BADGE checkpoint."""
        obs, info = self.env.reset()
        self.reward_sys.reset()
        
        # Load latest badge save if available
        ckpt_path = self._get_latest_checkpoint()
        if ckpt_path and os.path.exists(ckpt_path):
            self.env.load_state(ckpt_path)
            obs, info = self.env._get_obs(), self.env._get_info()
            
        self.current_badges = info.get("badges", 0)
        self.last_stage_save = ckpt_path
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.goal_step_count = 0
        return obs, info

    def _get_latest_checkpoint(self) -> str | None:
        """Find the save file with the highest badge count."""
        saves = glob.glob(os.path.join(self.save_dir, "badge_*.state"))
        if not saves: return None
        try:
            latest = max(saves, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
            return latest
        except:
            return None

    def _update_memory(self, info: Dict[str, Any]):
        """Check for sensory inputs and log them."""
        # Map Change
        current_map_id = info.get("map_id")
        current_map_name = map_id_to_name(current_map_id) # <--- UPDATED: Use util
        
        if not hasattr(self, "_last_map"): self._last_map = current_map_name
        if current_map_name != self._last_map:
            self.memory.log_event(self.steps_done, "MAP_CHANGE", f"Entered {current_map_name}")
            self._last_map = current_map_name
            
        # Text (The Eyes) - Only read if NOT moving to avoid visual tearing
        decoded_text = self.text_decoder.read_current_text()
        if decoded_text:
            self.memory.log_event(self.steps_done, "TEXT", decoded_text)

    def _poll_llm(self, info: Dict[str, Any], force: bool = False):
        """Construct prompt with memory log and query LLM."""
        if not self.goal_llm.enabled: return
        
        # 1. Get History
        context_log = self.memory.consume_history()
        
        # 2. Build Nearby Entities String (Define it before use!)
        valid_moves = []
        conns = info.get("map_connections", {})
        for direction, details in conns.items():
            if details.get("exists"):
                valid_moves.append(f"Exit {direction.title()}")

        warps = info.get("map_warps", [])
        for w in warps:
            valid_moves.append(f"Warp at ({w['x']},{w['y']})")

        # Sprites
        sprites = info.get("sprites", [])
        for s in sprites:
            # Output: "TRAINER at (5,5)" or "ITEM at (3,3)"
            valid_moves.append(f"{s['type']} at ({s['x']},{s['y']})")
            
        nearby_entities_str = "; ".join(valid_moves) if valid_moves else "None visible"

        # 3. Build Payload
        state_summary = {
            "history": context_log,
            "current_location": map_id_to_name(info.get("map_id")), # <--- UPDATED: Use util
            "nearby_entities": nearby_entities_str,
            "party_size": info.get("party_size"),
            "badges": info.get("badges"),
            "last_goal": self.director.active_goal.as_dict() if self.director.active_goal else None
        }
        
        goal_json = self.goal_llm.generate_goal(state_summary)
        
        if goal_json and goal_json.get("goal_type") in self.ALLOWED_GOAL_TYPES:
            if "priority" not in goal_json: goal_json["priority"] = 1
            if "max_steps" not in goal_json: goal_json["max_steps"] = 300
            
            target = goal_json.get("target", {})
            metadata = {"reasoning": goal_json.get("thought_process", "")}
            
            goal_vector = None
            if "x" in target and "y" in target:
                metadata["target_x"] = target["x"]
                metadata["target_y"] = target["y"]
                metadata["target_action"] = target.get("action", "move")
                
                curr_x, curr_y = info.get("x", 0), info.get("y", 0)
                dx = target["x"] - curr_x
                dy = target["y"] - curr_y
                mag = (dx**2 + dy**2)**0.5
                if mag > 0:
                    goal_vector = [dx / mag, dy / mag]
                else:
                    goal_vector = [0.0, 0.0]

            goal = Goal(
                name=goal_json.get("name", f"llm-{self.llm_goal_counter}"),
                goal_type=goal_json.get("goal_type"),
                target=target,
                max_steps=int(goal_json.get("max_steps")),
                priority=int(goal_json.get("priority")),
                metadata=metadata,
                goal_vector=goal_vector
            )
            self.llm_goal_counter += 1
            self.director.goal_queue.clear()
            self.director.enqueue_goal(goal, self.steps_done)

    def _select_action(self, obs_t: torch.Tensor, info: Dict[str, Any]) -> Dict[str, Any]:
        if not self.director.active_goal and not self.director.goal_queue:
             self._ensure_fallback_goal(info)

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
    
    def _ensure_fallback_goal(self, info: Dict[str, Any]):
        fallback = Goal(
            name="fallback-explore", 
            goal_type="NAVIGATE", 
            priority=0,
            target={"map_id": info.get("map_id")}, 
            max_steps=200,
            metadata={},       # <--- Added missing argument
            goal_vector=None   # <--- Added for safety
        )
        self.director.enqueue_goal(fallback, self.steps_done)

    def _compute_rewards(self, info: Dict[str, Any], next_info: Dict[str, Any], 
                         next_obs: np.ndarray, action_data: Dict[str, Any]) -> Dict[str, Any]:
        
        comps = self.reward_sys.compute_components(
            next_info, self.env.pyboy.memory, next_obs, action_data["env_action"], goal_ctx=action_data["goal_ctx"]
        )
        
        if next_info.get("map_id") != info.get("map_id"):
            if self.director.active_goal: self.director.complete_goal(status="map_changed")
            self.director.goal_queue.clear() 
            self.goal_step_count = 0
            self._poll_llm(next_info) 
        
        completion_reward, success = self._check_goal_completion(next_info)
        
        self.goal_step_count += 1
        max_steps = (self.director.active_goal.max_steps if self.director.active_goal else 256)
        timeout = self.goal_step_count >= max_steps
        if timeout: completion_reward -= 5.0

        total = comps.get("global_reward", 0.0) + comps.get("agent_reward", 0.0) + comps.get("goal_bonus", 0.0) + completion_reward
        clipped = float(np.clip(total, -10, 10))
        
        return {
            "total": clipped, "goal_bonus": comps.get("goal_bonus", 0.0) + completion_reward,
            "goal_success": success, "goal_timeout": timeout, "components": comps
        }

    def _check_goal_completion(self, next_info: Dict[str, Any]) -> Tuple[float, bool]:
        new_badges = next_info.get("badges", 0)
        if new_badges > self.current_badges:
            self.current_badges = new_badges
            save_path = os.path.join(self.save_dir, f"badge_{new_badges}.state")
            self.env.save_state(save_path)
            self.memory.log_event(self.steps_done, "SYSTEM", f"Badge Earned! Saved to {save_path}")
            return 50.0, False 
            
        if not self.director.active_goal: return 0.0, False
        
        target = self.director.active_goal.target or {}
        reached = False
        
        if "x" in target and "y" in target:
            curr_x, curr_y = next_info.get("x", -999), next_info.get("y", -999)
            dist = ((curr_x - target["x"])**2 + (curr_y - target["y"])**2)**0.5
            if dist < 1.0:
                reached = True
        elif target.get("map_id") is not None and next_info.get("map_id") == target.get("map_id"): 
            reached = True
        elif target.get("map_name") and next_info.get("map_name") == target.get("map_name"): 
            reached = True
        
        if reached:
            self.director.complete_goal(status="done")
            self.goal_step_count = 0
            return 10.0, True
            
        return 0.0, False

    def _update_agent(self, obs_t, action_data, reward_data, next_obs, done):
        next_feat = self.director.encode_features(torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)).to(self.device)
        rewards_t = torch.tensor([reward_data["total"]], dtype=torch.float32, device=self.device)
        dones_t = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=self.device)
        actions_t = torch.tensor([action_data["local"]], dtype=torch.int64, device=self.device)
        
        loss, stats = self.agent.train_step_return_loss(action_data["encoded"], actions_t, rewards_t, next_feat, dones_t)
        
        self.agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.agent.optimizer.step()
        
        if self.best_loss is None or loss.item() < self.best_loss: self.best_loss = loss.item()
        return {"loss": loss.item(), "stats": stats}

    def _update_epsilon(self):
        cfg = self.cfg["training"]["epsilon"]
        if self.epsilon > cfg["end"]:
            self.epsilon -= (cfg["start"] - cfg["end"]) / cfg["decay"]

    def _periodically_log_metrics(self, reward_data):
        if self.steps_done % self.cfg["training"]["fast_update_frequency"] != 0: return
        metrics = {"policy/epsilon": self.epsilon, "reward/total": reward_data["total"]}
        metrics.update(self.director.get_goal_metrics())
        self.logger.log_step(metrics, self.steps_done)

    def _log_progress(self, info, reward_data, pbar):
        self.episode_reward += reward_data["total"]
        self.episode_steps += 1
        
        # Safe extraction of goal details
        if self.director.active_goal:
            g_name = self.director.active_goal.name
            g_type = self.director.active_goal.goal_type
        else:
            g_name = "none"
            g_type = "none"
        
        pbar.set_description(
            f"[{self.run_name}] Rew:{self.episode_reward:.1f} | "
            f"x,y:({info.get('x',0)},{info.get('y',0)}) | "
            f"Badges:{self.current_badges} | "
            f"{g_type}:{g_name}"
        )

    def _handle_episode_end(self, final_reward, success):
        self.logger.log_step({"episode/reward": self.episode_reward, "episode/badges": self.current_badges}, self.steps_done)
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
        torch.save(self.agent.state_dict(), os.path.join(self.checkpoint_dir, f"agent_brain_{tag}.pth"))

    def _cleanup(self):
        self._save_agent(self.save_tag)
        self.logger.close()
        self.env.close()

def train(config_path=None, run_name=None, checkpoint_root="experiments", save_tag="latest", total_steps_override=None, headless=None, state_path_override=None, device_override=None, seed=None):
    cfg = TrainingConfig.load(config_path)
    cfg = TrainingConfig.apply_overrides(cfg, headless, total_steps_override, state_path_override)
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PokemonTrainer(cfg, run_name, checkpoint_root, save_tag, device, seed)
    trainer.setup()
    trainer.run()
    return {"done": True}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--checkpoint-root", type=str, default="experiments")
    parser.add_argument("--save-tag", type=str, default="latest")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--windowed", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--state-path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    train(config_path=args.config, 
          run_name=args.run_name, 
          checkpoint_root=args.checkpoint_root, 
          save_tag=args.save_tag, 
          total_steps_override=args.total_steps, 
          headless=args.headless, 
          state_path_override=args.state_path, 
          device_override=args.device, 
          seed=args.seed
          )