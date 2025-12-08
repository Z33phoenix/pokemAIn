import argparse
import os
import random
import time
from typing import Any, Dict, Tuple
import numpy as np
import torch
import yaml
from tqdm import tqdm
from collections import deque

from src.agent.director import Goal, LLMGoalCoordinator, Director
from src.agent.agent import CrossQNavAgent
from src.agent.goal_llm import PokemonGoalLLM
from src.agent.quest_manager import QuestManager
from src.env.pokemon_red_gym import PokemonRedGym
from src.env.rewards import RewardSystem
from src.utils.logger import Logger


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load hyperparameters YAML from the provided path or default location."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "hyperparameters.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(
    cfg: Dict[str, Any], headless: bool | None, total_steps_override: int | None, state_path_override: str | None = None
) -> Dict[str, Any]:
    """Return a shallow-copied config with the requested overrides applied."""
    cfg = cfg.copy()
    env_cfg = cfg.get("environment", {}).copy()

    if headless is not None:
        env_cfg["headless"] = headless

    if state_path_override:
        env_cfg["state_path"] = state_path_override

    effective_headless = env_cfg.get("headless")
    if effective_headless is True:
        env_cfg["emulation_speed"] = 0
    elif effective_headless is False:
        env_cfg["emulation_speed"] = 4

    cfg["environment"] = env_cfg

    if total_steps_override is not None:
        training_cfg = cfg.get("training", {}).copy()
        training_cfg["total_steps"] = total_steps_override
        cfg["training"] = training_cfg

    return cfg


def _prepare_dirs(run_name: str | None, base_root: str) -> Tuple[str, str]:
    """
    Create checkpoint and log roots. Checkpoints live under base/run_name;
    logs live under base/logs/.
    """
    checkpoint_dir = os.path.join(base_root, run_name) if run_name else base_root
    logs_root = os.path.join(base_root, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    return checkpoint_dir, logs_root


def _load_agent(agent: CrossQNavAgent, checkpoint_dir: str, tag: str, device: torch.device):
    """Load agent brain weights if available."""
    path = os.path.join(checkpoint_dir, f"agent_brain_{tag}.pth")
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        agent.load_state_dict(state)


def _save_agent(agent: CrossQNavAgent, checkpoint_dir: str, tag: str):
    """Persist agent brain weights."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_brain_{tag}.pth"))


def _seed_everything(seed: int | None) -> int:
    """Seed Python, NumPy, and Torch RNGs; return the effective seed."""
    if seed is None:
        seed = int(torch.randint(0, 10**6, (1,)).item())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def train(
    cfg: Dict[str, Any] | None = None,
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
    """
    Run a single training job. Returns a small summary dict that can be used by
    multi-agent orchestrators.
    """
    cfg = cfg or load_config(config_path)
    cfg = _apply_overrides(
        cfg,
        headless=headless,
        total_steps_override=total_steps_override,
        state_path_override=state_path_override,
    )
    training_cfg = cfg["training"]
    epsilon_cfg = training_cfg["epsilon"]

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = _seed_everything(seed)
    print(f"--- STARTING TRAINING ON {device} ({'run ' + run_name if run_name else 'single'}) [seed={seed}] ---")

    checkpoint_dir, log_dir = _prepare_dirs(run_name, checkpoint_root)

    env = PokemonRedGym(cfg["environment"])
    reward_sys = RewardSystem(cfg["rewards"])
    director = Director(cfg["director"])
    quest_manager = QuestManager(
        walkthrough_path=cfg.get("goal_llm", {}).get("walkthrough_path", "config/walkthrough_steps.json"),
        save_dir=cfg.get("saves_dir", "saves"),
    )
    agent_cfg = cfg.get("agent", {})
    agent_actions = agent_cfg.get("allowed_actions", list(range(env.action_space.n)))
    agent_cfg = agent_cfg.copy()
    agent_cfg["action_dim"] = len(agent_actions)
    agent_cfg["input_dim"] = agent_cfg.get("input_dim", 96 * 96)
    agent = CrossQNavAgent(agent_cfg).to(device)
    _load_agent(agent, checkpoint_dir, save_tag, device)

    agent_buffer = deque(maxlen=50)
    logger = Logger(log_dir=log_dir, run_name=run_name)
    global_debug = bool(cfg.get("debug", False))
    director_cfg = cfg.get("director", {})
    goal_llm_cfg = cfg.get("goal_llm") or director_cfg.get("goal_llm") or {}
    goal_llm_enabled = goal_llm_cfg.get("enabled")
    if goal_llm_enabled is None:
        goal_llm_enabled = True
    goal_llm_debug = goal_llm_cfg.get("debug", global_debug)
    goal_llm = PokemonGoalLLM(
        api_url=goal_llm_cfg.get("api_url", "http://localhost:11434/api/chat"),
        model=goal_llm_cfg.get("model", "pokemon-goal"),
        enabled=bool(goal_llm_enabled),
        timeout=goal_llm_cfg.get("timeout", 25.0),
        debug=bool(goal_llm_debug),
    )
    if goal_llm.debug or global_debug:
        print(f"[LLM][DEBUG] Config | enabled={goal_llm.enabled} url={goal_llm.api_url} model={goal_llm.model} timeout={goal_llm.timeout}")

    goal_defaults = cfg["director"].get("goals", {})

    total_steps = training_cfg["total_steps"]
    save_freq = training_cfg["save_frequency"]
    fast_freq = training_cfg["fast_update_frequency"]
    warmup = training_cfg["warmup_steps"]
    batch_size = training_cfg["batch_size"]

    epsilon = epsilon_cfg["start"]
    epsilon_end = epsilon_cfg["end"]
    epsilon_decay = epsilon_cfg["decay"]
    llm_goal_counter = 0
    last_completed_goal_count = 0
    last_goal_bonus = 0.0
    llm_retry_interval = 0
    last_llm_request_step = 0
    llm_future = None

    def _sanitize_goal(goal_json: Dict[str, Any], current_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize goal JSON coming from the LLM (text-only)."""
        return LLMGoalCoordinator.sanitize_goal_response(goal_json, goal_defaults, current_info)

    def maybe_request_new_goal(current_info: Dict[str, Any], step: int):
        """Request a new goal only when none are active/queued."""
        nonlocal last_completed_goal_count, llm_goal_counter, last_llm_request_step
        if not goal_llm.enabled:
            if goal_llm.debug:
                print("[LLM][DEBUG] Goal LLM disabled; skipping request")
            return

        # Drop any queued goals that have expired while we were busy.
        director.prune_expired_goals(step)

        completed_count = len(director.completed_goals)
        queue_len = len(director.goal_queue)
        has_active = director.active_goal is not None
        need_goal = queue_len == 0 and not has_active and (completed_count > last_completed_goal_count or llm_goal_counter == 0)
        if not need_goal or queue_len:
            if goal_llm.debug:
                print(f"[LLM][DEBUG] No new goal needed at step {step} | queue_len={queue_len} active={has_active} completed={completed_count} last_completed={last_completed_goal_count} llm_goal_counter={llm_goal_counter}")
            return
        last_llm_request_step = step
        # For the very first prompt, send the canonical startup payload.
        if llm_goal_counter == 0 and completed_count == 0:
            state_summary = {
                "location": {
                    "map_name": "Red's house 2F",
                    "map_id": 38,
                    "x": 3,
                    "y": 6,
                    "nearby_sprites": [],
                },
                "party": [],
                "inventory": {"key_items": [], "hms_owned": [], "items": []},
                "game_state": {"badges": 0, "money": 3000, "battle_status": "Overworld"},
                "last_goal": {"target": "Start Game", "status": "New"},
            }
        else:
            state_summary = LLMGoalCoordinator.build_state_summary(current_info, reward_sys)
        if goal_llm.debug:
            print(f"[LLM][DEBUG] Submitting goal request at step {step} | active_goal={director.active_goal}")
        goal_json = goal_llm.generate_goal(state_summary)
        completed_count = len(director.completed_goals)
        if goal_json is None or not isinstance(goal_json, dict):
            return
        try:
            goal_json = _sanitize_goal(goal_json, current_info)
        except Exception:
            return
        if goal_json.get("goal_type") == "survive" and (current_info.get("party_size", 0) or 0) <= 0:
            goal_json = {
                "goal_type": "explore",
                "priority": max(1, int(goal_defaults.get("explore", {}).get("priority", 1) or 1)),
                "target": {
                    "novel_states": goal_defaults.get("explore", {}).get("novel_states", 5),
                    "prefer_new_map": True,
                },
                "metadata": {"start_map": current_info.get("map_id")},
                "max_steps": goal_defaults.get("explore", {}).get("max_steps", 256),
                "goal_vector": goal_json.get("goal_vector"),
            }
        llm_goal_counter += 1
        goal = Goal(
            name=goal_json.get("name") or f"llm-{llm_goal_counter}",
            goal_type=goal_json.get("goal_type", "explore"),
            priority=int(goal_json.get("priority", 0) or 0),
            target=goal_json.get("target") or {},
            metadata=goal_json.get("metadata") or {},
            max_steps=int(goal_json.get("max_steps", 0) or 0),
            goal_vector=goal_json.get("goal_vector"),
        )
        director.enqueue_goal(goal, current_step=step)
        last_completed_goal_count = completed_count

    obs, info = env.reset()
    reward_sys.reset()
    # Attempt to load the furthest checkpoint for the current stage.
    current_checkpoint_path = quest_manager.get_checkpoint_path()
    if os.path.exists(current_checkpoint_path):
        env.load_state(current_checkpoint_path)
        obs = env._get_obs()
        info = env._get_info()
    maybe_request_new_goal(info, step=0)
    episode_reward = 0.0
    episode_steps = 0

    best_episode_reward = float("-inf")
    best_agent_loss = None
    last_agent_loss = None
    last_agent_loss_stats: Dict[str, float] = {}
    last_goal_name: str | None = None
    goal_completion_bonus = cfg["rewards"].get("goal_bonus", {}).get("explore", {}).get("map_match", 1.0)
    goal_timeout_penalty = -10.0
    last_map_id = info.get("map_id")
    goal_step_count = 0
    episode_count = 0
    last_stage_save = current_checkpoint_path

    pbar = tqdm(total=total_steps)
    steps_done = 0

    while steps_done < total_steps:
        current_step = steps_done + 1
        maybe_request_new_goal(info, step=current_step)

        # If no goal is active yet, wait until one arrives.
        if director.active_goal is None:
            if director.goal_queue:
                director._activate_goal(director.goal_queue.pop(0))
                goal_step_count = 0
            elif goal_llm.enabled:
                time.sleep(0.05)
                continue
            else:
                # Fallback: create a simple explore goal when LLM is disabled.
                director.enqueue_goal(
                    Goal(
                        name="fallback-explore",
                        goal_type="explore",
                        priority=0,
                        target={"map_id": info.get("map_id"), "map_name": info.get("map_name")},
                        metadata={},
                        max_steps=goal_defaults.get("explore", {}).get("max_steps", 256),
                        goal_vector=None,
                    )
                )
                director._activate_goal(director.goal_queue.pop(0))
                goal_step_count = 0

        # Director + agent: flatten observations and select action.
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        specialist_idx, features, forced_action, goal_ctx = director.select_specialist(obs_t, info, epsilon)
        action_meta = {"local_action": None, "goal": goal_ctx}
        encoded_features = director.encode_features(obs_t).to(device)
        if forced_action is not None:
            action = forced_action
            action_meta["local_action"] = 0
        else:
            local_action = agent.get_action(encoded_features, epsilon, goal_ctx)
            action = agent_actions[local_action]
            action_meta["local_action"] = local_action
        next_obs, _, terminated, truncated, next_info = env.step(action)
        goal_ctx = action_meta.get("goal")
        reward_components = reward_sys.compute_components(
            next_info, env.pyboy.memory, next_obs, action, goal_ctx=goal_ctx
        )
        # Detect map changes; replan on any change.
        current_map_id = next_info.get("map_id")
        if current_map_id is not None and current_map_id != last_map_id:
            last_map_id = current_map_id
            if director.active_goal:
                director.complete_goal(status="map_changed")
            # Drop any queued micro-goals on map transition to avoid stale targets.
            director.goal_queue.clear()
            goal_step_count = 0
            maybe_request_new_goal(next_info, step=current_step)
            if director.active_goal is None and director.goal_queue:
                director._activate_goal(director.goal_queue.pop(0))
                goal_step_count = 0

        goal_bonus = reward_components.get("goal_bonus", 0.0)
        agent_reward_component = reward_components.get("agent_reward", reward_components.get("nav_reward", 0.0))
        agent_total_reward = reward_components.get("global_reward", 0.0) + agent_reward_component + goal_bonus
        completion_reward = 0.0
        goal_success = False
        goal_timeout = False
        # Goal completion: mark done and grant bonus when new state matches target.
        if director.active_goal:
            target = director.active_goal.target or {}
            target_map_id = target.get("map_id")
            target_map_name = target.get("map_name")
            new_map_id = next_info.get("map_id")
            new_map_name = next_info.get("map_name")
            reached_map = False
            if target_map_id is not None and new_map_id == target_map_id:
                reached_map = True
            elif target_map_name and new_map_name == target_map_name:
                reached_map = True
            if reached_map:
                completion_reward = goal_completion_bonus
                director.complete_goal(status="done")
                goal_success = True
                # Request a new goal right away.
                maybe_request_new_goal(next_info, step=current_step)
                if director.active_goal is None and director.goal_queue:
                    director._activate_goal(director.goal_queue.pop(0))
                    goal_step_count = 0
                # Stage progression check and save.
                state_summary = LLMGoalCoordinator.build_state_summary(next_info, reward_sys)
                if quest_manager.check_completion(state_summary):
                    stage_save_path = quest_manager.get_checkpoint_path()
                    env.save_state(stage_save_path)
                    last_stage_save = stage_save_path

        agent_total_reward += completion_reward
        goal_bonus += completion_reward
        last_goal_bonus = goal_bonus
        agent_total_reward = float(
            np.clip(
                agent_total_reward,
                -training_cfg.get("reward_clip", np.inf),
                training_cfg.get("reward_clip", np.inf),
            )
        )
        goal_embedding_np = None

        done = terminated or truncated or goal_success or goal_timeout

        # Goal timeout handling (per-goal episode length).
        goal_step_count += 1
        active_goal = director.active_goal
        max_goal_steps = None
        if active_goal:
            max_goal_steps = active_goal.max_steps or goal_defaults.get("explore", {}).get("max_steps", 256)
        if max_goal_steps and goal_step_count >= max_goal_steps:
            agent_total_reward += goal_timeout_penalty
            goal_bonus += goal_timeout_penalty
            goal_timeout = True
            goal_step_count = 0

        # Online TD update for the single agent and loss tracking.
        next_features = director.encode_features(
            torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        ).to(device)
        rewards_t = torch.tensor([agent_total_reward], dtype=torch.float32, device=device)
        dones_t = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=device)
        actions_t = torch.tensor([local_action], dtype=torch.int64, device=device)
        loss, loss_stats = agent.train_step_return_loss(
            encoded_features, actions_t, rewards_t, next_features, dones_t
        )
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        agent.optimizer.step()
        last_agent_loss = loss.item()
        last_agent_loss_stats = loss_stats
        if best_agent_loss is None or last_agent_loss < best_agent_loss:
            best_agent_loss = last_agent_loss

        active_goal = director.active_goal
        current_goal_name = active_goal.name if active_goal else None
        if current_goal_name != last_goal_name and bool(cfg.get("debug", False)):
            print(
                f"[DEBUG][GOAL] Active goal changed | Step:{current_step}: "
                f"{last_goal_name or 'none'} -> {current_goal_name or 'none'} "
                f"(type={active_goal.goal_type if active_goal else 'none'}, "
                f"priority={active_goal.priority if active_goal else 'n/a'})"
            )
            last_goal_name = current_goal_name

        if specialist_idx == 0:
            episode_reward += agent_total_reward
        else:
            episode_reward += agent_total_reward
        episode_steps += 1

        local_action = action_meta.get("local_action")
        if local_action is None:
            raise ValueError("Missing local action for specialist index {}".format(specialist_idx))
        agent_buffer.append((info.get("map_id"), info.get("x"), info.get("y")))

        if current_step % fast_freq == 0:
            metrics: Dict[str, float] = {
                "policy/epsilon": epsilon,
                "reward/global": float(reward_components.get("global_reward", 0.0)),
                "reward/agent": float(agent_reward_component),
                "reward/goal_bonus": float(goal_bonus),
                "reward/total": float(agent_total_reward),
                "episode/steps": float(episode_steps),
                "goal/bonus": float(last_goal_bonus),
            }
            if last_agent_loss is not None:
                metrics["loss/agent"] = float(last_agent_loss)
            for k, v in last_agent_loss_stats.items():
                metrics[k] = float(v)
            if director.active_goal:
                metrics["goal/priority"] = float(director.active_goal.priority)
            metrics.update(director.get_goal_metrics())
            logger.log_step(metrics, current_step)

        obs = next_obs
        info = next_info

        if epsilon > epsilon_end:
            epsilon -= (epsilon_cfg["start"] - epsilon_end) / epsilon_decay

        goal_meta = {}
        if director.active_goal:
            goal_meta = director.active_goal.as_dict()
        elif action_meta.get("goal"):
            goal_meta = action_meta.get("goal") or {}
        goal_name = goal_meta.get("goal_type") or goal_meta.get("name") or "none"
        goal_target = goal_meta.get("target") or {}
        target_label = goal_target.get("map_name") or goal_target.get("map_id") or goal_target or "none"
        goal_display = f"{goal_name} -> {target_label}"
        pbar.set_description(
            f"[{run_name or 'solo'}] Rew: {episode_reward:.2f} | Ep: {episode_count} | {goal_display}"
        )

        if done:
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                _save_agent(agent, checkpoint_dir, tag="best_reward")
            # Log episodic summaries.
            logger.log_step(
                {
                    "episode/reward": float(episode_reward),
                    "episode/length": float(episode_steps),
                    "policy/epsilon": epsilon,
                    "loss/agent": float(last_agent_loss) if last_agent_loss is not None else None,
                },
                current_step,
            )
            episode_reward = 0.0
            episode_steps = 0
            episode_count += 1
            if goal_success:
                # Stay in the current world state and continue.
                obs = next_obs
                info = next_info
            else:
                # Reload from latest stage checkpoint (ratchet) on episode end.
                if last_stage_save and os.path.exists(last_stage_save):
                    env.load_state(last_stage_save)
                    obs = env._get_obs()
                    info = env._get_info()
                else:
                    obs, info = env.reset()
            reward_sys.reset()
            if director.active_goal is None and not director.goal_queue:
                maybe_request_new_goal(info, step=current_step + 1)

        if current_step % save_freq == 0:
            _save_agent(agent, checkpoint_dir, tag=save_tag)

        steps_done += 1
        pbar.update(1)

    _save_agent(agent, checkpoint_dir, tag=save_tag)
    try:
        llm_executor.shutdown(wait=False)
    except Exception:
        pass
    logger.close()
    env.close()
    return {
        "run_name": run_name or "solo",
        "checkpoint_dir": checkpoint_dir,
        "log_dir": logger.log_dir,
        "best_episode_reward": best_episode_reward,
        "best_agent_loss": best_agent_loss,
        "seed": seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single hierarchical agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to a hyperparameter YAML file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name (used for dirs/logging).")
    parser.add_argument("--checkpoint-root", type=str, default="experiments", help="Base directory for run folders containing checkpoints and logs.")
    parser.add_argument("--save-tag", type=str, default="latest", help="Tag used when saving checkpoints.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total training steps from config.")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Force PyBoy to run headless regardless of config.",
    )
    parser.add_argument(
        "--windowed",
        dest="headless",
        action="store_false",
        help="Force PyBoy to create a window regardless of config.",
    )
    parser.set_defaults(headless=None)
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Override environment state path (useful for phased training).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override torch device, e.g., cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")

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
