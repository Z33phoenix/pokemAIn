import argparse
import base64
import io
import os
import random
import sys
from typing import Any, Dict, Tuple, List
import concurrent.futures

import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.director import Goal
from src.agent.hierarchical_agent import HierarchicalAgent
from src.agent.goal_llm import PokemonGoalLLM
from src.env.pokemon_red_gym import PokemonRedGym
from src.env.rewards import RewardSystem
from src.env import ram_map
from src.utils.logger import Logger
from src.utils.memory_buffer import PrioritizedReplayBuffer


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load hyperparameters YAML from the provided path or default location."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "hyperparameters.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_buffers(
    cfg: Dict, device: torch.device, obs_shape: Tuple[int, ...]
) -> Tuple[PrioritizedReplayBuffer, PrioritizedReplayBuffer, PrioritizedReplayBuffer]:
    """Instantiate replay buffers for nav, battle, and menu specialists."""
    replay_cfg = cfg["training"]["replay"]
    nav_cfg = replay_cfg["nav"]
    battle_cfg = replay_cfg["battle"]
    menu_cfg = replay_cfg["menu"]
    menu_goal_dim = cfg.get("specialists", {}).get("menu", {}).get("goal_dim")
    menu_goal_shape = (menu_goal_dim,) if menu_goal_dim else None
    nav_buffer = PrioritizedReplayBuffer(
        nav_cfg["size"], obs_shape, alpha=nav_cfg["alpha"], device=device, store_context=True
    )
    battle_buffer = PrioritizedReplayBuffer(
        battle_cfg["size"], obs_shape, alpha=battle_cfg["alpha"], device=device, store_context=True
    )
    menu_buffer = PrioritizedReplayBuffer(
        menu_cfg["size"],
        obs_shape,
        alpha=menu_cfg["alpha"],
        device=device,
        goal_shape=menu_goal_shape,
        store_context=True,
    )
    return nav_buffer, battle_buffer, menu_buffer


def _encode_obs_png_base64(obs: np.ndarray) -> str:
    """Convert the agent's observation (1x84x84 uint8) into a base64 PNG for the LLM."""
    arr = np.squeeze(obs)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_state_summary(info: Dict[str, Any], reward_sys: RewardSystem, director) -> Dict[str, Any]:
    """Compact snapshot fed to the goal-setting LLM."""
    last_goal = director.get_last_completed_goal() if hasattr(director, "get_last_completed_goal") else None
    last_goal_desc = None
    if last_goal:
        name = last_goal.get("name") or last_goal.get("goal_type", "goal")
        status = last_goal.get("status")
        target = last_goal.get("target")
        parts = [str(name)]
        if status:
            parts.append(str(status))
        if target:
            parts.append(str(target))
        last_goal_desc = " | ".join(parts)

    return {
        "state_map_id": info.get("map_id"),
        "state_x": info.get("x"),
        "state_y": info.get("y"),
        "state_party_size": info.get("party_size"),
        "state_hp_percent": info.get("hp_percent"),
        "state_visited_maps": sorted(list(reward_sys.visited_maps)),
        "state_seen_coords_count": len(reward_sys.seen_coords),
        "state_battle_active": info.get("battle_active"),
        "state_last_goal": last_goal_desc,
        "state_badge_count": info.get("badge_count"),
        "state_badges": info.get("badges", {}),
        "state_flags": info.get("quest_flags", {}),
        "state_party_power": info.get("party_power"),
    }


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


def _prepare_dirs(run_name: str | None, checkpoint_root: str, log_root: str) -> Tuple[str, str]:
    """Create checkpoint/log directories and return their resolved paths."""
    checkpoint_dir = os.path.join(checkpoint_root, run_name) if run_name else checkpoint_root
    log_dir = os.path.join(log_root, run_name) if run_name else log_root
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return checkpoint_dir, log_dir


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
    checkpoint_root: str = "checkpoints",
    log_root: str = "experiments/logs",
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

    checkpoint_dir, log_dir = _prepare_dirs(run_name, checkpoint_root, log_root)

    env = PokemonRedGym(cfg["environment"])
    reward_sys = RewardSystem(cfg["rewards"])
    agent = HierarchicalAgent(
        action_dim=env.action_space.n,
        device=device,
        director_cfg=cfg["director"],
        specialist_cfg=cfg["specialists"],
    )
    agent.load(checkpoint_dir=checkpoint_dir, tag=save_tag)

    vision_optimizer = optim.AdamW(
        agent.director.encoder.parameters(), lr=cfg["director"].get("vision_learning_rate", 1e-4)
    )

    nav_buffer, battle_buffer, menu_buffer = make_buffers(cfg, device, env.observation_space.shape)
    logger = Logger(log_dir=log_dir, run_name=run_name)
    goal_llm_cfg = cfg["director"].get("goal_llm", {})
    goal_llm = PokemonGoalLLM(
        api_url=goal_llm_cfg.get("api_url", "http://localhost:11434/api/chat"),
        model=goal_llm_cfg.get("model", "pokemon-goal"),
        enabled=bool(goal_llm_cfg.get("enabled", False)),
        timeout=goal_llm_cfg.get("timeout", 5.0),
    )
    router_pretrain_path = cfg["director"].get("router_pretrain_path")
    if router_pretrain_path and os.path.exists(router_pretrain_path):
        try:
            director_state = torch.load(router_pretrain_path, map_location=device, weights_only=True)
            agent.director.load_state_dict(director_state, strict=False)
            print(f"[INFO] Loaded router pretrain from {router_pretrain_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load router pretrain from {router_pretrain_path}: {exc}")
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
    llm_retry_interval = int(goal_llm_cfg.get("retry_interval", 32) or 32)
    last_llm_request_step = -llm_retry_interval
    llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    llm_future: concurrent.futures.Future | None = None

    def _sanitize_goal(goal_json: Dict[str, Any], current_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize goal JSON coming from the LLM."""
        g = goal_json.copy()
        gtype = g.get("goal_type", "explore") or "explore"
        defaults = goal_defaults.get(gtype, {})
        target = g.get("target") or {}
        if not isinstance(target, dict):
            target = {}
        goal_vec = g.get("goal_vector")
        normalized_vec = None
        if isinstance(goal_vec, (list, tuple)) and len(goal_vec) >= 2:
            normalized_vec = [0.0, 0.0, 0.0, 0.0]
            for i in range(min(4, len(goal_vec))):
                try:
                    normalized_vec[i] = float(goal_vec[i])
                except Exception:
                    normalized_vec[i] = 0.0

        if gtype == "explore":
            nov_default = defaults.get("novel_states", 5)
            nov_target = target.get("novel_states", nov_default)
            target["novel_states"] = max(1, int(nov_target if nov_target is not None else nov_default))
            target["prefer_new_map"] = bool(target.get("prefer_new_map", True))
        elif gtype == "train":
            bw_default = defaults.get("battles_won", 1)
            bw_target = target.get("battles_won", bw_default)
            target["battles_won"] = max(1, int(bw_target if bw_target is not None else bw_default))
        elif gtype == "survive":
            hp_default = defaults.get("hp_target", 0.7)
            hp_target = target.get("hp_target", hp_default)
            target["hp_target"] = float(hp_target if hp_target is not None else hp_default)
        elif gtype == "menu":
            target.setdefault("menu_action", "open_menu")

        g["goal_type"] = gtype
        g["priority"] = int(g.get("priority", defaults.get("priority", 1) or 1))
        g["target"] = target
        metadata = g.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if "start_map" not in metadata and current_info.get("map_id") is not None:
            metadata["start_map"] = current_info.get("map_id")
        g["metadata"] = metadata
        max_steps = g.get("max_steps") or defaults.get("max_steps", 256)
        g["max_steps"] = int(max_steps) if max_steps else 256
        if normalized_vec:
            g["goal_vector"] = normalized_vec
        return g

    def maybe_request_new_goal(current_info: Dict[str, Any], current_obs: np.ndarray, step: int):
        """Manage asynchronous LLM goal requests and enqueue validated goals."""
        nonlocal last_completed_goal_count, llm_goal_counter, last_llm_request_step, llm_future
        if not goal_llm.enabled:
            return

        # Drop any queued goals that have expired while we were busy.
        agent.director.prune_expired_goals(step)

        # Collect finished async result if present.
        if llm_future and llm_future.done():
            try:
                goal_json = llm_future.result()
            except Exception:
                goal_json = None
            llm_future = None
            completed_count = len(agent.director.completed_goals)
            if goal_json is None or not isinstance(goal_json, dict):
                # Re-poll immediately on bad/empty response.
                state_summary = build_state_summary(current_info, reward_sys, agent.director)
                obs_image_b64 = _encode_obs_png_base64(current_obs)
                llm_future = llm_executor.submit(goal_llm.generate_goal, state_summary, obs_image_b64)
                last_llm_request_step = step
                return
            try:
                goal_json = _sanitize_goal(goal_json, current_info)
            except Exception:
                # Resubmit a fresh request instead of crashing.
                state_summary = build_state_summary(current_info, reward_sys, agent.director)
                obs_image_b64 = _encode_obs_png_base64(current_obs)
                llm_future = llm_executor.submit(goal_llm.generate_goal, state_summary, obs_image_b64)
                last_llm_request_step = step
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
            agent.director.enqueue_goal(goal, current_step=step)
            last_completed_goal_count = completed_count
            return

        if llm_future:
            return

        completed_count = len(agent.director.completed_goals)
        queue_len = len(agent.director.goal_queue)
        has_active = agent.director.active_goal is not None
        # Request a new goal when none are queued; allow prefetching while a goal is active.
        need_goal = queue_len == 0 and (completed_count > last_completed_goal_count or llm_goal_counter == 0 or has_active)
        if not need_goal or queue_len:
            return
        if step - last_llm_request_step < llm_retry_interval:
            return
        last_llm_request_step = step
        state_summary = build_state_summary(current_info, reward_sys, agent.director)
        obs_image_b64 = _encode_obs_png_base64(current_obs)
        llm_future = llm_executor.submit(goal_llm.generate_goal, state_summary, obs_image_b64)

    obs, info = env.reset()
    reward_sys.reset()
    maybe_request_new_goal(info, obs, step=0)
    episode_reward = 0.0

    best_episode_reward = float("-inf")
    best_nav_loss = float("inf")
    best_battle_loss = float("inf")
    best_menu_loss = float("inf")
    last_goal_name: str | None = None

    pbar = tqdm(range(1, total_steps + 1))

    for step in pbar:
        maybe_request_new_goal(info, obs, step=step)
        action, specialist_idx, action_meta = agent.get_action(obs, info, epsilon)
        next_obs, _, terminated, truncated, next_info = env.step(action)
        goal_ctx = action_meta.get("goal")
        reward_components = reward_sys.compute_components(
            next_info, env.pyboy.memory, next_obs, action, goal_ctx=goal_ctx
        )
        goal_bonus = reward_components.get("goal_bonus", 0.0)
        nav_total_reward = reward_components["global_reward"] + reward_components["nav_reward"] + goal_bonus
        battle_total_reward = reward_components["global_reward"] + reward_components["battle_reward"] + goal_bonus
        menu_total_reward = reward_components["menu_reward"] + goal_bonus
        last_goal_bonus = goal_bonus
        nav_total_reward = float(
            np.clip(nav_total_reward, -training_cfg.get("reward_clip", np.inf), training_cfg.get("reward_clip", np.inf))
        )
        battle_total_reward = float(
            np.clip(
                battle_total_reward,
                -training_cfg.get("reward_clip", np.inf),
                training_cfg.get("reward_clip", np.inf),
            )
        )
        menu_total_reward = float(
            np.clip(
                menu_total_reward,
                -training_cfg.get("reward_clip", np.inf),
                training_cfg.get("reward_clip", np.inf),
            )
        )
        goal_embedding = action_meta.get("goal_embedding")
        goal_embedding_np = (
            goal_embedding.detach().cpu().numpy().squeeze(0) if goal_embedding is not None else None
        )

        active_goal = agent.director.active_goal
        current_goal_name = active_goal.name if active_goal else None
        if current_goal_name != last_goal_name:
            print(
                f"[DEBUG][GOAL] Active goal changed | Step:{step}: "
                f"{last_goal_name or 'none'} -> {current_goal_name or 'none'} "
                f"(type={active_goal.goal_type if active_goal else 'none'}, "
                f"priority={active_goal.priority if active_goal else 'n/a'})"
            )
            last_goal_name = current_goal_name

        done = terminated or truncated
        if specialist_idx == 0:
            episode_reward += nav_total_reward
        elif specialist_idx == 1:
            episode_reward += battle_total_reward
        else:
            episode_reward += menu_total_reward

        local_action = action_meta.get("local_action")
        if local_action is None:
            raise ValueError("Missing local action for specialist index {}".format(specialist_idx))
        if specialist_idx == 0:
            nav_buffer.add(
                obs,
                local_action,
                nav_total_reward,
                next_obs,
                done,
                goal=goal_embedding_np,
                next_goal=goal_embedding_np,
                goal_ctx=goal_ctx,
                next_goal_ctx=goal_ctx,
            )
        elif specialist_idx == 1:
            battle_buffer.add(
                obs,
                local_action,
                battle_total_reward,
                next_obs,
                done,
                goal=goal_embedding_np,
                next_goal=goal_embedding_np,
                goal_ctx=goal_ctx,
                next_goal_ctx=goal_ctx,
            )
        else:
            menu_buffer.add(
                obs,
                local_action,
                menu_total_reward,
                next_obs,
                done,
                goal=goal_embedding_np,
                next_goal=goal_embedding_np,
                goal_ctx=goal_ctx,
                next_goal_ctx=goal_ctx,
            )

        if step > warmup and step % fast_freq == 0:
            metrics: Dict[str, float] = {}

            if len(nav_buffer) > batch_size:
                s, a, r, ns, d, w, idx, gctx, ngctx = nav_buffer.sample(
                    batch_size, include_context=True
                )
                s_feat = agent.director.encoder(s)
                ns_feat = agent.director.encoder(ns)
                nav_loss, nav_stats = agent.nav_brain.train_step_return_loss(s_feat, a, r, ns_feat, d)

                vision_optimizer.zero_grad()
                agent.nav_brain.optimizer.zero_grad()
                nav_loss.backward()
                vision_optimizer.step()
                agent.nav_brain.optimizer.step()

                nav_buffer.update_priorities(idx, [nav_loss.item() + 1e-5] * batch_size)
                metrics.update(nav_stats)
                if nav_loss.item() < best_nav_loss:
                    best_nav_loss = nav_loss.item()
                    agent.save_component("nav", checkpoint_dir=checkpoint_dir, tag="best_nav")

            if len(battle_buffer) > batch_size:
                s, a, r, ns, d, w, idx, gctx, ngctx = battle_buffer.sample(
                    batch_size, include_context=True
                )
                s_feat = agent.director.encoder(s)
                ns_feat = agent.director.encoder(ns)
                battle_loss, battle_stats = agent.battle_brain.train_step_return_loss(s_feat, a, r, ns_feat, d)

                vision_optimizer.zero_grad()
                agent.battle_brain.optimizer.zero_grad()
                battle_loss.backward()
                vision_optimizer.step()
                agent.battle_brain.optimizer.step()

                battle_buffer.update_priorities(idx, [battle_loss.item() + 1e-5] * batch_size)
                metrics.update(battle_stats)
                if battle_loss.item() < best_battle_loss:
                    best_battle_loss = battle_loss.item()
                    agent.save_component("battle", checkpoint_dir=checkpoint_dir, tag="best_battle")

            if len(menu_buffer) > batch_size:
                sample = menu_buffer.sample(batch_size, include_goals=True, include_context=True)
                s, a, r, ns, d, w, idx, goal_embed, next_goal_embed, gctx, ngctx = sample
                s_feat = agent.director.encoder(s)
                ns_feat = agent.director.encoder(ns)
                if goal_embed is None:
                    goal_embed = agent.menu_brain.encode_goal_batch(gctx, device=device)
                if next_goal_embed is None:
                    next_goal_embed = agent.menu_brain.encode_goal_batch(ngctx, device=device)
                menu_loss, menu_stats = agent.menu_brain.train_step(
                    s_feat, goal_embed, a, r, ns_feat, next_goal_embed, d
                )

                vision_optimizer.zero_grad()
                agent.menu_brain.optimizer.zero_grad()
                menu_loss.backward()
                vision_optimizer.step()
                agent.menu_brain.optimizer.step()

                menu_buffer.update_priorities(idx, [menu_loss.item() + 1e-5] * batch_size)
                metrics.update(menu_stats)
                if menu_loss.item() < best_menu_loss:
                    best_menu_loss = menu_loss.item()
                    agent.save_component("menu", checkpoint_dir=checkpoint_dir, tag="best_menu")

            metrics["policy/epsilon"] = epsilon
            metrics["buffer/nav_fill"] = len(nav_buffer) / training_cfg["replay"]["nav"]["size"]
            metrics["buffer/battle_fill"] = len(battle_buffer) / training_cfg["replay"]["battle"]["size"]
            metrics["buffer/menu_fill"] = len(menu_buffer) / training_cfg["replay"]["menu"]["size"]
            metrics["goal/bonus"] = float(last_goal_bonus)
            metrics.update(agent.director.get_goal_metrics())

            if metrics:
                logger.log_step(metrics, step)

        obs = next_obs
        info = next_info

        if epsilon > epsilon_end:
            epsilon -= (epsilon_cfg["start"] - epsilon_end) / epsilon_decay

        specialist_name = "Nav" if specialist_idx == 0 else ("Bat" if specialist_idx == 1 else "Menu")
        # Prefer the director's active goal (which may have come from the LLM);
        # fall back to any goal context passed through the action metadata.
        goal_meta = {}
        if agent.director.active_goal:
            goal_meta = agent.director.active_goal.as_dict()
        elif action_meta.get("goal"):
            goal_meta = action_meta.get("goal") or {}
        goal_name = goal_meta.get("goal_type", "none")
        goal_pri = goal_meta.get("priority")
        goal_display = f"{goal_name}[p{goal_pri}]" if goal_pri is not None else goal_name
        pbar.set_description(
            f"[{run_name or 'solo'}] Rew: {episode_reward:.2f} | Eps: {epsilon:.2f} | {specialist_name} -> {goal_display}"
        )

        if done:
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                agent.save(checkpoint_dir, tag="best_reward")
            episode_reward = 0.0
            obs, info = env.reset()
            reward_sys.reset()
            maybe_request_new_goal(info, obs, step=step + 1)

        if step % save_freq == 0:
            agent.save(checkpoint_dir, tag=save_tag)

    agent.save(checkpoint_dir, tag=save_tag)
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
        "best_nav_loss": best_nav_loss,
        "best_battle_loss": best_battle_loss,
        "best_menu_loss": best_menu_loss,
        "seed": seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single hierarchical agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to a hyperparameter YAML file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name (used for dirs/logging).")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints", help="Base directory for checkpoints.")
    parser.add_argument("--log-root", type=str, default="experiments/logs", help="Base directory for TensorBoard logs.")
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
        log_root=args.log_root,
        save_tag=args.save_tag,
        total_steps_override=args.total_steps,
        headless=args.headless,
        state_path_override=args.state_path,
        device_override=args.device,
        seed=args.seed,
    )
