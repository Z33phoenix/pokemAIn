"""
Shared utilities for phased specialist pretraining (navigation, battle, menu).

Each phase reuses the same Gym wrapper, reward system, and hierarchical agent
while constraining action selection to a single specialist head. This keeps the
training loop compatible with the multi-agent orchestrator and the existing
checkpoint layout (nav_brain_*.pth, battle_brain_*.pth, menu_brain_*.pth).
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.train_end_to_end import (  # noqa: E402
    _apply_overrides,
    _prepare_dirs,
    _seed_everything,
    load_config,
    make_buffers,
)
from src.agent.hierarchical_agent import HierarchicalAgent  # noqa: E402
from src.env.pokemon_red_gym import PokemonRedGym  # noqa: E402
from src.env.rewards import RewardSystem  # noqa: E402
from src.utils.logger import Logger  # noqa: E402

PHASES = {"nav", "battle", "menu"}


def _phase_reward(phase: str, components: Dict[str, float], training_cfg: Dict[str, Any]) -> float:
    """Combine global shaping with the phase-specific component."""
    total = components.get("global_reward", 0.0)
    if phase == "nav":
        total += components.get("nav_reward", 0.0)
    elif phase == "battle":
        total += components.get("battle_reward", 0.0)
    elif phase == "menu":
        total += components.get("menu_reward", 0.0)
    clip = training_cfg.get("reward_clip", np.inf)
    return float(np.clip(total, -clip, clip))


def _menu_goal_ctx(info: Dict[str, Any], menu_goal_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the goal context passed to the menu specialist so it can learn to
    reach a specific cursor position / menu item.
    """
    target_cursor = info.get("menu_cursor")
    if target_cursor is None:
        target_cursor = (menu_goal_defaults.get("cursor_row"), menu_goal_defaults.get("cursor_col"))
    return {
        "goal_type": "menu",
        "target": {
            "menu_target": info.get("menu_target", menu_goal_defaults.get("menu_target")),
            "cursor": target_cursor,
            "menu_depth": info.get("menu_depth", menu_goal_defaults.get("menu_depth")),
        },
        "metadata": {"menu_open": info.get("menu_open", False)},
    }


def _select_action(
    phase: str,
    agent: HierarchicalAgent,
    features: torch.Tensor,
    info: Dict[str, Any],
    epsilon: float,
    device: torch.device,
    menu_goal_defaults: Dict[str, Any],
) -> Tuple[int, int, Optional[Dict[str, Any]], Optional[torch.Tensor]]:
    """Route to the requested specialist and keep track of goal metadata."""
    goal_ctx = None
    goal_embedding = None
    if phase == "nav":
        local_action = agent.nav_brain.get_action(features, epsilon, goal=None)
        action = agent.nav_actions[local_action]
    elif phase == "battle":
        local_action = agent.battle_brain.get_action(features, goal=None)
        action = agent.battle_actions[local_action]
    else:
        goal_ctx = _menu_goal_ctx(info, menu_goal_defaults)
        goal_embedding = agent.menu_brain.encode_goal(goal_ctx, device=device)
        local_action = agent.menu_brain.get_action(features, goal_embedding, epsilon=min(epsilon, 0.2))
        action = agent.menu_actions[local_action]
    return action, local_action, goal_ctx, goal_embedding


def _phase_buffer_fill(phase: str, buffers: Dict[str, Any], replay_cfg: Dict[str, Any]) -> float:
    """Return the fill ratio for the active replay buffer."""
    if phase == "nav":
        buf = buffers["nav"]
        cap = replay_cfg["nav"]["size"]
    elif phase == "battle":
        buf = buffers["battle"]
        cap = replay_cfg["battle"]["size"]
    else:
        buf = buffers["menu"]
        cap = replay_cfg["menu"]["size"]
    return len(buf) / cap


def train_phase(
    phase: str,
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
    Train a single specialist head while keeping the director's encoder
    synchronized. Returns a summary compatible with train_multi_agent.
    """
    phase = phase.lower()
    if phase not in PHASES:
        raise ValueError(f"Unknown phase '{phase}'. Expected one of {sorted(PHASES)}")

    cfg = cfg or load_config(config_path)
    phase_state_map = cfg.get("environment", {}).get("phase_states", {})
    effective_state = state_path_override or phase_state_map.get(phase)
    cfg = _apply_overrides(
        cfg,
        headless=headless,
        total_steps_override=total_steps_override,
        state_path_override=effective_state,
    )
    training_cfg = cfg["training"]
    epsilon_cfg = training_cfg["epsilon"]

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = _seed_everything(seed)
    run_label = run_name or f"{phase}_phase"
    print(f"--- STARTING {phase.upper()} PHASE ON {device} ({run_label}) [seed={seed}] ---")

    checkpoint_dir, log_dir = _prepare_dirs(run_label, checkpoint_root, log_root)

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
    buffers = {"nav": nav_buffer, "battle": battle_buffer, "menu": menu_buffer}
    logger = Logger(log_dir=log_dir, run_name=run_label)

    total_steps = training_cfg["total_steps"]
    save_freq = training_cfg["save_frequency"]
    fast_freq = training_cfg["fast_update_frequency"]
    warmup = training_cfg["warmup_steps"]
    batch_size = training_cfg["batch_size"]

    epsilon = epsilon_cfg["start"]
    epsilon_end = epsilon_cfg["end"]
    epsilon_decay = epsilon_cfg["decay"]

    try:
        obs, info = env.reset()
    except RuntimeError as exc:
        # Handle user-closing the PyBoy window gracefully.
        print(f"[WARN] Environment reset failed: {exc}")
        return {
            "phase": phase,
            "run_name": run_label,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "best_episode_reward": None,
            "best_nav_loss": None,
            "best_battle_loss": None,
            "best_menu_loss": None,
            "seed": seed,
            "window_closed": True,
        }
    reward_sys.reset()
    episode_reward = 0.0
    best_episode_reward = float("-inf")
    best_nav_loss = float("inf")
    best_battle_loss = float("inf")
    best_menu_loss = float("inf")
    menu_goal_defaults = cfg.get("director", {}).get("goals", {}).get("menu", {})

    pbar = tqdm(range(1, total_steps + 1))

    for step in pbar:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        features = agent.director.encoder(obs_t)
        action, local_action, goal_ctx, goal_embedding = _select_action(
            phase, agent, features, info, epsilon, device, menu_goal_defaults
        )

        next_obs, _, terminated, truncated, next_info = env.step(action)
        reward_components = reward_sys.compute_components(
            next_info, env.pyboy.memory, next_obs, action, goal_ctx=goal_ctx
        )
        total_reward = _phase_reward(phase, reward_components, training_cfg)

        nav_battle_abort = phase == "nav" and next_info.get("battle_active", False)
        done = terminated or truncated or nav_battle_abort
        episode_reward += total_reward

        goal_embedding_np = (
            goal_embedding.detach().cpu().numpy().squeeze(0) if goal_embedding is not None else None
        )

        if phase == "nav":
            nav_buffer.add(obs, local_action, total_reward, next_obs, done)
        elif phase == "battle":
            battle_buffer.add(obs, local_action, total_reward, next_obs, done)
        else:
            menu_buffer.add(
                obs,
                local_action,
                total_reward,
                next_obs,
                done,
                goal=goal_embedding_np,
                next_goal=goal_embedding_np,
                goal_ctx=goal_ctx,
                next_goal_ctx=goal_ctx,
            )

        if step > warmup and step % fast_freq == 0:
            metrics: Dict[str, float] = {}

            if phase == "nav" and len(nav_buffer) > batch_size:
                s, a, r, ns, d, w, idx = nav_buffer.sample(batch_size)
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
                    agent.save_component("director", checkpoint_dir=checkpoint_dir, tag="best_nav")

            if phase == "battle" and len(battle_buffer) > batch_size:
                s, a, r, ns, d, w, idx = battle_buffer.sample(batch_size)
                s_feat = agent.director.encoder(s)
                ns_feat = agent.director.encoder(ns)
                battle_loss, battle_stats = agent.battle_brain.train_step_return_loss(
                    s_feat, a, r, ns_feat, d
                )

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
                    agent.save_component("director", checkpoint_dir=checkpoint_dir, tag="best_battle")

            if phase == "menu" and len(menu_buffer) > batch_size:
                sample = menu_buffer.sample(batch_size, include_context=True)
                s, a, r, ns, d, w, idx, gctx, ngctx = sample
                s_feat = agent.director.encoder(s)
                ns_feat = agent.director.encoder(ns)
                goal_embed = agent.menu_brain.encode_goal_batch(gctx, device=device)
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
                    agent.save_component("director", checkpoint_dir=checkpoint_dir, tag="best_menu")

            metrics["policy/epsilon"] = epsilon
            metrics["buffer/fill"] = _phase_buffer_fill(phase, buffers, training_cfg["replay"])
            if metrics:
                logger.log_step(metrics, step)

        obs = next_obs
        info = next_info

        if epsilon > epsilon_end:
            epsilon -= (epsilon_cfg["start"] - epsilon_end) / epsilon_decay

        pbar.set_description(
            f"[{run_label}] Rew: {episode_reward:.2f} | Eps: {epsilon:.2f} | Phase: {phase}"
        )

        if info.get("window_closed"):
            print("[WARN] PyBoy window closed by user; ending training loop.")
            break

        if done:
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                agent.save_component(phase, checkpoint_dir=checkpoint_dir, tag="best_reward")
                agent.save_component("director", checkpoint_dir=checkpoint_dir, tag="best_reward")
            episode_reward = 0.0
            try:
                obs, info = env.reset()
            except RuntimeError as exc:
                print(f"[WARN] Environment reset failed: {exc}")
                break
            reward_sys.reset()

        if step % save_freq == 0:
            agent.save_component("director", checkpoint_dir=checkpoint_dir, tag=save_tag)
            agent.save_component(phase, checkpoint_dir=checkpoint_dir, tag=save_tag)

    agent.save_component("director", checkpoint_dir=checkpoint_dir, tag=save_tag)
    agent.save_component(phase, checkpoint_dir=checkpoint_dir, tag=save_tag)
    logger.close()
    env.close()

    if best_episode_reward == float("-inf"):
        best_episode_reward = None
    best_nav_loss = None if best_nav_loss == float("inf") else best_nav_loss
    best_battle_loss = None if best_battle_loss == float("inf") else best_battle_loss
    best_menu_loss = None if best_menu_loss == float("inf") else best_menu_loss

    return {
        "phase": phase,
        "run_name": run_label,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": logger.log_dir,
        "best_episode_reward": best_episode_reward,
        "best_nav_loss": best_nav_loss if phase == "nav" else None,
        "best_battle_loss": best_battle_loss if phase == "battle" else None,
        "best_menu_loss": best_menu_loss if phase == "menu" else None,
        "seed": seed,
    }


def build_phase_arg_parser(default_phase: str) -> Any:
    """Common CLI options for the phase-specific wrappers."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Train the {default_phase} specialist head.")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameter YAML.")
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
        help="Override environment state path for this phase (e.g., a battle or menu save state).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override torch device, e.g., cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser
