import argparse
import os
import random
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.hierarchical_agent import HierarchicalAgent
from src.env.pokemon_red_gym import PokemonRedGym
from src.env.rewards import RewardSystem
from src.utils.logger import Logger
from src.utils.memory_buffer import PrioritizedReplayBuffer


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "hyperparameters.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_buffers(
    cfg: Dict, device: torch.device, obs_shape: Tuple[int, ...]
) -> Tuple[PrioritizedReplayBuffer, PrioritizedReplayBuffer, PrioritizedReplayBuffer]:
    replay_cfg = cfg["training"]["replay"]
    nav_cfg = replay_cfg["nav"]
    battle_cfg = replay_cfg["battle"]
    menu_cfg = replay_cfg["menu"]
    menu_goal_dim = cfg.get("specialists", {}).get("menu", {}).get("goal_dim")
    menu_goal_shape = (menu_goal_dim,) if menu_goal_dim else None
    nav_buffer = PrioritizedReplayBuffer(nav_cfg["size"], obs_shape, alpha=nav_cfg["alpha"], device=device)
    battle_buffer = PrioritizedReplayBuffer(battle_cfg["size"], obs_shape, alpha=battle_cfg["alpha"], device=device)
    menu_buffer = PrioritizedReplayBuffer(
        menu_cfg["size"],
        obs_shape,
        alpha=menu_cfg["alpha"],
        device=device,
        goal_shape=menu_goal_shape,
        store_context=True,
    )
    return nav_buffer, battle_buffer, menu_buffer


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
    checkpoint_dir = os.path.join(checkpoint_root, run_name) if run_name else checkpoint_root
    log_dir = os.path.join(log_root, run_name) if run_name else log_root
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return checkpoint_dir, log_dir


def _seed_everything(seed: int | None) -> int:
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

    total_steps = training_cfg["total_steps"]
    save_freq = training_cfg["save_frequency"]
    fast_freq = training_cfg["fast_update_frequency"]
    warmup = training_cfg["warmup_steps"]
    batch_size = training_cfg["batch_size"]

    epsilon = epsilon_cfg["start"]
    epsilon_end = epsilon_cfg["end"]
    epsilon_decay = epsilon_cfg["decay"]

    obs, info = env.reset()
    reward_sys.reset()
    episode_reward = 0.0

    best_episode_reward = float("-inf")
    best_nav_loss = float("inf")
    best_battle_loss = float("inf")
    best_menu_loss = float("inf")

    pbar = tqdm(range(1, total_steps + 1))

    for step in pbar:
        action, specialist_idx, action_meta = agent.get_action(obs, info, epsilon)
        next_obs, _, terminated, truncated, next_info = env.step(action)
        goal_ctx = action_meta.get("goal")
        reward_components = reward_sys.compute_components(
            next_info, env.pyboy.memory, next_obs, action, goal_ctx=goal_ctx
        )
        nav_total_reward = reward_components["global_reward"] + reward_components["nav_reward"]
        battle_total_reward = reward_components["global_reward"] + reward_components["battle_reward"]
        menu_total_reward = reward_components["menu_reward"]
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
            nav_buffer.add(obs, local_action, nav_total_reward, next_obs, done)
        elif specialist_idx == 1:
            battle_buffer.add(obs, local_action, battle_total_reward, next_obs, done)
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

            if len(battle_buffer) > batch_size:
                s, a, r, ns, d, w, idx = battle_buffer.sample(batch_size)
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

            metrics["policy/epsilon"] = epsilon
            metrics["buffer/nav_fill"] = len(nav_buffer) / training_cfg["replay"]["nav"]["size"]
            metrics["buffer/battle_fill"] = len(battle_buffer) / training_cfg["replay"]["battle"]["size"]
            metrics["buffer/menu_fill"] = len(menu_buffer) / training_cfg["replay"]["menu"]["size"]
            metrics.update(agent.director.get_goal_metrics())

            if metrics:
                logger.log_step(metrics, step)

        obs = next_obs
        info = next_info

        if epsilon > epsilon_end:
            epsilon -= (epsilon_cfg["start"] - epsilon_end) / epsilon_decay

        specialist_name = "Nav" if specialist_idx == 0 else ("Bat" if specialist_idx == 1 else "Menu")
        goal_name = action_meta.get("goal", {}).get("goal_type", "none")
        pbar.set_description(
            f"[{run_name or 'solo'}] Rew: {episode_reward:.2f} | Eps: {epsilon:.2f} | {specialist_name} -> {goal_name}"
        )

        if done:
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                agent.save(checkpoint_dir, tag="best_reward")
            episode_reward = 0.0
            obs, info = env.reset()
            reward_sys.reset()

        if step % save_freq == 0:
            agent.save(checkpoint_dir, tag=save_tag)

    agent.save(checkpoint_dir, tag=save_tag)
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
