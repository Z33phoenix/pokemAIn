"""Tiny-RL trainer for the menu specialist.

This replaces the previous CNN-based MenuBrain training for the menu phase
on this branch. It trains TinyMenuAgent purely from RAM-exposed menu signals
with a simple DQN loop and a lightweight replay buffer.

Episodes:
  - Start from a save state (ideally already inside a menu).
  - Sample a random target index in [0, last_index].
  - Reward:
      +1.0  when the agent CONFIRMs the correct index while the menu is open.
      -1.0  when the menu closes before success.
      -0.01 per step otherwise.
    All rewards are thus in [-1, 1].
  - Episode ends on success, menu close, or step-limit from config.

The resulting weights are saved to `menu_brain_{tag}.pth` so that
HierarchicalAgent on this branch can load them as the menu specialist.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.train_end_to_end import (  # noqa: E402
    _apply_overrides,
    _prepare_dirs,
    _seed_everything,
    load_config,
)
from src.agent.specialists.menu_tiny_rl import TinyMenuAgent  # noqa: E402
from src.env.pokemon_red_gym import PokemonRedGym  # noqa: E402
from src.env import ram_map  # noqa: E402
from src.utils.logger import Logger  # noqa: E402


@dataclass
class SimpleReplayBuffer:
    capacity: int
    state_dim: int

    def __post_init__(self) -> None:
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.pos
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        size = len(self)
        idx = np.random.randint(0, size, size=batch_size)
        states = torch.as_tensor(self.states[idx], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idx], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


def _menu_active(info: Dict[str, Any]) -> bool:
    """Consistent check for interactive menus (mirrors RewardSystem._menu_active)."""
    menu_open = info.get("menu_open")
    has_options = info.get("menu_has_options")
    if menu_open is None and has_options is None:
        return False
    if menu_open is None:
        return bool(has_options)
    if has_options is None:
        return bool(menu_open)
    return bool(menu_open and has_options)


def _mart_context(info: Dict[str, Any], memory, mart_maps: list[int]) -> bool:
    """Detect whether the agent is in (or interacting with) a Poké Mart."""
    map_id = info.get("map_id")
    inventory_active = ram_map.is_mart_inventory_active(memory)
    if mart_maps:
        return bool(map_id is not None and map_id in mart_maps) and inventory_active
    return inventory_active


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
    """Train TinyMenuAgent as the menu specialist.

    Returns a summary compatible with train_multi_agent.combine_best_brains.
    """
    cfg = cfg or load_config(config_path)
    # Use the existing override helpers to respect CLI flags and YAML config.
    phase_state_map = cfg.get("environment", {}).get("phase_states", {})
    effective_state = state_path_override or phase_state_map.get("menu")
    cfg = _apply_overrides(
        cfg,
        headless=headless,
        total_steps_override=total_steps_override,
        state_path_override=effective_state,
    )

    training_cfg = cfg["training"]
    epsilon_cfg = training_cfg["epsilon"]

    rewards_cfg = cfg.get("rewards", {})
    menu_cfg = rewards_cfg.get("menu", {})
    economy_cfg = menu_cfg.get("economy", {})

    mart_maps: list[int] = menu_cfg.get("mart_maps", [])
    step_penalty = float(economy_cfg.get("step_penalty", -0.01))
    open_menu_bonus = float(economy_cfg.get("open_menu_bonus", 0.5))
    cursor_on_target_bonus = float(economy_cfg.get("cursor_on_target_bonus", 0.2))
    close_penalty = float(economy_cfg.get("close_penalty", -1.0))
    buy_item_scale = float(economy_cfg.get("buy_item_scale", 0.1))
    buy_item_cap = float(economy_cfg.get("buy_item_cap", 0.5))
    sell_nugget_scale = float(economy_cfg.get("sell_nugget_scale", 0.1))
    sell_nugget_cap = float(economy_cfg.get("sell_nugget_cap", 0.5))
    low_money_threshold = int(economy_cfg.get("low_money_threshold", 2000))
    low_money_penalty = float(economy_cfg.get("low_money_penalty", 0.05))

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = _seed_everything(seed)
    run_label = run_name or "menu_tiny_rl"
    print(f"--- STARTING MENU TINY-RL ON {device} ({run_label}) [seed={seed}] ---")

    checkpoint_dir, log_dir = _prepare_dirs(run_label, checkpoint_root, log_root)

    # Environment and agent
    env = PokemonRedGym(cfg["environment"])
    agent = TinyMenuAgent().to(device)

    # Optionally load existing weights for fine-tuning.
    menu_path = os.path.join(checkpoint_dir, f"menu_brain_{save_tag}.pth")
    if os.path.exists(menu_path):
        state = torch.load(menu_path, map_location=device)
        try:
            agent.load_state_dict(state)
            print(f"[INFO] Loaded existing TinyMenuAgent weights from {menu_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load existing menu weights: {exc}")

    logger = Logger(log_dir=log_dir, run_name=run_label)

    total_steps = training_cfg.get("total_steps", 50000)
    max_episode_steps = cfg.get("director", {}).get("goals", {}).get("menu", {}).get("max_steps", 128)
    save_freq = training_cfg.get("save_frequency", 2000)
    warmup = training_cfg.get("warmup_steps", 1000)
    batch_size = training_cfg.get("batch_size", 32)

    epsilon = epsilon_cfg.get("start", 0.9)
    epsilon_end = epsilon_cfg.get("end", 0.05)
    epsilon_decay = epsilon_cfg.get("decay", 50000)

    # Simple uniform replay buffer sized similarly to the menu buffer in YAML.
    replay_cfg = training_cfg.get("replay", {}).get("menu", {})
    buffer_capacity = replay_cfg.get("size", 100000)
    buffer = SimpleReplayBuffer(capacity=buffer_capacity, state_dim=agent.cfg.state_dim)

    try:
        obs, info = env.reset()
    except RuntimeError as exc:
        print(f"[WARN] Environment reset failed: {exc}")
        logger.close()
        env.close()
        return {
            "phase": "menu",
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

    # Episode bookkeeping for mart-aware training.
    start_map_id = info.get("map_id")
    start_in_mart = _mart_context(info, env.memory, mart_maps)

    # Sample an initial target index for this episode.
    last_index = info.get("menu_last_index") or 0
    if last_index < 0:
        last_index = 0
    target_index = int(np.random.randint(0, last_index + 1)) if last_index >= 0 else 0

    # Track money and selected bag counts so we can reward purchases/sales.
    last_money = ram_map.read_money(env.memory)
    last_pokeballs = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_POKE_BALL)
    last_potions = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_POTION)
    last_nuggets = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_NUGGET)

    episode_reward = 0.0
    best_episode_reward = float("-inf")
    best_menu_loss = float("inf")
    episode_steps = 0

    pbar = tqdm(range(1, total_steps + 1))

    for step in pbar:
        # Encode current RAM state for the tiny agent.
        state_vec = agent.encode_state(info, target_index)
        action_idx = agent.act(state_vec, epsilon=min(epsilon, 0.2), device=device)

        # Map abstract action to env action index.
        # We mirror the mapping from HierarchicalAgent: 0=UP,1=DOWN,2=CONFIRM/OPEN,3=CANCEL.
        from pyboy.utils import WindowEvent  # local import to avoid cycles

        menu_active_now = _menu_active(info)
        if menu_active_now:
            abstract_to_window = {
                0: WindowEvent.PRESS_ARROW_UP,
                1: WindowEvent.PRESS_ARROW_DOWN,
                2: WindowEvent.PRESS_BUTTON_A,      # confirm within menu
                3: WindowEvent.PRESS_BUTTON_B,      # cancel/back within menu
            }
        else:
            abstract_to_window = {
                0: WindowEvent.PRESS_ARROW_UP,
                1: WindowEvent.PRESS_ARROW_DOWN,
                2: WindowEvent.PRESS_BUTTON_START,  # open menu from overworld
                3: WindowEvent.PRESS_BUTTON_B,
            }
        window_event = abstract_to_window.get(action_idx, WindowEvent.PRESS_BUTTON_A)

        # PokemonRedGym.valid_actions is ordered; find the index matching this event.
        env_action = 0
        for i, ev in enumerate(env.valid_actions):
            if ev == window_event:
                env_action = i
                break

        next_obs, _, terminated, truncated, next_info = env.step(env_action)

        # Tiny menu reward in [-1, 1].
        menu_active_next = _menu_active(next_info)
        menu_active_prev = menu_active_now
        success = False
        reward = step_penalty  # small per-step penalty

        # Money / inventory deltas for shaping.
        money = ram_map.read_money(env.memory)
        pokeballs = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_POKE_BALL)
        potions = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_POTION)
        nuggets = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_NUGGET)

        money_delta = money - last_money
        pokeball_delta = pokeballs - last_pokeballs
        potion_delta = potions - last_potions
        nugget_delta = nuggets - last_nuggets

        # Case 1: menu was closed and is now open -> reward opening.
        if not menu_active_prev and menu_active_next:
            reward = max(reward, open_menu_bonus)

        # Case 2: inside menu.
        if menu_active_next:
            current_idx = next_info.get("menu_target", 0) or 0
            # Optional shaping for having cursor on target (but not yet confirming).
            if current_idx == target_index:
                reward += cursor_on_target_bonus

            # Success when confirming on the correct index.
            if window_event == WindowEvent.PRESS_BUTTON_A and current_idx == target_index:
                reward = 1.0
                success = True

        # Case 3: menu was open and is now closed without success -> strong penalty.
        if menu_active_prev and not menu_active_next and not success:
            reward = close_penalty

        # Case 4: item economics shaping.
        # - Reward buying Poké Balls / Potions (counts increase while money decreases).
        # - Reward selling Nuggets (nugget count drops while money increases).
        # - Penalize large money drops that don't result in more useful items.
        in_mart_next = _mart_context(next_info, env.memory, mart_maps)

        if in_mart_next:
            if money_delta < 0:
                # Spent money.
                if pokeball_delta > 0 or potion_delta > 0:
                    # Reward per useful item gained.
                    useful_gain = max(pokeball_delta, 0) + max(potion_delta, 0)
                    reward += min(buy_item_cap, buy_item_scale * useful_gain)
                else:
                    # Spent money without gaining tracked useful items.
                    reward -= low_money_penalty
            elif money_delta > 0:
                # Gained money.
                if nugget_delta < 0:
                    # Likely sold Nuggets.
                    reward += min(sell_nugget_cap, sell_nugget_scale * abs(nugget_delta))

        # Discourage depleting funds globally: soft penalty when below a threshold.
        if money < low_money_threshold:
            reward -= low_money_penalty

        # Update last tracked values.
        last_money = money
        last_pokeballs = pokeballs
        last_potions = potions
        last_nuggets = nuggets

        # Clip for safety (should already be in [-1,1], but shaping may push it).
        reward = float(np.clip(reward, -1.0, 1.0))

        episode_steps += 1
        episode_reward += reward

        # Treat leaving the starting map as terminal for this training run
        # when we started in a mart (or when mart_maps is not configured), so
        # the agent does not wander the overworld indefinitely.
        left_start_map = start_in_mart and next_info.get("map_id") is not None and next_info.get("map_id") != start_map_id

        done = terminated or truncated or success or left_start_map or (episode_steps >= max_episode_steps)

        # Next state encoding for buffer.
        next_state_vec = agent.encode_state(next_info, target_index)
        buffer.add(state_vec, action_idx, reward, next_state_vec, done)

        # Training step after warmup.
        metrics: Dict[str, float] = {}
        if step > warmup and len(buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size, device=device)
            loss, stats = agent.train_step(states_b, actions_b, rewards_b, next_states_b, dones_b)
            best_menu_loss = min(best_menu_loss, loss.item())
            metrics.update(stats)

        # Epsilon decay
        if epsilon > epsilon_end:
            epsilon -= (epsilon_cfg["start"] - epsilon_end) / max(1, epsilon_decay)

        # Logging
        metrics["policy/epsilon"] = epsilon
        metrics["menu_tiny/episode_reward"] = episode_reward
        # Log map id so we can inspect mart behavior in TensorBoard.
        if next_info.get("map_id") is not None:
            metrics["env/map_id"] = float(next_info.get("map_id"))
        if metrics:
            logger.log_step(metrics, step)

        # Progress bar description
        pbar.set_description(
            f"[menu_tiny:{run_label}] Rew: {episode_reward:.2f} | Eps: {epsilon:.2f}"
        )

        obs, info = next_obs, next_info

        if done:
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                # Save best-performing menu specialist.
                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "menu_brain_best_reward.pth"))

            # Reset episode
            episode_reward = 0.0
            episode_steps = 0
            try:
                obs, info = env.reset()
            except RuntimeError as exc:
                print(f"[WARN] Environment reset failed: {exc}")
                break

            # Sample a new target index for the next episode.
            last_index = info.get("menu_last_index") or 0
            if last_index < 0:
                last_index = 0
            target_index = int(np.random.randint(0, last_index + 1)) if last_index >= 0 else 0

            # Reset money/item baselines and starting map for the new episode.
            start_map_id = info.get("map_id")
            last_money = ram_map.read_money(env.memory)
            last_pokeballs = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_POKE_BALL)
            last_potions = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_POTION)
            last_nuggets = ram_map.count_item_in_bag(env.memory, ram_map.ITEM_ID_NUGGET)

            # Console logging to inspect map ids used during menu training.
            start_in_mart = _mart_context(info, env.memory, mart_maps)
            print(
                f"[MENU_TINY] Episode reset: map_id={start_map_id}, "
                f"start_in_mart={start_in_mart}, target_index={target_index}"
            )

        if step % save_freq == 0:
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"menu_brain_{save_tag}.pth"))

    # Final save
    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"menu_brain_{save_tag}.pth"))
    logger.close()
    env.close()

    if best_episode_reward == float("-inf"):
        best_episode_reward = None
    best_menu_loss = None if best_menu_loss == float("inf") else best_menu_loss

    return {
        "phase": "menu",
        "run_name": run_label,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": logger.log_dir,
        "best_episode_reward": best_episode_reward,
        "best_nav_loss": None,
        "best_battle_loss": None,
        "best_menu_loss": best_menu_loss,
        "seed": seed,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the menu specialist via Tiny RL.")
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
        help="Override environment state path for menu training (e.g., states/menu).",
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


if __name__ == "__main__":
    main()
