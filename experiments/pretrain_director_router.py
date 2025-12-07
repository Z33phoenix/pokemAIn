import argparse
import os
import random
import sys
from typing import Any, Dict, List, Tuple
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.director import Director  # noqa: E402
from src.env.pokemon_red_gym import PokemonRedGym  # noqa: E402
from src.env.rewards import RewardSystem  # noqa: E402
from experiments.train_end_to_end import load_config, _apply_overrides, _seed_everything  # noqa: E402


def _menu_active(info: Dict[str, Any]) -> bool:
    """Return True when RAM reports an interactive menu with selectable options."""
    menu_open = info.get("menu_open")
    has_options = info.get("menu_has_options")
    if menu_open is None and has_options is None:
        return False
    if menu_open is None:
        return bool(has_options)
    if has_options is None:
        return bool(menu_open)
    return bool(menu_open and has_options)


def _label_from_info(info: Dict[str, Any]) -> int:
    """Map environment info into a specialist label (nav=0, battle=1, menu=2)."""
    if info.get("battle_active", False):
        return 1  # battle specialist
    if _menu_active(info):
        return 2  # menu specialist
    return 0  # navigation specialist


def _collect_state_files(paths: List[str]) -> List[str]:
    """Expand directories and files into a deduped list of .state paths."""
    seen = []
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if os.path.isfile(full_path) and entry.lower().endswith(".state"):
                    seen.append(full_path)
        elif os.path.isfile(path) and path.lower().endswith(".state"):
            seen.append(path)
    deduped = []
    for p in seen:
        if p not in deduped:
            deduped.append(p)
    return deduped


def collect_dataset(
    env,
    steps: int,
    state_files: List[str],
    steps_per_state: int,
    max_collect_seconds: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run random policy to gather labeled frames. If state_files provided, load each
    state and collect steps_per_state frames from it; otherwise fallback to a single
    long rollout of length `steps`.
    """
    obs_list: List[np.ndarray] = []
    label_list: List[int] = []

    if state_files:
        pbar = tqdm(total=steps, desc="Collecting from states")
        idx = 0
        while len(obs_list) < steps:
            path = state_files[idx % len(state_files)]
            idx += 1
            try:
                with open(path, "rb") as f:
                    data = f.read()
                env.pyboy.load_state(io.BytesIO(data))
                env.step_count = 0
                obs = env._get_obs()
                info = env._get_info()
            except Exception:
                continue
            start_time = time.monotonic()
            max_steps = random.randint(1, max(1, steps_per_state))
            for _ in range(max_steps):
                if len(obs_list) >= steps:
                    break
                if time.monotonic() - start_time > max_collect_seconds:
                    break
                label = _label_from_info(info)
                obs_list.append(obs.astype(np.float32) / 255.0)
                label_list.append(label)
                pbar.update(1)
                action = env.action_space.sample()
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    # Reset to the same state so we can get additional offsets
                    try:
                        env.pyboy.load_state(io.BytesIO(data))
                        env.step_count = 0
                        obs = env._get_obs()
                        info = env._get_info()
                    except Exception:
                        break
        pbar.close()
    else:
        obs, info = env.reset()
        pbar = tqdm(range(steps), desc="Collecting rollout")
        for _ in pbar:
            label = _label_from_info(info)
            obs_list.append(obs.astype(np.float32) / 255.0)
            label_list.append(label)
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    obs_tensor = torch.tensor(np.stack(obs_list, axis=0))
    labels = torch.tensor(label_list, dtype=torch.long)
    return obs_tensor, labels


def train_router(
    director: Director,
    dataset: TensorDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr_encoder: float,
    lr_router: float,
) -> Dict[str, float]:
    """Train the director encoder+router on the labeled dataset."""
    director.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    params = [
        {"params": director.encoder.parameters(), "lr": lr_encoder},
        {"params": director.router.parameters(), "lr": lr_router},
    ]
    optim_all = optim.AdamW(params)
    loss_fn = nn.CrossEntropyLoss()
    director.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            feats = director.encoder(xb)
            logits = director.router(feats)
            loss = loss_fn(logits, yb)
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    # simple eval
    with torch.no_grad():
        xb = dataset.tensors[0].to(device)
        yb = dataset.tensors[1].to(device)
        feats = director.encoder(xb)
        logits = director.router(feats)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == yb).float().mean().item()
    return {"acc": acc}


def main():
    """CLI to pretrain the director router using heuristic labels from saved states."""
    parser = argparse.ArgumentParser(description="Pretrain Director router with heuristic labels.")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameter YAML.")
    parser.add_argument("--run-name", type=str, default="router_pretrain", help="Run name for seeding/logging.")
    parser.add_argument("--steps", type=int, default=20000, help="Target labeled frames to collect.")
    parser.add_argument("--steps-per-state", type=int, default=500, help="Maximum steps to collect per .state file when provided.")
    parser.add_argument("--max-collect-seconds", type=float, default=5.0, help="Wall-clock cap per state during collection.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr-encoder", type=float, default=1e-4, help="Encoder learning rate.")
    parser.add_argument("--lr-router", type=float, default=3e-4, help="Router head learning rate.")
    parser.add_argument("--save-path", type=str, default="checkpoints/director_router_pretrained.pth", help="Where to save the director weights.")
    parser.add_argument("--headless", dest="headless", action="store_true", help="Force headless env.")
    parser.add_argument("--windowed", dest="headless", action="store_false", help="Force windowed env.")
    parser.set_defaults(headless=None)
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g., cpu or cuda:0.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, headless=args.headless, total_steps_override=None, state_path_override=None)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _seed_everything(None)

    env = PokemonRedGym(cfg["environment"])
    reward_sys = RewardSystem(cfg["rewards"])
    director = Director(cfg["director"])

    phase_states = cfg.get("environment", {}).get("phase_states", {})
    state_paths = _collect_state_files(list(phase_states.values()))
    if not state_paths:
        print(f"[INFO] Collecting {args.steps} labeled frames (no state files found)...")
    else:
        print(f"[INFO] Collecting labeled frames from {len(state_paths)} state files, {args.steps_per_state} steps each...")
    obs_tensor, labels = collect_dataset(
        env,
        steps=args.steps,
        state_files=state_paths,
        steps_per_state=args.steps_per_state,
        max_collect_seconds=args.max_collect_seconds,
    )
    dataset = TensorDataset(obs_tensor, labels)
    print(f"[INFO] Dataset size: {len(dataset)}")

    print("[INFO] Training router head...")
    stats = train_router(
        director,
        dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_router=args.lr_router,
    )
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(director.state_dict(), args.save_path)
    print(f"[INFO] Saved pretrained director to {args.save_path}")
    print(f"[INFO] Router accuracy on collected set: {stats['acc']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
