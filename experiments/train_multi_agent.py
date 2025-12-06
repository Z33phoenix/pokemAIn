import argparse
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import Process, Queue, set_start_method
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import psutil
except ImportError:
    psutil = None

import train_end_to_end


def _resolve_trainer(phase: str):
    """Select the appropriate training entry point for the requested phase."""
    phase = (phase or "full").lower()
    if phase == "nav":
        import train_nav_phase as trainer
    elif phase == "battle":
        import train_battle_phase as trainer
    elif phase == "menu":
        import train_menu_phase as trainer
    else:
        trainer = train_end_to_end
    return trainer


def _worker(agent_idx: int, args: argparse.Namespace, result_queue: Queue) -> None:
    """Runs a single agent training job in its own process."""
    # Keep each worker single-threaded to avoid intra-process contention.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if args.pin_cores:
        _pin_worker_to_core(agent_idx)

    # Advertise agent id for window title / logging in the environment.
    os.environ.setdefault("POKEMAIN_AGENT_ID", str(agent_idx))

    run_name = f"{args.phase}_agent{agent_idx}" if args.phase else f"agent{agent_idx}"
    checkpoint_root = os.path.join(args.checkpoint_root, args.run_prefix)
    log_root = os.path.join(args.log_root, args.run_prefix)
    trainer = _resolve_trainer(args.phase)
    try:
        summary = trainer.train(
            config_path=args.config,
            run_name=run_name,
            checkpoint_root=checkpoint_root,
            log_root=log_root,
            save_tag=args.save_tag,
            total_steps_override=args.total_steps,
            headless=args.headless,
            state_path_override=args.state_path,
            device_override=args.device,
            seed=(args.seed + agent_idx) if args.seed is not None else None,
        )
        summary["worker_index"] = agent_idx
        result_queue.put(summary)
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"worker_index": agent_idx, "error": str(exc)})


def _copy_if_exists(src: str, dst: str) -> bool:
    if os.path.exists(src):
        shutil.copy(src, dst)
        return True
    return False


def _pin_worker_to_core(agent_idx: int) -> Optional[int]:
    """Try to bind this worker to a single logical core."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process()
        cores = psutil.cpu_count(logical=True) or 1
        target_core = agent_idx % cores
        proc.cpu_affinity([target_core])
        print(f"[worker {agent_idx}] pinned to core {target_core}")
        return target_core
    except Exception as exc:  # noqa: BLE001
        print(f"[worker {agent_idx}] affinity not set: {exc}")
        return None


def combine_best_brains(summaries: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """Pick the best specialist from each run and assemble a combined checkpoint directory."""
    os.makedirs(output_dir, exist_ok=True)

    def _best_by(key: str, better=min):
        candidates = [
            s
            for s in summaries
            if key in s and s.get(key) is not None and isinstance(s.get(key), (int, float)) and math.isfinite(s[key])
        ]
        if not candidates:
            return None
        return better(candidates, key=lambda s: s[key])

    best_nav = _best_by("best_nav_loss", better=min)
    best_battle = _best_by("best_battle_loss", better=min)
    best_menu = _best_by("best_menu_loss", better=min)
    best_overall = _best_by("best_episode_reward", better=max)

    selection = {}

    if best_nav:
        src = os.path.join(best_nav["checkpoint_dir"], "nav_brain_best_nav.pth")
        if _copy_if_exists(src, os.path.join(output_dir, "nav_brain_latest.pth")):
            selection["nav"] = {"from_run": best_nav["run_name"], "score": best_nav["best_nav_loss"], "path": src}

    if best_battle:
        src = os.path.join(best_battle["checkpoint_dir"], "battle_brain_best_battle.pth")
        if _copy_if_exists(src, os.path.join(output_dir, "battle_brain_latest.pth")):
            selection["battle"] = {
                "from_run": best_battle["run_name"],
                "score": best_battle["best_battle_loss"],
                "path": src,
            }

    if best_menu:
        src = os.path.join(best_menu["checkpoint_dir"], "menu_brain_best_menu.pth")
        if _copy_if_exists(src, os.path.join(output_dir, "menu_brain_latest.pth")):
            selection["menu"] = {"from_run": best_menu["run_name"], "score": best_menu["best_menu_loss"], "path": src}

    if best_overall:
        src = os.path.join(best_overall["checkpoint_dir"], "director_best_reward.pth")
        if _copy_if_exists(src, os.path.join(output_dir, "director_latest.pth")):
            selection["director"] = {
                "from_run": best_overall["run_name"],
                "score": best_overall["best_episode_reward"],
                "path": src,
            }
        # Also pull the overall best full agent for convenience.
        full_src = os.path.join(best_overall["checkpoint_dir"], "nav_brain_best_reward.pth")
        _copy_if_exists(full_src, os.path.join(output_dir, "nav_brain_best_reward.pth"))
        full_src = os.path.join(best_overall["checkpoint_dir"], "battle_brain_best_reward.pth")
        _copy_if_exists(full_src, os.path.join(output_dir, "battle_brain_best_reward.pth"))
        full_src = os.path.join(best_overall["checkpoint_dir"], "menu_brain_best_reward.pth")
        _copy_if_exists(full_src, os.path.join(output_dir, "menu_brain_best_reward.pth"))

    with open(os.path.join(output_dir, "combined_selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch multiple agents in parallel and combine the best brains.")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameters YAML.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of parallel agents to launch.")
    parser.add_argument("--run-prefix", type=str, default=None, help="Optional prefix for this multi-agent batch.")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints", help="Base directory for checkpoints.")
    parser.add_argument("--log-root", type=str, default="experiments/logs", help="Base directory for TensorBoard logs.")
    parser.add_argument("--save-tag", type=str, default="latest", help="Tag used when saving checkpoints.")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["full", "nav", "battle", "menu"],
        default="full",
        help="Which training routine to launch for each worker.",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Override environment state path for all workers (use nav/battle/menu states).",
    )
    parser.add_argument("--total-steps", type=int, default=None, help="Override training steps from config.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override for all workers.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed; workers offset by index.")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=False,
        help="Force headless PyBoy for all workers (default).",
    )
    parser.add_argument(
        "--windowed",
        dest="headless",
        action="store_false",
        help="Allow PyBoy windows (not recommended for >1 worker).",
    )
    parser.add_argument(
        "--combine-best",
        action="store_true",
        help="After all workers finish, assemble a directory containing the best specialist from each run.",
    )
    parser.add_argument(
        "--pin-cores",
        dest="pin_cores",
        action="store_true",
        default=True,
        help="Pin each worker to a logical core (requires psutil).",
    )
    parser.add_argument(
        "--no-pin-cores",
        dest="pin_cores",
        action="store_false",
        help="Disable core pinning.",
    )
    args = parser.parse_args()

    if args.pin_cores and psutil is None:
        print("[WARN] psutil not installed; core pinning is disabled.")
        args.pin_cores = False

    if args.run_prefix is None:
        phase_label = args.phase or "multi"
        args.run_prefix = time.strftime(f"{phase_label}_%Y%m%d-%H%M%S")

    set_start_method("spawn", force=True)

    result_queue: Queue = Queue()
    processes: List[Process] = []

    for i in range(args.num_agents):
        proc = Process(target=_worker, args=(i, args, result_queue))
        proc.start()
        processes.append(proc)

    summaries: List[Dict[str, Any]] = []
    for _ in range(args.num_agents):
        summaries.append(result_queue.get())

    for proc in processes:
        proc.join()

    checkpoint_root = os.path.join(args.checkpoint_root, args.run_prefix)
    os.makedirs(checkpoint_root, exist_ok=True)
    summary_path = os.path.join(checkpoint_root, "run_summaries.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[INFO] Wrote run summaries to {summary_path}")

    errors = [s for s in summaries if "error" in s]
    if errors:
        print("[WARN] One or more workers failed:")
        for err in errors:
            print(f"  worker {err.get('worker_index')}: {err.get('error')}")

    if args.combine_best and not errors:
        combined_dir = os.path.join(checkpoint_root, "combined_best")
        selection = combine_best_brains(summaries, combined_dir)
        print(f"[INFO] Combined best brains saved to {combined_dir}: {selection}")


if __name__ == "__main__":
    main()
