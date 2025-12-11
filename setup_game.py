import argparse
import json
import os
import sys
import time
import traceback

import pyboy
import yaml



def load_config(config_path: str = "config/hyperparameters.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_pyboy(
    rom_path: str,
    resume_from: str | None = None,
    emulation_speed: int = 1,
    window_backend: str = "SDL2",
) -> pyboy.PyBoy:
    """
    Set up PyBoy emulator with CGB mode enabled.
    """
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM not found at {rom_path}")

    if resume_from and not os.path.exists(resume_from):
        print(f"[WARN] Resume state '{resume_from}' does not exist; starting from ROM boot.")
        resume_from = None

    pb = pyboy.PyBoy(rom_path, window=window_backend, cgb=True)
    pb.set_emulation_speed(emulation_speed)

    if resume_from:
        with open(resume_from, "rb") as f:
            pb.load_state(f)
        print(f"[INFO] Loaded starting state from: {resume_from}")

    return pb


def capture_state(
    rom_path: str,
    state_path: str,
    resume_from: str | None = None,
    emulation_speed: int = 1,
    window_backend: str = "SDL2",
) -> str:
    """
    Open a PyBoy window, allow the user to play to the desired scenario, and
    persist the resulting emulator state to disk.
    """
    pb = setup_pyboy(rom_path, resume_from, emulation_speed, window_backend)

    print("\n[INFO] Play until you've reached the desired state.")
    print("      Close the window (X) when you want to snapshot the emulator.\n")
    try:
        while pb.tick():
            pass
    finally:
        try:
            os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
            with open(state_path, "wb") as f:
                pb.save_state(f)
            print(f"[SUCCESS] Saved state to: {state_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to save state: {exc}")
            traceback.print_exc()
        pb.stop()
    return state_path


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for interactive state capture."""
    parser = argparse.ArgumentParser(
        description="Interactively capture emulator state using config file settings."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/hyperparameters.yaml",
        help="Path to configuration file."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional starting state to resume from.",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Override state path from config.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBoy headless.",
    )
    return parser.parse_args()




def main() -> None:
    """Entry point for capturing emulator state using config file settings."""
    args = parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] Failed to load config: {exc}")
        sys.exit(1)
    
    rom_path = config["environment"]["rom_path"]
    state_path = args.state_path or config["environment"]["state_path"]
    emulation_speed = config["environment"]["emulation_speed"]
    
    window_backend = "headless" if args.headless or config["environment"]["headless"] else "SDL2"
    
    print(f"[INFO] Using ROM: {rom_path}")
    print(f"[INFO] Will save state to: {state_path}")
    
    if args.resume_from:
        print(f"[INFO] Starting from: {args.resume_from}")

    try:
        capture_state(
            rom_path=rom_path,
            state_path=state_path,
            resume_from=args.resume_from,
            emulation_speed=emulation_speed,
            window_backend=window_backend,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[CRITICAL ERROR] The game crashed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
