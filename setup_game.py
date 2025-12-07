import argparse
import json
import os
import sys
import time
import traceback

import pyboy

# Default state destinations so we never clobber initial.state
DEFAULT_STATE_PATHS = {
    # Phase folders allow multiple distinct starts (routes/battles/menus)
    "nav": os.path.join("states", "nav"),
    "battle": os.path.join("states", "battle"),
    "menu": os.path.join("states", "menu"),
    "director": os.path.join("states", "initial.state"),
    "full": os.path.join("states", "initial.state"),
}


def _write_metadata(state_path: str, phase: str | None, notes: str | None) -> str | None:
    """Optionally drop a sidecar JSON file to document how the state was captured."""
    if not notes and not phase:
        return None
    meta = {
        "phase": phase,
        "notes": notes,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_path": os.path.abspath(state_path),
    }
    meta_path = f"{state_path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def capture_state(
    rom_path: str,
    resume_from: str | None = None,
    emulation_speed: int = 1,
    window_backend: str = "SDL2",
    notes: str | None = None,
    default_phase: str | None = None,
) -> tuple[str, str]:
    """
    Open a PyBoy window, allow the user to play to the desired scenario, and
    persist the resulting emulator state to disk. Phase selection happens at
    save-time so a single playthrough can produce states for multiple phases.
    """
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM not found at {rom_path}")

    if resume_from and not os.path.exists(resume_from):
        print(f"[WARN] Resume state '{resume_from}' does not exist; starting from ROM boot.")
        resume_from = None

    pb = pyboy.PyBoy(rom_path, window=window_backend)
    pb.set_emulation_speed(emulation_speed)

    if resume_from:
        with open(resume_from, "rb") as f:
            pb.load_state(f)
        print(f"[INFO] Loaded starting state from: {resume_from}")

    print("\n[INFO] Play until you've reached the desired route/menu/battle.")
    print("      Close the window (X) when you want to snapshot the emulator.\n")
    try:
        while pb.tick():
            pass
    finally:
        try:
            phase = _prompt_phase(default=default_phase, show_options_always=True)
            output_path = _auto_phase_path(phase)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                pb.save_state(f)
            print(f"[SUCCESS] Saved state to: {output_path}")
            meta_path = _write_metadata(output_path, phase, notes)
            if meta_path:
                print(f"[INFO] Wrote metadata to: {meta_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to save state: {exc}")
            traceback.print_exc()
        pb.stop()
    return output_path, phase


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for interactive state capture."""
    parser = argparse.ArgumentParser(
        description="Interactively capture multiple emulator states for phased training."
    )
    parser.add_argument("--rom-path", type=str, default="pokemon_red.gb", help="Path to the Pokemon Red ROM.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional starting state (e.g., states/initial.state or a prior capture) to branch new scenarios.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Short freeform note describing the route/battle/menu you are capturing.",
    )
    parser.add_argument(
        "--emulation-speed",
        type=int,
        default=1,
        help="Speed multiplier for PyBoy while you play through the scenario.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBoy headless; useful for quickly advancing to scripted points.",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Capture a single state then exit (default is to offer another prompt).",
    )
    return parser.parse_args()


def _auto_name(phase: str | None) -> str:
    """Generate a timestamped filename for new state captures."""
    phase_prefix = (phase or "state").lower()
    return f"{phase_prefix}_{time.strftime('%Y%m%d-%H%M%S')}.state"


def _auto_phase_path(phase: str) -> str:
    """Return an auto-named .state path under the default directory for a phase."""
    base = DEFAULT_STATE_PATHS.get(phase, os.path.join("states", "initial.state"))
    if base.endswith(".state"):
        return base
    return os.path.join(base, _auto_name(phase))


def _prompt_phase(default: str | None = None, show_options_always: bool = False) -> str:
    """Ask the user which phase this capture belongs to (always shows all options)."""
    options = ["nav", "battle", "menu", "full", "director"]
    prompt_default = default if default in options else None
    prompt = f"Select phase [{'/'.join(options)}]"
    if prompt_default:
        prompt += f" (default: {prompt_default})"
    prompt += ": "
    while True:
        choice = input(prompt).strip().lower()
        if not choice and prompt_default:
            return prompt_default
        if choice in options:
            return choice
        print(f"Please enter one of: {', '.join(options)}")


def _resolve_resume(user_resume: str | None, output_path: str | None = None) -> str | None:
    """
    Pick a resume state when the user opts not to start from the last capture.
    Always prefer states/initial.state, then legacy initial.state.
    """
    if user_resume:
        return user_resume
    preferred_initial = os.path.join("states", "initial.state")
    if os.path.exists(preferred_initial) and output_path != preferred_initial:
        return preferred_initial
    if os.path.exists("initial.state") and output_path != "initial.state":
        return "initial.state"
    return None


def main() -> None:
    """Entry point for capturing one or more emulator states via PyBoy."""
    args = parse_args()
    rom_path = args.rom_path
    # Default resume state: prefer an existing states/initial.state so phases never overwrite it.
    resume_from = _resolve_resume(args.resume_from, "")

    window_backend = "headless" if args.headless else "SDL2"

    if resume_from:
        print(f"[INFO] Starting from: {resume_from}")

    try:
        last_saved, last_phase = capture_state(
            rom_path=rom_path,
            resume_from=resume_from,
            emulation_speed=args.emulation_speed,
            window_backend=window_backend,
            notes=args.notes,
            default_phase=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[CRITICAL ERROR] The game crashed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    if args.one_shot:
        return

    while True:
        choice = input("\nCapture another state? [y/N]: ").strip().lower()
        if choice not in {"y", "yes"}:
            break

        resume_choice = input("Start from last saved state? [Y/n]: ").strip().lower()
        if resume_choice in {"n", "no"}:
            next_resume = _resolve_resume(None, "")
        else:
            next_resume = last_saved
        next_notes = input("Optional notes for this capture (blank to reuse): ").strip()
        if not next_notes:
            next_notes = args.notes

        last_saved, last_phase = capture_state(
            rom_path=rom_path,
            resume_from=next_resume,
            emulation_speed=args.emulation_speed,
            window_backend=window_backend,
            notes=next_notes,
            default_phase=last_phase,
        )


if __name__ == "__main__":
    main()
