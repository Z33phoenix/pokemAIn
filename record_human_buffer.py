import argparse

from train import train


def main():
    parser = argparse.ArgumentParser(
        description="Record a fixed number of human demonstration steps."
    )
    parser.add_argument("--output", type=str, default="saves/human_buffer.pt",
                        help="Path to save the recorded human buffer (torch file).")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of steps to record before stopping.")
    parser.add_argument("--config", type=str, help="Custom hyperparameters.yaml path.")
    parser.add_argument("--strategy", type=str, default="reactive",
                        choices=["llm", "heuristic", "reactive", "hybrid"],
                        help="Goal-setting strategy to run while recording.")
    parser.add_argument("--headless", action="store_true", help="Run emulator headless.")
    parser.add_argument("--windowed", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--state-path", type=str, help="Optional savestate to load.")
    parser.add_argument("--run-name", type=str, help="Optional run name for logging.")
    parser.add_argument("--checkpoint-root", type=str, default="experiments",
                        help="Directory for checkpoints/logs.")
    parser.add_argument("--save-tag", type=str, default="latest", help="Checkpoint tag.")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility.")
    parser.add_argument("--device", type=str, help="Force device (cpu/cuda).")
    parser.add_argument("--speed", type=int, default=2,
                        help="Emulation speed multiplier while recording (default: 2).")

    args = parser.parse_args()

    train(
        config_path=args.config,
        brain_type="human",
        strategy_preset=args.strategy,
        memory_preset="low",
        run_name=args.run_name,
        checkpoint_root=args.checkpoint_root,
        save_tag=args.save_tag,
        total_steps_override=None,
        headless=args.headless,
        state_path_override=args.state_path,
        device_override=args.device,
        seed=args.seed,
        record_human_buffer_path=args.output,
        human_buffer_steps=args.steps,
        emulation_speed_override=args.speed,
    )


if __name__ == "__main__":
    main()
