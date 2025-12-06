import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.phased_training import build_phase_arg_parser, train_phase

PHASE = "menu"


def train(
    cfg=None,
    config_path=None,
    run_name=None,
    checkpoint_root="checkpoints",
    log_root="experiments/logs",
    save_tag="latest",
    total_steps_override=None,
    headless=None,
    state_path_override=None,
    device_override=None,
    seed=None,
):
    """Shim to keep the interface identical to train_end_to_end.train."""
    return train_phase(
        PHASE,
        cfg=cfg,
        config_path=config_path,
        run_name=run_name,
        checkpoint_root=checkpoint_root,
        log_root=log_root,
        save_tag=save_tag,
        total_steps_override=total_steps_override,
        headless=headless,
        state_path_override=state_path_override,
        device_override=device_override,
        seed=seed,
    )


def main() -> None:
    parser = build_phase_arg_parser(PHASE)
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
