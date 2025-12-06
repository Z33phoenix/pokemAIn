# Nested Director-Specialist Pok�mon Red Agent

This project bootstraps a hierarchical reinforcement learning agent for Pok�mon Red that follows Google's Nested Learning paradigm. A convolutional **Director** consumes PyBoy-rendered pixels and routes each latent state toward the most appropriate **Specialist** head. Specialists focus on well-scoped skills (battle strategy or overworld navigation) and are trained with state-of-the-art distributional RL algorithms.

## Key Ideas
- **Pixels only**: Policies never peek into game RAM; only the reward code and logging utilities may read from 
am_map.py.
- **Vision-first**: vision/encoder.py compresses 84x84 grayscale frames into a latent space shared across the hierarchy.
- **Director-Specialist routing**: agent/director.py determines which specialist brain should act, enabling compositional policies.
- **Specialized RL heads**: battle_brain.py follows Beyond the Rainbow while nav_brain.py implements CrossQ for controllable exploration.
- **Custom Gym wrapper**: pokemon_red_gym.py wires PyBoy into Gymnasium, ensuring repeatable rollouts for Stable Baselines3 / CleanRL style loops.

## Repository Layout
- src/agent: Director network and specialist brains.
- src/env: PyBoy-based Gym, RAM metadata, and constrained reward shaping utilities.
- src/vision: Visual encoder definitions for latent discovery.
- src/utils: Experience replay and experiment logging helpers.
- experiments/phase*_*.py: Sequential training stages (vision pretraining, director routing, full-loop fine-tuning).
- config/hyperparameters.yaml: Centralized training knobs for Director, Specialists, and environment wrappers.

## Phased Specialist Training
- Capture focused save states into `states/`, organized per phase (e.g., `states/nav/`, `states/battle/`, `states/menu/`). Each folder can hold multiple `.state` files (different towns/routes, different trainer battles/party comps, different menu contexts or no-menu states to train opening START). Use `python setup_game.py --phase nav --state-path states/nav/nav_route1.state` (repeat as needed; add `--resume-from` to branch from an existing state).
- Pretrain each specialist with `python experiments/train_nav_phase.py --state-path states/nav`, `python experiments/train_battle_phase.py --state-path states/battle`, and `python experiments/train_menu_phase.py --state-path states/menu`. Each episode will randomly pick one of the available states in the specified directory.
- Scale out any phase with the multi-agent launcher: `python experiments/train_multi_agent.py --phase nav --state-path states/nav --num-agents 4 --combine-best`.
- After specialists are prepared, run the full loop with `experiments/train_end_to_end.py` (or `train_multi_agent.py --phase full`) using `states/initial.state` for director + integration training.

## Next Steps
1. Implement the VQ-VAE / ResNet encoder and collect an unsupervised dataset using PyBoy screenshots.
2. Bring up the custom Gym wrapper so that the experiments can step the emulator headlessly.
3. Train the Director and Specialists following the provided scripts, ensuring all reward queries go through the RAM map.
