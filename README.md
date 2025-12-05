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

## Next Steps
1. Implement the VQ-VAE / ResNet encoder and collect an unsupervised dataset using PyBoy screenshots.
2. Bring up the custom Gym wrapper so that the experiments can step the emulator headlessly.
3. Train the Director and Specialists following the provided scripts, ensuring all reward queries go through the RAM map.
