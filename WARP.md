# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Environment & installation

- Recommended Python: 3.10+.
- ROM: place `pokemon_red.gb` in the repo root.
- Preferred environment: a Conda env named `pokemain` that already contains the required packages.
  - Activate it before running any commands:
    - `conda activate pokemain`
- If you need to (re)create the environment from scratch (Windows / PowerShell example):
  - `conda create -n pokemain python=3.10`
  - `conda activate pokemain`
  - `pip install -r requirements.txt`

Everything is plain Python; there is no separate build step.

## Core commands & workflows

All commands are intended to be run from the repo root.

### Capture save states for training

Phased training depends on curated `.state` files under `states/`.

- Interactive capture tool:
  - `python setup_game.py`
- Typical workflow:
  1. Run `python setup_game.py`.
  2. Play to the scenario you want (overworld navigation, battle, menu, or initial-game state).
  3. Close the PyBoy window; you will be prompted to label the phase (nav/battle/menu/full).
  4. States are written under `states/<phase>/...` and are consumed by training scripts.

### Single-agent end‑to‑end training (director + all specialists)

- Default config (`config/hyperparameters.yaml`) and environment state:
  - `python experiments/train_end_to_end.py --run-name full001`
- Override initial state:
  - `python experiments/train_end_to_end.py --state-path states/initial.state --run-name full001`
- Useful flags (from `experiments/train_end_to_end.py`):
  - `--config PATH` — alternate YAML config.
  - `--headless` / `--windowed` — force headless or SDL2 window regardless of YAML.
  - `--total-steps N` — override `training.total_steps`.
  - `--device cpu|cuda:0` — explicit torch device.

Checkpoints are written under `checkpoints/<run_name>/`, logs under `experiments/logs/<run_name>/`.

### Phased specialist pretraining (single head at a time)

These use the shared `HierarchicalAgent` but restrict training to one specialist head, keeping the director encoder synchronized. They expect phase-specific `.state` files under `states/nav`, `states/battle`, `states/menu` (configured in `config/hyperparameters.yaml` under `environment.phase_states`).

Navigation specialist:
- `python experiments/train_nav_phase.py --state-path states/nav --run-name nav001`

Battle specialist:
- `python experiments/train_battle_phase.py --state-path states/battle --run-name bat001`

Menu specialist:
- `python experiments/train_menu_phase.py --state-path states/menu --run-name menu001`

Common options (all three wrappers share `build_phase_arg_parser`):
- `--config PATH` — override config.
- `--headless` / `--windowed` — control PyBoy window.
- `--total-steps N` — override training steps for this phase.
- `--state-path PATH` — override the phase directory/file from YAML.
- `--device` / `--seed` — device and RNG seed overrides.

### Multi‑agent orchestration & model selection

`experiments/train_multi_agent.py` launches multiple workers (processes) and can optionally assemble a "best of" checkpoint directory.

Examples:

- Multi-agent full training (default phase `full`):
  - `python experiments/train_multi_agent.py --num-agents 4 --combine-best`
- Multi-agent navigation only:
  - `python experiments/train_multi_agent.py --phase nav --state-path states/nav --num-agents 4 --combine-best`

Key flags:
- `--config PATH` — config file for all workers.
- `--phase {full,nav,battle,menu}` — which training entrypoint to use.
- `--num-agents N` — number of parallel workers.
- `--run-prefix PREFIX` — prefix for output directories; if omitted, a timestamped prefix is generated.
- `--checkpoint-root`, `--log-root` — base directories for outputs.
- `--state-path` — shared override for all workers.
- `--device` — device override (e.g. `cuda:0`, `cpu`).
- `--headless` / `--windowed` — PyBoy display mode.
- `--combine-best` — after all workers finish, copies the best specialist weights into `<checkpoint-root>/<run-prefix>/combined_best/`.

Per-worker summaries are stored in `<checkpoint-root>/<run-prefix>/run_summaries.json`.

### Visualizing training with TensorBoard

Logs are written under `experiments/logs/` by `src/utils/logger.Logger`.

- Start TensorBoard:
  - `tensorboard --logdir experiments/logs`

Each run is logged into a subdirectory combining the run name and a timestamp.

### Watch the environment / observation pipeline

`watch_agent.py` runs the `PokemonRedGym` environment with a random policy and shows the downsampled observation that the model sees.

- Real-time viewer (SDL2 window + OpenCV window):
  - `python watch_agent.py`

This uses `headless=False, emulation_speed=1` so you see the game at real-time speed.

### Tests and linting

As of this WARP.md, there are no test files or lint/type-check configurations in the repository. There is no canonical "run tests" or "lint" command; if you add tests later, prefer exposing them via a top-level script or documented command so they can be referenced here.

## High‑level architecture

### Configuration as the central contract

- `config/hyperparameters.yaml` is the primary configuration surface for the system. It defines:
  - `environment`: ROM path, initial state(s), headless vs windowed, emulation speed, action repeat, and episode length.
  - `training`: global hyperparameters (total steps, batch size, replay sizes per phase, epsilon schedule, save/log frequencies).
  - `rewards`: global shaping weights plus a nested `menu` block for menu-specific reward shaping.
  - `director`: router topology, goal head types, goal routing/bias, graph-memory size, and per-goal targets (explore/train/survive/menu).
  - `specialists`: per-specialist learning rates, discount factors, allowed action IDs, and menu_goal encodings.

Most Python modules consume this config directly or via wrapper helpers (`load_config`, `_apply_overrides`), so changes here propagate cleanly through training code.

### Environment & RAM abstraction layer

- `src/env/pokemon_red_gym.py` is a thin Gymnasium `Env` around PyBoy:
  - Constructs `pyboy.PyBoy` with either `headless` or `SDL2` backend and controls emulation speed.
  - Defines a discrete action space over `[DOWN, LEFT, RIGHT, UP, A, B, START, SELECT]` and mirrors them to press/release `WindowEvent`s with configurable `action_repeat` and `release_frame`.
  - Manages `.state` files:
    - On initialization, it scans configured paths (`environment.state_path` and fallbacks) and preloads all `.state` files into memory buffers.
    - On `reset`, it samples from these buffers using an internal queue to avoid immediate repeats.
  - Produces observations as `(1, 84, 84)` grayscale frames (cropped & resized in `_get_obs`).
  - Builds a compact `info` dict in `_get_info` from RAM (via `ram_map`) including:
    - Overworld map and coordinates.
    - Player and enemy HP (absolute and normalized).
    - Party size.
    - Menu state (open flag, cursor position, active item index, depth, and whether options are available).

- `src/env/ram_map.py` is the single source of truth for RAM offsets and related helpers:
  - Encapsulates raw addresses for player/enemy HP, overworld coordinates, map ID, battle status, menu cursor/item metadata, and experience.
  - Exposes small helpers (`read_player_hp`, `read_player_position`, `read_map_id`, `read_menu_cursor`, `is_menu_open`, etc.) used by both `PokemonRedGym` and the reward system.
  - Higher-level code never touches magic RAM addresses directly; it goes through these helpers.

### Reward shaping system

- `src/env/rewards.py` defines `RewardSystem`, which interprets `info` + raw RAM reads into shaped rewards for each subsystem:
  - Navigation shaping:
    - Tracks visited maps and tile coordinates to reward exploring new tiles/maps (`new_tile`, `new_map`).
    - Detects stalling or repeatedly bumping into walls to add negative shaping (`wall_bump`, `stale_penalty`).
  - Battle shaping:
    - Uses HP fractions and `is_battle_active` to detect battle transitions and outcome (win vs loss threshold), plus incremental damage dealt/taken.
    - Rewards finishing battles with high HP and penalizes large post-battle HP drops.
  - Menu shaping:
    - Uses menu-open flags, cursor movement, and goal context (`goal_ctx`) to encourage entering menus, moving the cursor, highlighting the correct target, and avoiding stalling or premature closes.
  - Global shaping:
    - Rewards catching new Pokémon (party size increases), leveling up (via `read_first_mon_exp`), and successful healing.

The training loops (`train_end_to_end` and `phased_training`) compute reward components via `RewardSystem.compute_components`, then combine/clip them according to the `training.reward_clip` and per-phase logic.

### Vision encoder & shared representation

- `src/vision/encoder.py` implements `NatureCNN`, a standard DQN-style convolutional encoder producing a 512‑dimensional feature vector from `(1, 84, 84)` observations.
- The encoder is owned by the `Director` and shared by all specialists:
  - Training code always runs frames through `Director.encoder` before feeding them into the head networks.
  - The vision optimizer (`AdamW`) is defined in `train_end_to_end` and `phased_training`, and gradients from all specialist losses flow back through this shared encoder.

### Director, goals, and graph‑based novelty

- `src/agent/director.py` defines two main pieces:
  - `Goal` dataclass: structured representation of high-level directives (explore/train/survive/menu) with target conditions, max-steps, progress metrics, and status.
  - `Director` module:
    - Holds the shared `NatureCNN` encoder and a small MLP `router` that outputs logits over specialists.
    - Maintains `GraphMemory` (`src/agent/graph_memory.py`), a directed graph keyed by coarse state hashes (either RAM-based map/x/y or a downsampled image hash). This is used to:
      - Estimate novelty (new nodes) for exploration goals.
      - Provide potential backtracking paths; backtracking is exposed to the agent as forced navigation actions.
    - Implements a goal head MLP that scores candidate goal types, plus a small bias mechanism (`goal_bias`, `goal_specialist_map`) to nudge the router toward specialists aligned with the active goal.
    - Encapsulates all goal lifecycle management (creation from config, activation, progress updates, completion, and metrics for logging).

- `GraphMemory`:
  - Downsamples frames to a small grid, quantizes intensities, and uses the result as a node key when RAM-based coordinates are unavailable.
  - Stores per-node visit counts, prunes seldom-visited nodes when exceeding `max_nodes`, and can provide shortest paths back to a given state as sequences of actions.

### Hierarchical agent & specialists

- `src/agent/hierarchical_agent.py` wires everything together in `HierarchicalAgent`:
  - Takes `director_cfg` and `specialist_cfg` (from YAML) and constructs:
    - `Director`.
    - `CrossQNavAgent` (navigation head).
    - `CrossQBattleAgent` (battle head).
    - `MenuBrain` (goal‑conditioned menu head).
  - Maintains mappings between environment action IDs (PyBoy `WindowEvent` enums) and each specialist's local action indices via `allowed_actions` lists.
  - Implements `get_action(obs, info, epsilon)`:
    - Pushes the frame through the director to get a specialist index, potential forced navigation action (for backtracking), and optional `goal_ctx`.
    - Routes to the appropriate specialist `get_action` method, translating between local and global action indices.
    - For menu goals, enriches goal context with live menu state (cursor, open flag) and computes/attaches a goal embedding for training.
  - Provides save/load helpers both for the full agent and for individual components (`save_component`) so `train_multi_agent` can cherry-pick best brains across runs.

- Specialists (`src/agent/specialists/*`):
  - `nav_brain.CrossQNavAgent` and `battle_brain.CrossQBattleAgent`:
    - Lightweight MLP DQNs on top of the shared 512‑dim feature vector.
    - Use simple CrossQ‑style TD updates without target networks.
    - Expose `train_step_return_loss` that returns a scalar loss and diagnostic stats; outer loops own the optimizer steps so the encoder can be updated jointly.
  - `menu_brain.MenuBrain`:
    - Goal-conditioned DQN that concatenates the latent frame features with a compact goal embedding (menu target index, cursor row/col, depth, open flag).
    - Provides `encode_goal_batch` / `encode_goal` helpers and a `get_action` that optionally forces the "open menu" action when the menu is closed.
    - Returns loss and minimal metrics from `train_step`, leaving optimizer steps to callers for consistency with other heads.

### Replay buffer & logging

- `src/utils/memory_buffer.py` implements prioritized replay using segment trees:
  - `SegmentTree` gives O(log N) priority queries.
  - `PrioritizedReplayBuffer` stores observations (uint8), actions, rewards, dones, optional goal embeddings, and optional raw goal contexts.
  - Used by both `train_end_to_end` and `phased_training` to sample batches for all specialists with importance sampling weights and updatable priorities.

- `src/utils/logger.py` wraps `SummaryWriter`:
  - Recursively flattens nested metric dicts (e.g. `agent.director.get_goal_metrics()`) into scalar tags.
  - Every training loop uses `Logger.log_step` to record epsilon, buffer fill ratios, losses, and goal metrics.

### Experiment entrypoints & orchestration

- `experiments/train_end_to_end.py` is the primary single-run training script, responsible for:
  - Loading YAML config and applying CLI overrides.
  - Instantiating environment, reward system, hierarchical agent, buffers, and logger.
  - Main RL loop, including epsilon schedule, prioritized replay sampling, and checkpointing when losses or rewards improve.

- `experiments/phased_training.py` provides a phase-parameterized training loop (`train_phase`) plus shared CLI parser builder (`build_phase_arg_parser`):
  - Uses the same environment and reward code as end-to-end training.
  - Restricts replay buffers and update logic to the active specialist.
  - Adds early termination conditions specific to phases (e.g. nav episodes ending when a battle or menu is detected).

- `experiments/train_nav_phase.py`, `train_battle_phase.py`, `train_menu_phase.py` are thin wrappers around `train_phase` with different `PHASE` constants and CLIs.

- `experiments/train_multi_agent.py` handles multi-process orchestration, optional CPU core pinning (via `psutil`), aggregation of worker summaries, and optional best-brain selection in `combined_best/`.

These entrypoints and directory conventions (`checkpoints/`, `experiments/logs/`, `states/`) are the main integration points for new tooling or automation.
