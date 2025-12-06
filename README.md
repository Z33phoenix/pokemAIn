# Pokemon Red Hierarchical RL Agent

A hierarchical RL stack for Pokemon Red built on PyBoy. A shared vision encoder feeds a **Director** that routes control to phase-specific **Specialists** (navigation, battle, menu). Training is phased around curated save states so each specialist masters its domain before full integration.

## What the AI Looks Like
- **Director (`src/agent/director.py`)**: Chooses a goal (explore, train, survive, menu) and biases routing to the right specialist. Maintains a lightweight graph for novelty and basic goal bookkeeping.
- **Navigation Brain (`src/agent/specialists/nav_brain.py`)**: CrossQ-style DQN head for overworld movement. Episodes auto-end if a battle starts to keep nav clean.
- **Battle Brain (`src/agent/specialists/battle_brain.py`)**: Distributional (Rainbow-like) head with NoisyLinear exploration for fights.
- **Menu Brain (`src/agent/specialists/menu_brain.py`)**: Goal-conditioned DQN that learns to open/start menus and move the cursor to targets (bag, party, PC, or START).
- **Environment (`src/env/pokemon_red_gym.py`)**: Gymnasium wrapper around PyBoy; pulls only rewards/signals from RAM; supports multiple starting states per phase and handles user window close gracefully.
- **Rewards (`src/env/rewards.py`)**: Configurable shaping for nav/battle/menu with menu-open bonuses and exploration/battle heuristics wired through `ram_map.py`.

## Repository Layout
- `config/hyperparameters.yaml`: Central training and env settings (state paths, replay sizes, reward weights).
- `experiments/`: Training entry points (`train_*_phase.py`, `train_end_to_end.py`, `train_multi_agent.py`) plus shared `phased_training.py`.
- `setup_game.py`: Interactive state capture tool; saves to `states/<phase>/...` with optional metadata sidecars.
- `src/agent/`: Director, graph memory, hierarchical agent wiring, and specialists.
- `src/env/`: Gym wrapper, RAM map helpers, reward shaping.
- `src/utils/`: Replay buffer, logging utilities.

## Prerequisites
- Python 3.10+ recommended.
- `pokemon_red.gb` in the project root.
- Optional GPU (CUDA) for faster training.
- Install deps: `pip install -r requirements.txt` (consider a venv/conda env).

## Installation
```bash
git clone <repo>
cd pokemAIn
pip install -r requirements.txt
```
Ensure `pokemon_red.gb` is present in the repo root before running anything.

## Capturing Training States
Phased training relies on curated starting points:
1) Run `python setup_game.py`.
2) Play to the scenario you want (route, trainer battle, menu, or no-menu to train opening START).
3) Close the PyBoy window when ready; you will be prompted to pick a phase (nav/battle/menu/full).
4) States are auto-saved under `states/<phase>/<phase>_YYYYMMDD-HHMMSS.state` with optional `.meta.json`.

Tips:
- Use `--resume-from` to branch from an existing state (e.g., `states/initial.state`).
- `states/` is ignored by git; keep your local captures there.

## Training Workflows
Single-phase specialists (uses random state selection from the provided directory):
- Navigation: `python experiments/train_nav_phase.py --state-path states/nav --run-name nav001`
- Battle: `python experiments/train_battle_phase.py --state-path states/battle --run-name bat001`
- Menu: `python experiments/train_menu_phase.py --state-path states/menu --run-name menu001`

Multi-agent scale-out (spawns N workers and can combine best brains):
```bash
python experiments/train_multi_agent.py --phase nav --state-path states/nav --num-agents 4 --combine-best
```

Full integration (director + all specialists together):
```bash
python experiments/train_end_to_end.py --state-path states/initial.state --run-name full001
# or via multi-agent
python experiments/train_multi_agent.py --phase full --num-agents 2
```

Notes:
- Navigation episodes terminate early if a battle begins; supply varied nav states that avoid battles when possible.
- Menu rewards include a small bonus for being inside a menu to encourage opening START when needed.
- Closing the PyBoy window mid-run will end the episode and can stop training cleanly.

## Checkpoints and Logs
- Checkpoints: `checkpoints/<run_name>/` (director + specialist weights; per-phase best tags when available).
- Logs: `experiments/logs/<run_name>/` (TensorBoard scalars).
- Multi-agent runs write aggregated summaries to `checkpoints/<prefix>/run_summaries.json`; combined best brains land in `checkpoints/<prefix>/combined_best` when `--combine-best` is used.

## Configuration
Tune `config/hyperparameters.yaml` for:
- Environment paths (`state_path`, `phase_states`), emulation speed, action repeat, headless vs windowed.
- Replay sizes per phase, epsilon schedule, reward weights, and menu shaping.
- Specialist action sets and learning rates; director goal biases and graph memory sizing.

## Troubleshooting
- If a reset fails with “PyBoy window is closed,” restart the script; closing the window signals termination.
- Ensure `states/initial.state` exists for full training; recreate via `python setup_game.py` if needed.
- On Windows, SDL2 warnings are expected when using `pysdl2-dll`; training should proceed.
