# Pokémon Red RL + Modular AI Architecture

Pluggable RL agent for Pokémon Red with swappable components: RL algorithms (CrossQ, BBF, Rainbow), goal-setting strategies (LLM, Heuristic, Reactive), and reward systems. The modular design allows easy experimentation with different AI approaches while maintaining a consistent interface.

## What's Here

### Core Training
- `train.py` — Main training script with pluggable architecture for brain, strategy, and reward systems.
- `setup_game.py` — Game initialization utility.
- `watch_agent.py` — Agent monitoring tool.

### Agent Components
- `src/agent/pokemon_agent.py` — Main agent shell that wraps any RL brain implementation.
- `src/agent/rl_brain.py` — Abstract base class for RL algorithms.
- `src/agent/crossq_brain.py` — CrossQ implementation for continuous control.
- `src/agent/director.py` — Builds JSON state summaries and manages goal routing.
- `src/agent/goal_llm.py` — HTTP client for LLM endpoint integration.
- `src/agent/goal_strategy.py` — Strategy pattern for different goal-setting approaches.
- `src/agent/graph_memory.py` — Graph-based memory system for navigation.

### Environment & Games
- `src/core/env_factory.py` — Factory for creating game environments.
- `src/core/game_interface.py` — Unified game interface abstraction.
- `src/games/gb/pokemon_red_gym.py` — PyBoy wrapper for Pokémon Red/Blue.
- `src/games/gb/game_data.py` — Game-specific data structures and utilities.
- `src/games/gb/ram_map.py` — Memory mapping for game state extraction.
- `src/games/gb/text_decoder.py` — Text extraction from game memory.

### Vision & Processing
- `src/vision/encoder.py` — Visual processing and encoding.

### Configuration
- `config/brain_configs.yaml` — RL algorithm configurations with memory presets.
- `config/hyperparameters.yaml` — Training, rewards, and environment settings.
- `config/ModelFile.txt` — Ollama model definition for LLM goal-setting.
- `config/walkthrough_steps.json` — Stage progression definitions.

## Prerequisites
- Python 3.10+ and PyTorch (CUDA optional).
- `pokemon_red.gb` in the project root.
- Ollama installed and running (`ollama serve`).

## Ollama Setup
1) Start Ollama:
```
ollama serve
```
2) Build the goal model from `config/ModelFile.txt`:
```
ollama create pokemon-goal -f config/ModelFile.txt
```
3) The default endpoint is `http://localhost:11434/api/chat` (matches `goal_llm.api_url` in `config/hyperparameters.yaml`).

## Running Training

### Basic Usage
```bash
# LLM-based training with CrossQ algorithm
python train.py --brain crossq --strategy llm

# Pure reactive RL (no goals, no goal rewards)
python train.py --brain crossq --strategy reactive

# Heuristic goals with CrossQ
python train.py --brain crossq --strategy heuristic

# Hybrid approach with memory optimization
python train.py --brain crossq --strategy hybrid --memory-preset low
```

### Available Options
- **Brains**: `crossq`, `bbf`, `rainbow` - Different RL algorithms
- **Strategies**: `llm`, `heuristic`, `reactive`, `hybrid` - Goal-setting approaches
- **Memory Presets**: `minimal`, `low`, `medium`, `high` - Memory usage optimization

### Training Behavior
- Strategy-dependent goal setting (LLM waits for goals, reactive acts immediately).
- Episode management varies by strategy - some use goal-based episodes, others continuous learning.
- Map changes trigger strategy-specific responses (fresh goals for LLM, continued execution for reactive).
- Progress tracking and logging to `experiments/logs/<timestamp>/`.

## Configuration Files

### `config/hyperparameters.yaml` - Main Configuration
- `game`: Game selection (currently `pokemon_red`)
- `environment`: ROM path, emulation settings, action configuration
- `training`: Learning parameters, batch sizes, update frequencies
- `rewards`: Comprehensive reward shaping system
- `goal_llm`: LLM integration settings (for LLM strategy)

### `config/brain_configs.yaml` - RL Algorithm Settings
- `memory_presets`: Pre-configured memory usage levels (minimal, low, medium, high)
- Algorithm-specific parameters for CrossQ, BBF, Rainbow brains
- Hardware-optimized defaults for different GPU memory constraints

### `config/ModelFile.txt` - LLM Model Definition
- Ollama model configuration for goal-setting LLM
- System prompts and response formatting

### `config/walkthrough_steps.json` - Stage Progression
- Defines game progression stages and checkpoints

## State Summary Sent to the LLM
After the first hardcoded startup prompt, every request includes:
```
{
  "location": { "map_name", "map_id", "x", "y", "nearby_sprites": [...] },
  "party": [...],
  "inventory": { "key_items": [...], "hms_owned": [...], "items": [...] },
  "game_state": { "badges", "money", "battle_status" },
  "last_goal": { "target", "status" }
}
```
The LLM returns JSON with `goal_type`, `target_map_name` (or `target_location_name`), and optional metadata. The code sanitizes and enqueues the goal.

## Testing the Endpoint Manually (CMD)
```
curl -X POST "http://localhost:11434/api/chat" -H "Content-Type: application/json" -d "{\"model\":\"pokemon-goal\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"{\\\"location\\\":{\\\"map_name\\\":\\\"Red's house 2F\\\",\\\"map_id\\\":38,\\\"x\\\":3,\\\"y\\\":6,\\\"nearby_sprites\\\":[]},\\\"party\\\":[],\\\"inventory\\\":{\\\"key_items\\\":[],\\\"hms_owned\\\":[],\\\"items\\\":[]},\\\"game_state\\\":{\\\"badges\\\":0,\\\"money\\\":3000,\\\"battle_status\\\":\\\"Overworld\\\"},\\\"last_goal\\\":{\\\"target\\\":\\\"Start Game\\\",\\\"status\\\":\\\"New\\\"}}\"}]}"
```

## Checkpoints and Logging
- Agent model checkpoints saved to `experiments/<timestamp>/` directory
- TensorBoard logs under `experiments/logs/<timestamp>/`
- Game state saves managed by individual strategies (stage-based for goal strategies)

## Troubleshooting
- **LLM Strategy Issues**: Ensure `ollama serve` is running and the `pokemon-goal` model exists.
- **Memory Issues**: Use `--memory-preset minimal` or `low` for limited GPU memory.
- **Performance**: Try different brain algorithms (`crossq`, `bbf`, `rainbow`) based on your hardware.
- **Debugging**: Enable debug options in `config/hyperparameters.yaml` for detailed logging.
- **Strategy Problems**: Switch strategies if current approach isn't working (`--strategy reactive` for simplest setup).

