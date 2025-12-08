# Pokémon Red RL + LLM Goals

Single-agent RL controller for Pokémon Red guided by a local, text-only LLM that sets high-level navigation goals. No vision to the LLM—only clean JSON summaries. The RL agent handles all button presses; the LLM provides concise goals via Ollama.

## What’s Here
- `train.py` — Main training loop with per-goal episodes, ratcheted checkpoints, and synchronous LLM goal requests.
- `src/agent/agent.py` — CrossQ-style navigation agent (the only neural net).
- `src/agent/goal_llm.py` — HTTP client for the LLM endpoint (`/api/chat`).
- `src/agent/director.py` — Builds JSON state summaries, sanitizes LLM goals, and routes them to the agent.
- `src/agent/quest_manager.py` — Simple stage tracking and checkpoint paths for the “ratchet” progression.
- `src/env/pokemon_red_gym.py` — PyBoy gym wrapper with save/load state support.
- `config/ModelFile.txt` — Ollama model definition and system prompt for the goal-setter.
- `config/hyperparameters.yaml` — Training, rewards, agent, and LLM configuration.
- `config/walkthrough_steps.json` — Minimal stage progression (extend as needed).

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
```
python train.py --run-name <NAME> --state-path states/initial.state
```
Behavior:
- Waits for an LLM goal before acting.
- Each goal is an episode. Success may continue without reset; failure/timeout reloads the latest stage checkpoint (ratchet).
- Map changes trigger a fresh LLM poll and clear stale micro-goals.
- Progress bar shows reward, episode count, and current goal. Logs to `experiments/logs/<run>_*`.

## Config Highlights (`config/hyperparameters.yaml`)
- `goal_llm`: `enabled`, `api_url`, `model` (`pokemon-goal`), `timeout`, `debug`.
- `agent`: learning rate, gamma, allowed_actions, input_dim (96x96 flattened).
- `rewards`: shaping, goal bonuses, timeout penalty.
- `saves_dir`: where ratchet checkpoints are stored (default `saves/`).

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

## Checkpoints and Ratchet
- Stage checkpoints saved to `saves/stage_<idx>.state` when a stage completes (QuestManager).
- Agent checkpoints under `experiments/<run>/agent_brain_latest.pth`.
- TensorBoard logs under `experiments/logs/<run>_*`.

## Troubleshooting
- If stuck with no goal: ensure `ollama serve` is running and the `pokemon-goal` model exists.
- Enable LLM debug in `hyperparameters.yaml` to print requests/responses.
- Map-change loops: we re-poll immediately and clear stale goals; no retry delay.

