"""
New Director - Pluggable goal-setting coordinator.

This is the refactored Director that uses pluggable GoalStrategy objects.
It separates goal management from goal generation logic.

Key improvements over old director.py:
- Goal generation is delegated to GoalStrategy
- No longer creates fallback goals when goals are disabled
- Cleaner separation of concerns
- Compatible with reward strategies that check goal enablement

Usage:
    from src.agent.director_new import DirectorNew
    from src.agent.goal_strategy import create_goal_strategy

    # Create with LLM goals
    goal_strategy = create_goal_strategy("llm", config)
    director = DirectorNew(config, goal_strategy)

    # Or disable goals completely
    goal_strategy = create_goal_strategy("none", config)
    director = DirectorNew(config, goal_strategy)
"""

from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import torch

from src.agent.graph_memory import GraphMemory
from src.agent.goal_strategy import Goal, GoalStrategy


class EpisodicMemory:
    """Tracks the agent's recent history for LLM context."""

    def __init__(self):
        self.events = []
        self._seen_narratives = set()  # Track unique narrative texts to avoid duplicates

    def log_event(self, step: int, event_type: str, detail: str):
        """Log a significant event (Map change, Battle, Item)."""
        # For NARRATIVE_TEXT, check if we've seen this exact text before (anywhere in history)
        # This prevents duplicate logs when AI talks to the same NPC multiple times
        if event_type == "NARRATIVE_TEXT":
            if detail in self._seen_narratives:
                return  # Skip duplicate narrative
            self._seen_narratives.add(detail)

        entry = f"[{step}] {event_type}: {detail}"
        # Avoid duplicate consecutive logs
        if self.events and self.events[-1] == entry:
            return
        self.events.append(entry)
        #print("Memory Log:", self.events)

    def consume_history(self) -> str:
        """
        Returns the log of events since the last call, ordered MOST RECENT -> OLDEST.
        Clears the buffer after reading.
        """
        if not self.events:
            return "No recent events."

        history_str = "\n".join(reversed(self.events))
        self.events.clear()
        self._seen_narratives.clear()  # Reset seen narratives for new context window
        return history_str


class Director:
    """
    Lightweight coordinator that manages goal queue and delegates goal generation
    to a pluggable GoalStrategy.

    Key differences from old Director:
    - Uses GoalStrategy for goal generation
    - Doesn't create fallback goals when strategy is disabled
    - Cleaner interface for training loop
    """

    def __init__(self, config: Dict[str, Any], goal_strategy: GoalStrategy):
        """
        Initialize the Director.

        Args:
            config: Configuration dictionary
            goal_strategy: GoalStrategy instance (LLM, heuristic, or none)
        """
        self.config = config
        self.goal_strategy = goal_strategy

        # Graph memory for spatial tracking
        graph_cfg = config.get("graph", {})
        self.graph = GraphMemory(
            max_nodes=graph_cfg.get("max_nodes", 5000),
            downsample_size=graph_cfg.get("downsample_size", 8),
            quantization_step=graph_cfg.get("quantization_step", 32),
        )

        # Goal management
        self.active_goal: Optional[Goal] = None
        self.goal_queue: List[Goal] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.goal_counter = 0

        # Tracking
        self.prev_battle_active = False
        self.max_goal_capacity = config.get("max_goal_capacity", 3)
        self.poll_goal_threshold = config.get("goal_poll_threshold", 2)
        self._goal_executor = ThreadPoolExecutor(max_workers=1)
        self._goal_request_future: Optional[Future] = None

    def encode_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Flatten normalized observation tensor into a feature vector."""
        if obs.dim() == 4:
            obs = obs.squeeze(0)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs.view(1, -1).to(torch.float32)

    def encode_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Flatten a batch of observations."""
        if obs_batch.max() > 1.0:
            obs_batch = obs_batch / 255.0
        return obs_batch.view(obs_batch.size(0), -1).to(torch.float32)

    def select_specialist(
        self, obs: torch.Tensor, info: Dict[str, Any], epsilon: float
    ) -> Tuple[int, torch.Tensor, Optional[int], Optional[Dict[str, Any]]]:
        """
        Returns (specialist_index, features, forced_action, goal_ctx).

        Note: goal_ctx will be None if goal_strategy is disabled.
        """
        obs_cpu = obs.detach().cpu().numpy()
        self.graph.update(obs_cpu, info, last_action=None)

        # Activate next goal from queue if no active goal
        if self.active_goal is None and self.goal_queue:
            self._activate_goal(self.goal_queue.pop(0))

        features = self.encode_features(obs)

        # CRITICAL FIX: Only return goal_ctx if strategy is enabled
        goal_ctx = None
        if self.goal_strategy.enabled() and self.active_goal:
            goal_ctx = self.active_goal.as_dict()

        return (
            0,  # always navigation specialist
            features,
            None,  # no forced action
            goal_ctx  # None if goals disabled
        )

    def poll_for_goal(self, state_summary: Dict[str, Any], step: int, update_frequency: int) -> Optional[Goal]:
        """
        Asynchronously poll the goal strategy for a new goal when capacity permits.
        """
        ready_goal = self._harvest_async_goal()
        if ready_goal:
            return ready_goal

        if not self.goal_strategy.enabled():
            return None

        if self._request_in_flight() or self._goal_executor is None:
            return None

        total_goals = self._current_goal_count()
        if total_goals >= self.max_goal_capacity:
            return None
        if total_goals >= self.poll_goal_threshold:
            return None

        if not self.goal_strategy.should_generate_goal(step, update_frequency):
            return None

        summary_copy = deepcopy(state_summary)
        self._goal_request_future = self._goal_executor.submit(
            self.goal_strategy.generate_goal,
            summary_copy,
            self
        )
        return None

    def _current_goal_count(self) -> int:
        """Compute total goals including the active one."""
        return (1 if self.active_goal else 0) + len(self.goal_queue)

    def _request_in_flight(self) -> bool:
        return self._goal_request_future is not None and not self._goal_request_future.done()

    def _harvest_async_goal(self) -> Optional[Goal]:
        """Check if an async goal request finished and return the result."""
        if self._goal_request_future is None:
            return None

        if not self._goal_request_future.done():
            return None

        try:
            goal = self._goal_request_future.result()
        except Exception as exc:
            print(f"[Director] Goal request failed: {exc}")
            goal = None
        finally:
            self._goal_request_future = None

        if goal and self._current_goal_count() < self.max_goal_capacity:
            return goal
        return None

    def shutdown(self):
        """Cleanup executor resources."""
        if self._goal_executor:
            self._goal_executor.shutdown(wait=False)
            self._goal_executor = None
        self._goal_request_future = None

    def enqueue_goal(self, goal: Goal, current_step: int = 0):
        """Add a goal to the queue, sorted by priority."""
        self.goal_queue.append(goal)
        self.goal_queue.sort(key=lambda g: g.priority, reverse=True)

    def complete_goal(self, status: str = "done"):
        """Mark the active goal as complete and archive it."""
        if self.active_goal:
            self.active_goal.status = status
            self.completed_goals.append(self.active_goal.as_dict())
            self.active_goal = None

    def clear_goals(self):
        """Clear active goal and queue (used on map changes)."""
        self.active_goal = None
        self.goal_queue.clear()

    def get_last_completed_goal(self) -> Optional[Dict[str, Any]]:
        """Get the most recently completed goal."""
        if not self.completed_goals:
            return None
        return self.completed_goals[-1]

    def get_goal_metrics(self) -> Dict[str, float]:
        """Get metrics for logging."""
        return {
            "goal/active": 1 if self.active_goal else 0,
            "goal/queue_len": len(self.goal_queue),
            "goal/completed": len(self.completed_goals),
            "goal/enabled": 1 if self.goal_strategy.enabled() else 0,
        }

    def _activate_goal(self, goal: Goal):
        """Move a goal from queue to active."""
        self.active_goal = goal
        self.active_goal.status = "active"
