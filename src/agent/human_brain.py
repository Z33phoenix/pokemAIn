"""
HumanBrain - A 'brain' that takes keyboard input as the RL agent.

This allows manual gameplay to go through the full training pipeline:
- Memory logging
- Reward calculation
- Experience storage
- All diagnostic information

Perfect for debugging and verifying the training infrastructure works
without waiting for an RL agent to explore.

Keyboard Controls:
  W/A/S/D = Move UP/LEFT/DOWN/RIGHT
  Z/X = A/B Buttons
  P/O = START/SELECT Buttons
  C = Save state (prompts for name)
  L = Load state (shows list to choose)
"""

from abc import ABC
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from src.agent.rl_brain import RLBrain


class HumanBrain(RLBrain):
    """
    A 'brain' that reads keyboard input from the user.

    Implements the RLBrain interface so it integrates seamlessly into
    the training pipeline. The player becomes the RL agent, but all
    the infrastructure (rewards, memory, logging) still works normally.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the human brain.

        Args:
            config: Configuration dict (must have action_names list)
        """
        super().__init__(config)
        self.action_names = config.get("action_names", [])
        self.action_map = {name: idx for idx, name in enumerate(self.action_names)}
        # Find "NOOP" action if it exists, otherwise use None (no default action)
        self.last_action = self.action_map.get("NOOP", None)
        self._print_help()

    def _print_help(self):
        """Print keyboard controls to user."""
        print("\n" + "="*70)
        print("HUMAN BRAIN MODE - You are the RL Agent")
        print("="*70)
        print("\nAvailable actions:")
        for idx, name in enumerate(self.action_names):
            print(f"  {idx}: {name}", end="  ")
            if (idx + 1) % 4 == 0:
                print()
        print("\n\nKeyboard Controls:")
        print("  W = UP    | S = DOWN  | A = LEFT  | D = RIGHT")
        print("  Z = A Button | X = B Button | P = START | O = SELECT")
        print("\nState Management (type in terminal):")
        print("  c = Save state | l = Load state")
        print("\nThe game runs with NORMAL training pipeline:")
        print("  ✓ All rewards calculated")
        print("  ✓ Memory logging active")
        print("  ✓ Battle logic processed")
        print("  ✓ All diagnostics displayed")
        print("="*70 + "\n")

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation (pass-through for human brain).

        Human brain doesn't need to encode observations since it doesn't
        use neural networks. Just return as-is.
        """
        # Ensure it's on the right device
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        # Convert from numpy if needed
        return torch.from_numpy(obs).float().to(self.device)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        goal: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Get action from keyboard input.

        Args:
            state: Current state (unused, but required by interface)
            deterministic: Ignored (human doesn't use exploration)
            goal: Ignored (human doesn't use goal context)

        Returns:
            Action index selected by user (or idle action if no input)
        """
        # Default to idle action (NOOP or first action if NOOP doesn't exist)
        idle_action = self.last_action if self.last_action is not None else 0

        if not HAS_MSVCRT:
            # Fallback on non-Windows - return idle action
            return idle_action

        try:
            action = self._read_keyboard_input()
            if action is not None:
                self.last_action = action
                return action
        except Exception:
            # If keyboard reading fails, return idle action
            pass

        # No input, return idle action
        return idle_action

    def _read_keyboard_input(self) -> Optional[int]:
        """
        Read keyboard input and map to action.

        Returns:
            Action index, or None if no valid input
        """
        if not msvcrt.kbhit():
            return None

        try:
            key = msvcrt.getch().decode('utf-8', errors='ignore').upper()
        except Exception:
            return None

        # Movement keys
        if key == 'W' and 'UP' in self.action_map:
            return self.action_map['UP']
        elif key == 'S' and 'DOWN' in self.action_map:
            return self.action_map['DOWN']
        elif key == 'A' and 'LEFT' in self.action_map:
            return self.action_map['LEFT']
        elif key == 'D' and 'RIGHT' in self.action_map:
            return self.action_map['RIGHT']

        # Button keys
        elif key == 'Z' and 'A' in self.action_map:
            return self.action_map['A']
        elif key == 'X' and 'B' in self.action_map:
            return self.action_map['B']
        elif key == 'P' and 'START' in self.action_map:
            return self.action_map['START']
        elif key == 'O' and 'SELECT' in self.action_map:
            return self.action_map['SELECT']

        # State management commands (signal to trainer, not action)
        elif key == 'C':
            print("\n[HUMAN] Press 'c' in terminal to save state")
            return self.last_action
        elif key == 'L':
            print("\n[HUMAN] Press 'l' in terminal to load state")
            return self.last_action

        return None

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        **kwargs
    ) -> None:
        """
        Store experience (no-op for human, but required by interface).

        The training pipeline still logs everything, we just don't train
        a neural network.
        """
        pass

    def train_step(self, global_step: Optional[int] = None) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Training step (no-op for human brain).

        Args:
            global_step: Global step counter (ignored for human brain)

        Returns:
            (None, {}) - no loss, no metrics to compute
        """
        return None, {}

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint (no-op for human brain)."""
        pass

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint (no-op for human brain)."""
        pass

    def set_train_mode(self, mode: bool = True) -> None:
        """Set train mode (no-op for human brain)."""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        action_value = self.last_action if self.last_action is not None else 0
        return {"brain/last_action": float(action_value)}
