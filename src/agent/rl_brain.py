"""
Abstract base class for RL algorithms (brains) that can be plugged into the Agent.
This allows easy swapping between different RL algorithms (CrossQ, BBF, Rainbow DQN, etc.)
while maintaining a consistent interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np


class RLBrain(ABC):
    """
    Abstract base class for RL algorithms.

    This interface standardizes how different RL algorithms interact with the
    environment, allowing you to swap algorithms by changing one line of code.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL brain.

        Args:
            config: Algorithm-specific configuration dictionary
        """
        self.config = config
        self.action_dim = config.get("action_dim", 8)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    @abstractmethod
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        goal: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Select an action given the current state.

        Args:
            state: Encoded state/features (shape: [1, feature_dim] or [batch, feature_dim])
            deterministic: If True, select greedy action; if False, use exploration
            goal: Optional goal context for goal-conditioned policies

        Returns:
            Selected action index (int)
        """
        pass

    @abstractmethod
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
        Store a transition in the replay buffer.

        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            next_state: Next state features
            done: Whether episode terminated
            **kwargs: Additional metadata (e.g., goal, priority)
        """
        pass

    @abstractmethod
    def train_step(self) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Perform one training update step.

        Returns:
            loss: PyTorch loss tensor (None if not enough data to train)
            metrics: Dictionary of training metrics for logging
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save the brain's state to disk.

        Args:
            path: File path to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load the brain's state from disk.

        Args:
            path: File path to load checkpoint from
        """
        pass

    @abstractmethod
    def set_train_mode(self, mode: bool = True) -> None:
        """
        Set the brain to training or evaluation mode.

        Args:
            mode: True for training mode, False for evaluation
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return the brain's configuration."""
        return self.config.copy()

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics/statistics from the brain.

        Returns:
            Dictionary of metrics (e.g., epsilon, buffer size, loss)
        """
        return {}


class ReplayBuffer:
    """
    Enhanced replay buffer with prioritization for sparse reward environments.
    Designed to be memory-efficient for GTX 1080 Ti (6GB VRAM).
    """

    def __init__(self, capacity: int, feature_dim: int, device: str = "cpu", alpha: float = 0.6):
        """
        Initialize replay buffer with prioritized sampling.

        Args:
            capacity: Maximum number of transitions to store
            feature_dim: Dimensionality of state features
            device: Device to store tensors on ("cpu" or "cuda")
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.device = device
        self.position = 0
        self.size = 0
        self.alpha = alpha

        # Pre-allocate memory for efficiency
        self.states = torch.zeros((capacity, feature_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, feature_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        # Priority tracking for sparse rewards
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """Add a transition to the buffer with priority."""
        self.states[self.position] = state.squeeze().to(self.device)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.squeeze().to(self.device)
        self.dones[self.position] = float(done)
        
        # Higher priority for non-zero rewards (important in sparse environments)
        if abs(reward) > 0.001:
            priority = self.max_priority
        else:
            priority = 0.1  # Low priority for step penalties
        
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a prioritized batch of transitions."""
        if self.alpha == 0:
            # Uniform sampling
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            # Prioritized sampling
            priorities = self.priorities[:self.size] ** self.alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(self.size, size=batch_size, p=probs)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices  # Return indices for priority updates
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6  # Small epsilon to avoid zero priorities
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def clear(self) -> None:
        """Clear the buffer."""
        self.position = 0
        self.size = 0
