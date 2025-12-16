"""
Pokemon Agent - A shell that wraps any RLBrain implementation.

This agent handles:
- Frame preprocessing
- Interaction with the Pokemon environment
- Delegating learning to the pluggable brain
"""
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np

from src.agent.rl_brain import RLBrain


class PokemonAgent:
    """
    Shell agent that accepts any RLBrain implementation.

    This class is responsible ONLY for:
    1. Preprocessing observations (if needed)
    2. Routing actions to/from the brain
    3. Managing the brain's lifecycle (save/load)

    The actual learning algorithm is delegated to the brain.
    """

    def __init__(
        self,
        brain: RLBrain,
        allowed_actions: Optional[list] = None,
        device: str = "cuda"
    ):
        """
        Initialize the Pokemon agent.

        Args:
            brain: An RLBrain implementation (CrossQBrain, BBFBrain, etc.)
            allowed_actions: List of allowed action indices (default: all 8 buttons)
            device: Device to run on ("cuda" or "cpu")
        """
        self.brain = brain
        self.device = torch.device(device)
        self.allowed_actions = allowed_actions if allowed_actions is not None else list(range(8))

        # Verify that brain's action_dim matches the number of allowed actions
        # If they don't match, it's a configuration error that should be caught early
        if len(self.allowed_actions) != self.brain.action_dim:
            raise ValueError(
                f"Mismatch: Brain has action_dim={self.brain.action_dim}, "
                f"but allowed_actions has {len(self.allowed_actions)} actions. "
                f"Please ensure brain_config['action_dim'] matches len(allowed_actions)."
            )

        # Preprocessing config (can be expanded later)
        self.normalize_obs = True
        self.obs_dtype = torch.float32

    def encode_obs(
        self,
        obs: np.ndarray | torch.Tensor,
        text_embedding: Optional[np.ndarray | torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Preprocess raw observation into encoded features.

        Args:
            obs: Raw observation (grayscale image, shape [96, 96] or [1, 96, 96])

        Returns:
            Encoded feature tensor
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.obs_dtype)

        # Add batch and channel dimensions if needed
        if obs.dim() == 2:  # [H, W]
            obs = obs.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif obs.dim() == 3:  # [1, H, W] or [C, H, W]
            obs = obs.unsqueeze(0)  # [1, C, H, W]

        # Normalize to [0, 1] if needed
        if self.normalize_obs and obs.max() > 1.0:
            obs = obs / 255.0

        # Delegate encoding to brain
        features = self.brain.encode_obs(obs)
        if features.requires_grad:
            features = features.detach()

        if text_embedding is None or features.dim() != 2:
            return features

        text_tensor = self._prepare_text_tensor(text_embedding, features.size(0), features.device, features.dtype)
        return torch.cat([features, text_tensor], dim=1)

    def _prepare_text_tensor(
        self,
        text_embedding: np.ndarray | torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert a text embedding into a tensor aligned with feature batches.
        """
        if isinstance(text_embedding, torch.Tensor):
            text_tensor = text_embedding.to(device=device, dtype=dtype)
        else:
            text_tensor = torch.as_tensor(text_embedding, dtype=dtype, device=device)

        if text_tensor.dim() == 1:
            text_tensor = text_tensor.unsqueeze(0)

        if text_tensor.size(0) != batch_size:
            text_tensor = text_tensor.expand(batch_size, -1)
        return text_tensor

    def get_action(
        self,
        obs: np.ndarray | torch.Tensor,
        deterministic: bool = False,
        goal: Optional[Dict[str, Any]] = None,
        text_embedding: Optional[np.ndarray | torch.Tensor] = None
    ) -> int:
        """
        Select an action given the current observation.

        Args:
            obs: Raw observation from environment
            deterministic: If True, select greedy action
            goal: Optional goal context

        Returns:
            Environment action index (mapped to allowed actions)
        """
        # Encode observation
        features = self.encode_obs(obs, text_embedding=text_embedding)

        # Get action from brain (local action space)
        local_action = self.brain.get_action(features, deterministic, goal)

        # Map to environment action space
        env_action = self.allowed_actions[local_action]

        return env_action

    def get_action_with_features(
        self,
        obs: np.ndarray | torch.Tensor,
        deterministic: bool = False,
        goal: Optional[Dict[str, Any]] = None,
        text_embedding: Optional[np.ndarray | torch.Tensor] = None
    ) -> Tuple[int, int, torch.Tensor]:
        """
        Like get_action, but also returns features and local action.

        Useful for training when you need to store both the action
        and the encoded features.

        Returns:
            (env_action, local_action, features)
        """
        features = self.encode_obs(obs, text_embedding=text_embedding)
        local_action = self.brain.get_action(features, deterministic, goal)
        env_action = self.allowed_actions[local_action]

        return env_action, local_action, features

    def store_experience(
        self,
        obs: np.ndarray | torch.Tensor,
        action: int,
        reward: float,
        next_obs: np.ndarray | torch.Tensor,
        done: bool,
        text_embedding: Optional[np.ndarray | torch.Tensor] = None,
        next_text_embedding: Optional[np.ndarray | torch.Tensor] = None,
        **kwargs
    ) -> None:
        """
        Store a transition in the brain's replay buffer.

        Args:
            obs: Current observation
            action: Local action index (NOT environment action)
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated
            **kwargs: Additional metadata
        """
        # Encode observations
        state = self.encode_obs(obs, text_embedding=text_embedding)
        next_state = self.encode_obs(next_obs, text_embedding=next_text_embedding)

        # Store in brain
        self.brain.store_experience(state, action, reward, next_state, done, **kwargs)

    def train_step(self, global_step: Optional[int] = None) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Perform one training update.

        Args:
            global_step: Global step counter for epsilon decay and logging

        Returns:
            (loss, metrics) from the brain
        """
        return self.brain.train_step(global_step=global_step)

    def save_checkpoint(self, path: str) -> None:
        """Save agent state."""
        self.brain.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state."""
        self.brain.load_checkpoint(path)

    def set_train_mode(self, mode: bool = True) -> None:
        """Set training or evaluation mode."""
        self.brain.set_train_mode(mode)

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics from the brain."""
        return self.brain.get_metrics()

    def get_brain_config(self) -> Dict[str, Any]:
        """Get the brain's configuration."""
        return self.brain.get_config()


def create_agent(
    brain_type: str,
    brain_config: Dict[str, Any],
    allowed_actions: Optional[list] = None,
    device: str = "cuda",
    action_names: Optional[list] = None
) -> PokemonAgent:
    """
    Factory function to create agents with different brains.

    Args:
        brain_type: Name of brain type ("crossq", "bbf", "rainbow", "human")
        brain_config: Configuration dict for the brain
        allowed_actions: List of allowed action indices (default: all 8 buttons)
        device: Device to run on
        action_names: List of action names (required for "human" brain)

    Returns:
        PokemonAgent with the specified brain

    Example:
        agent = create_agent("crossq", config, device="cuda")
        agent = create_agent("human", {}, action_names=["DOWN", "LEFT", ...])
    """
    brain_config = brain_config.copy()  # Don't mutate the original config
    brain_config["device"] = device

    # Set action_dim based on allowed_actions
    if allowed_actions is not None and len(allowed_actions) > 0:
        brain_config["action_dim"] = len(allowed_actions)
    else:
        # Default to 8 if not specified
        allowed_actions = list(range(brain_config.get("action_dim", 8)))

    if brain_type.lower() == "crossq":
        from src.agent.crossq_brain import CrossQBrain
        brain = CrossQBrain(brain_config)
    elif brain_type.lower() == "human":
        from src.agent.human_brain import HumanBrain
        brain_config["action_names"] = action_names or []
        brain = HumanBrain(brain_config)
    elif brain_type.lower() == "bbf":
        # Placeholder for future BBF implementation
        raise NotImplementedError("BBF brain not yet implemented")
    elif brain_type.lower() == "rainbow":
        # Placeholder for future Rainbow DQN implementation
        raise NotImplementedError("Rainbow DQN brain not yet implemented")
    else:
        raise ValueError(f"Unknown brain type: {brain_type}")

    return PokemonAgent(brain, allowed_actions, device)
