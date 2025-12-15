"""
CrossQ Brain implementation - a sample-efficient Q-learning variant.
This is a concrete implementation of the RLBrain interface.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.rl_brain import RLBrain, ReplayBuffer
from src.vision.encoder import NatureCNN


class CrossQBrain(RLBrain):
    """
    CrossQ implementation without target network.
    Optimized for sample efficiency on limited hardware.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CrossQ brain.

        Config keys:
            - action_dim: Number of discrete actions
            - feature_dim: CNN encoder output dimension
            - input_dim: Raw input dimension (if no encoder)
            - use_encoder: Whether to use NatureCNN encoder
            - learning_rate: Optimizer learning rate
            - gamma: Discount factor
            - epsilon_start: Initial exploration rate
            - epsilon_end: Final exploration rate
            - epsilon_decay: Steps to decay epsilon over
            - buffer_capacity: Replay buffer size
            - batch_size: Training batch size
            - min_buffer_size: Minimum buffer size before training
            - device: "cuda" or "cpu"
        """
        super().__init__(config)

        # Network architecture
        use_encoder = config.get("use_encoder", True)
        feature_dim = int(config.get("feature_dim", 512))
        input_dim = int(config.get("input_dim", 9216))  # 96x96 pixels
        self.text_feature_dim = int(config.get("text_feature_dim", 0))

        self.encoder = NatureCNN(
            input_channels=1, 
            input_height=160, 
            input_width=240, 
            features_dim=feature_dim
        ) if use_encoder else None
        if self.encoder is not None:
            fc_input_dim = feature_dim + self.text_feature_dim
        else:
            fc_input_dim = input_dim + self.text_feature_dim

        self.q_net = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

        # Move networks to device
        if self.encoder is not None:
            self.encoder.to(self.device)
        self.q_net.to(self.device)

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 100000)
        self.epsilon_start = self.epsilon
        self.steps_done = 0

        # Training settings
        self.batch_size = config.get("batch_size", 32)
        self.min_buffer_size = config.get("min_buffer_size", 1000)

        # Optimizer
        params = list(self.q_net.parameters())
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        self.optimizer = optim.AdamW(params, lr=config.get("learning_rate", 1e-4))
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        buffer_capacity = config.get("buffer_capacity", 50000)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            feature_dim=fc_input_dim,
            device="cpu"  # Store on CPU to save VRAM
        )

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode raw observations into feature vector."""
        if self.encoder is not None:
            return self.encoder(obs.to(self.device))

        # If no encoder, flatten the observation
        if obs.dim() > 2:
            if obs.max() > 1.0:
                obs = obs / 255.0
            return obs.view(obs.size(0), -1).to(self.device)
        return obs.to(self.device)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        goal: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Encoded state features [1, feature_dim]
            deterministic: If True, always pick best action
            goal: Optional goal context to adjust epsilon

        Returns:
            Selected action index
        """
        epsilon = 0.0 if deterministic else self.epsilon

        # Goal-specific epsilon adjustments
        if goal and not deterministic:
            goal_type = goal.get("goal_type")
            if goal_type == "survive":
                epsilon = min(epsilon, 0.05)
            elif goal_type == "train":
                epsilon = min(epsilon, 0.2)
            elif goal_type and goal_type != "explore":
                epsilon = min(epsilon, 0.2)

        # Exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        # Exploitation
        self.set_train_mode(False)
        with torch.no_grad():
            q_values = self.q_net(state.to(self.device))
            action = torch.argmax(q_values, dim=1).item()
        self.set_train_mode(True)

        return action

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        **kwargs
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Perform one CrossQ training update.

        Returns:
            loss: PyTorch loss tensor (None if not enough data)
            metrics: Training metrics dictionary
        """
        # Not enough data yet
        if len(self.replay_buffer) < self.min_buffer_size:
            return None, {"brain/buffer_size": len(self.replay_buffer)}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Move to device for computation
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        q_values = self.q_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values (no target network - this is CrossQ)
        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            q_target = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(q_pred, q_target)

        # Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self._update_epsilon()

        # Metrics
        with torch.no_grad():
            td_error = (q_pred - q_target).abs().mean().item()

        metrics = {
            "brain/loss": loss.item(),
            "brain/q_pred_mean": q_pred.mean().item(),
            "brain/td_error": td_error,
            "brain/epsilon": self.epsilon,
            "brain/buffer_size": len(self.replay_buffer),
        }

        return loss, metrics

    def _update_epsilon(self) -> None:
        """Linearly decay epsilon."""
        self.steps_done += 1
        if self.epsilon > self.epsilon_end:
            decay_amount = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)

    def save_checkpoint(self, path: str) -> None:
        """Save brain state to disk."""
        checkpoint = {
            "q_net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "config": self.config,
        }
        if self.encoder is not None:
            checkpoint["encoder"] = self.encoder.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load brain state from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps_done = checkpoint.get("steps_done", 0)

        if self.encoder is not None and "encoder" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder"])

    def set_train_mode(self, mode: bool = True) -> None:
        """Set training or evaluation mode."""
        if self.encoder is not None:
            self.encoder.train(mode)
        self.q_net.train(mode)

    def get_metrics(self) -> Dict[str, float]:
        """Get current brain metrics."""
        return {
            "brain/epsilon": self.epsilon,
            "brain/buffer_size": len(self.replay_buffer),
            "brain/steps_done": self.steps_done,
        }
