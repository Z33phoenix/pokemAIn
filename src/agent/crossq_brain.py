"""
CrossQ Brain implementation - a sample-efficient Q-learning variant.
This is a concrete implementation of the RLBrain interface.
"""
from typing import Any, Dict, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.rl_brain import RLBrain, ReplayBuffer
from src.vision.encoder import NatureCNN
from src.utils.human_buffer import HumanExperienceBuffer, load_human_dataset


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
        self.fc_input_dim = fc_input_dim

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
        
        # Exploration improvements for sparse rewards
        self.action_counts = np.zeros(self.action_dim)  # Track action frequency
        self.state_visit_counts = {}  # Track state visitation for novelty
        self.exploration_bonus_decay = 0.99  # Decay rate for exploration bonuses

        # Training settings
        self.batch_size = config.get("batch_size", 32)
        self.min_buffer_size = config.get("min_buffer_size", 1000)

        # Optimizer
        params = list(self.q_net.parameters())
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        self.optimizer = optim.AdamW(params, lr=config.get("learning_rate", 1e-4))
        self.loss_fn = nn.MSELoss()

        # Enhanced replay buffer with prioritization
        buffer_capacity = config.get("buffer_capacity", 50000)
        prioritization_alpha = config.get("prioritization_alpha", 0.6)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            feature_dim=fc_input_dim,
            device="cpu",  # Store on CPU to save VRAM
            alpha=prioritization_alpha
        )
        self.human_batch_size = int(config.get("human_batch_size", 0))
        if self.human_batch_size <= 0:
            human_ratio = float(config.get("human_batch_ratio", 0.0))
            self.human_batch_size = int(round(self.batch_size * human_ratio))
        self.human_batch_size = min(self.human_batch_size, self.batch_size)
        self.human_buffer_min_size = int(config.get("human_buffer_min_size", max(1, self.human_batch_size)))
        self.human_buffer: Optional[HumanExperienceBuffer] = None
        human_buffer_path = config.get("human_buffer_path")
        if human_buffer_path:
            self._load_human_buffer(human_buffer_path)

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

        # Enhanced exploration for sparse rewards
        if np.random.random() < epsilon:
            # Count-based exploration: prefer less-tried actions
            action_probs = 1.0 / (self.action_counts + 1)
            action_probs /= action_probs.sum()
            action = np.random.choice(self.action_dim, p=action_probs)
            self.action_counts[action] += 1
            return action

        # Exploitation with exploration bonus
        self.set_train_mode(False)
        with torch.no_grad():
            q_values = self.q_net(state.to(self.device))
            
            # Add small exploration bonus to Q-values based on action frequency
            exploration_bonus = 0.1 / (torch.tensor(self.action_counts, device=self.device) + 1)
            q_values_with_bonus = q_values + exploration_bonus.unsqueeze(0)
            
            action = torch.argmax(q_values_with_bonus, dim=1).item()
        self.set_train_mode(True)

        self.action_counts[action] += 1
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
        human_batch = 0
        if self.human_buffer and len(self.human_buffer) >= self.human_buffer_min_size:
            human_batch = min(self.human_batch_size, self.batch_size)
        agent_batch = self.batch_size - human_batch

        if agent_batch > 0 and len(self.replay_buffer) < max(self.min_buffer_size, agent_batch):
            return None, {
                "brain/buffer_size": len(self.replay_buffer),
                "brain/human_buffer": len(self.human_buffer) if self.human_buffer else 0,
            }
        if agent_batch == 0 and human_batch == 0:
            return None, {
                "brain/buffer_size": len(self.replay_buffer),
                "brain/human_buffer": len(self.human_buffer) if self.human_buffer else 0,
            }

        # Sample batch with indices for priority updates
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        indices = None
        agent_sample_count = 0

        if agent_batch > 0:
            states, actions, rewards, next_states, dones, indices = self.replay_buffer.sample(agent_batch)
            agent_sample_count = agent_batch
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)

        if human_batch > 0 and self.human_buffer:
            h_states, h_actions, h_rewards, h_next_states, h_dones = self.human_buffer.sample(human_batch)
            batch_states.append(h_states)
            batch_actions.append(h_actions)
            batch_rewards.append(h_rewards)
            batch_next_states.append(h_next_states)
            batch_dones.append(h_dones)

        states = torch.cat(batch_states, dim=0)
        actions = torch.cat(batch_actions, dim=0)
        rewards = torch.cat(batch_rewards, dim=0)
        next_states = torch.cat(batch_next_states, dim=0)
        dones = torch.cat(batch_dones, dim=0)

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

        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors = (q_pred - q_target).abs().cpu().numpy()
            if indices is not None and agent_sample_count > 0:
                self.replay_buffer.update_priorities(indices, td_errors[:agent_sample_count])

        # Metrics
        td_error_mean = td_errors.mean()
        metrics = {
            "brain/loss": loss.item(),
            "brain/q_pred_mean": q_pred.mean().item(),
            "brain/td_error": td_error_mean,
            "brain/epsilon": self.epsilon,
            "brain/buffer_size": len(self.replay_buffer),
            "brain/human_buffer": len(self.human_buffer) if self.human_buffer else 0,
            "brain/max_replay_buffer_priority": self.replay_buffer.max_priority,
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
            "brain/human_buffer": len(self.human_buffer) if self.human_buffer else 0,
        }

    def _prepare_obs_tensor(self, obs_batch: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(obs_batch)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _encode_human_observations(
        self,
        frames: np.ndarray,
        text_embeddings: Optional[np.ndarray]
    ) -> torch.Tensor:
        if frames.size == 0:
            return torch.zeros((0, self.fc_input_dim), dtype=torch.float32)

        batch_size = 64
        encoded_batches = []
        total = frames.shape[0]
        text_array = None
        if text_embeddings is not None:
            text_array = torch.from_numpy(text_embeddings).float()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            obs_tensor = self._prepare_obs_tensor(frames[start:end]).to(self.device)
            with torch.no_grad():
                features = self.encode_obs(obs_tensor)
            features = features.detach().cpu()
            if text_array is not None:
                text_batch = text_array[start:end]
                if text_batch.dim() == 1:
                    text_batch = text_batch.unsqueeze(0)
                features = torch.cat([features, text_batch], dim=1)
            encoded_batches.append(features)

        return torch.cat(encoded_batches, dim=0)

    def _load_human_buffer(self, path: str) -> None:
        dataset = load_human_dataset(path)
        if dataset is None:
            print(f"ƒsÿ Human buffer not found at {path}")
            return

        data, metadata = dataset
        obs = data.get("obs")
        next_obs = data.get("next_obs")
        if obs is None or next_obs is None:
            print(f"ƒsÿ Incomplete human buffer at {path}")
            return

        text_embeddings = data.get("text_embeddings")
        next_text_embeddings = data.get("next_text_embeddings")
        encoded_states = self._encode_human_observations(obs, text_embeddings)
        encoded_next_states = self._encode_human_observations(next_obs, next_text_embeddings)

        actions_arr = data.get("actions")
        rewards_arr = data.get("rewards")
        dones_arr = data.get("dones")
        if actions_arr is None or rewards_arr is None or dones_arr is None:
            print(f"ƒsÿ Missing fields in human buffer at {path}")
            return

        actions = torch.from_numpy(actions_arr).long()
        rewards = torch.from_numpy(rewards_arr).float()
        dones = torch.from_numpy(dones_arr).float()

        self.human_buffer = HumanExperienceBuffer.from_tensors(
            states=encoded_states,
            actions=actions,
            rewards=rewards,
            next_states=encoded_next_states,
            dones=dones
        )
        size = len(self.human_buffer)
        print(f"ƒo\" Loaded human buffer ({size} samples) from {path}")
