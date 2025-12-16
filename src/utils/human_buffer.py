"""
Utilities for recording and loading human demonstration buffers.

The recorder is used while a human player controls the agent to capture the
exact observations, actions, and rewards encountered. The loader provides a
simple sampling API so RL brains can mix human and agent data during training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _to_numpy_frame(frame: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert an observation/frame to a contiguous numpy array."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def _to_numpy_vector(vector: Optional[np.ndarray | torch.Tensor], dim: int) -> np.ndarray:
    """Convert optional text embedding to numpy."""
    if vector is None:
        return np.zeros(dim, dtype=np.float32)
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.zeros(dim, dtype=np.float32)
    if arr.shape[-1] != dim:
        padded = np.zeros(dim, dtype=np.float32)
        length = min(dim, arr.shape[-1])
        padded[:length] = arr[:length]
        return padded
    return arr.astype(np.float32, copy=False)


class HumanExperienceRecorder:
    """
    Collect raw human demonstration transitions and persist them to disk.
    """

    def __init__(self, capacity: int, output_path: str, text_feature_dim: int = 0):
        self.capacity = capacity
        self.output_path = output_path
        self.text_feature_dim = text_feature_dim
        self._buffer: list[Dict[str, np.ndarray]] = []

    def record(
        self,
        obs: np.ndarray | torch.Tensor,
        action: int,
        env_action: int,
        reward: float,
        next_obs: np.ndarray | torch.Tensor,
        done: bool,
        text_embedding: Optional[np.ndarray | torch.Tensor] = None,
        next_text_embedding: Optional[np.ndarray | torch.Tensor] = None
    ) -> None:
        if self.is_full:
            return
        entry = {
            "obs": _to_numpy_frame(obs),
            "action": int(action),
            "env_action": int(env_action),
            "reward": float(reward),
            "next_obs": _to_numpy_frame(next_obs),
            "done": bool(done),
            "text_embedding": _to_numpy_vector(text_embedding, self.text_feature_dim),
            "next_text_embedding": _to_numpy_vector(next_text_embedding, self.text_feature_dim),
        }
        self._buffer.append(entry)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity > 0

    def save(self) -> str:
        if not self._buffer:
            return self.output_path

        obs = np.stack([entry["obs"] for entry in self._buffer])
        next_obs = np.stack([entry["next_obs"] for entry in self._buffer])
        actions = np.array([entry["action"] for entry in self._buffer], dtype=np.int64)
        env_actions = np.array([entry["env_action"] for entry in self._buffer], dtype=np.int64)
        rewards = np.array([entry["reward"] for entry in self._buffer], dtype=np.float32)
        dones = np.array([entry["done"] for entry in self._buffer], dtype=np.float32)
        text_embeddings = np.stack([entry["text_embedding"] for entry in self._buffer])
        next_text_embeddings = np.stack([entry["next_text_embedding"] for entry in self._buffer])

        payload = {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "env_actions": env_actions,
            "rewards": rewards,
            "dones": dones,
            "text_embeddings": text_embeddings,
            "next_text_embeddings": next_text_embeddings,
        }
        metadata = {
            "num_steps": self.size,
            "text_feature_dim": self.text_feature_dim,
        }

        _ensure_dir(self.output_path)
        torch.save({"data": payload, "metadata": metadata}, self.output_path)
        return self.output_path


def load_human_dataset(path: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, int]]]:
    """Load a recorded human dataset from disk."""
    if not path or not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu")
    data = payload.get("data", payload)
    metadata = payload.get("metadata", {})
    return data, metadata


@dataclass
class HumanExperienceBuffer:
    """Static experience buffer backed by tensors for uniform sampling."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor

    @classmethod
    def from_tensors(
        cls,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> "HumanExperienceBuffer":
        return cls(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if len(self) == 0:
            raise ValueError("HumanExperienceBuffer is empty.")
        indices = np.random.randint(0, len(self), size=batch_size)
        idx = torch.from_numpy(indices)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
