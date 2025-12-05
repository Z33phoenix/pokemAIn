import numpy as np
import torch

class SegmentTree:
    """
    Helper structure for O(log N) priority sampling.
    """
    def __init__(self, size, operation, init_value):
        self.size = size
        self.operation = operation
        self.tree = np.full(2 * size, init_value, dtype=np.float32)

    def _operate(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate(start, end, 2 * node, node_start, mid)
        else:
            if start > mid:
                return self._operate(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate(start, mid, 2 * node, node_start, mid),
                    self._operate(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def add(self, idx, value):
        idx += self.size
        self.tree[idx] = value
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])

    def get_total(self):
        return self.tree[1]

    def find(self, value):
        idx = 1
        while idx < self.size:
            if self.tree[2 * idx] >= value:
                idx = 2 * idx
            else:
                value -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.size

class PrioritizedReplayBuffer:
    """
    Robust Experience Replay with Priority Support (PER).
    - Stores images as uint8 to save RAM (4x space saving).
    - Returns float32 tensors for training.
    """
    def __init__(self, capacity, obs_shape, alpha=0.6, device="cpu"):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device
        self.pos = 0
        self.full = False
        
        # Pre-allocate memory for speed
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action_buf = np.zeros((capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        
        # Priority Structures
        self.sum_tree = SegmentTree(capacity, operation=lambda x, y: x + y, init_value=0.0)
        self.min_tree = SegmentTree(capacity, operation=min, init_value=float('inf'))
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done):
        """Add a new experience to the buffer."""
        self.obs_buf[self.pos] = obs
        self.next_obs_buf[self.pos] = next_obs
        self.action_buf[self.pos] = action
        self.reward_buf[self.pos] = reward
        self.done_buf[self.pos] = float(done)
        
        # New experiences get max priority to ensure they are seen at least once
        self.sum_tree.add(self.pos, self.max_priority ** self.alpha)
        self.min_tree.add(self.pos, self.max_priority ** self.alpha)
        
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences.
        Args:
            batch_size (int): Number of samples.
            beta (float): Importance Sampling correction factor (0.4 -> 1.0 over training).
        Returns:
            Tuple of Tensors + Indices + Weights
        """
        current_len = self.capacity if self.full else self.pos
        indices = []
        weights = []
        
        total_priority = self.sum_tree.get_total()
        # Divide probability space into k segments
        segment = total_priority / batch_size
        
        # Calculate max weight for normalization
        min_prob = self.min_tree.get_total() / total_priority
        # Guard against zero priority initialization
        if min_prob == 0: min_prob = 1e-10 
        max_weight = (current_len * min_prob) ** (-beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            val = np.random.uniform(a, b)
            idx = self.sum_tree.find(val)
            indices.append(idx)
            
            # Calculate Importance Sampling Weight
            prob = self.sum_tree.tree[idx + self.capacity] / total_priority
            weight = (current_len * prob) ** (-beta)
            weights.append(weight / max_weight)

        indices = np.array(indices)
        
        # Convert to Tensors on the correct device
        obs = torch.as_tensor(self.obs_buf[indices], dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.as_tensor(self.next_obs_buf[indices], dtype=torch.float32, device=self.device) / 255.0
        actions = torch.as_tensor(self.action_buf[indices], device=self.device)
        rewards = torch.as_tensor(self.reward_buf[indices], device=self.device)
        dones = torch.as_tensor(self.done_buf[indices], device=self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        return obs, actions, rewards, next_obs, dones, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities after a training step based on TD-error."""
        for idx, priority in zip(indices, priorities):
            priority = float(priority + 1e-5) # Add epsilon to prevent 0 probability
            self.sum_tree.add(idx, priority ** self.alpha)
            self.min_tree.add(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.capacity if self.full else self.pos