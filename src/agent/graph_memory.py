import cv2
import networkx as nx
import numpy as np
from typing import Any, Optional, Tuple


class GraphMemory:
    """
    Lightweight topological map keyed by coarse vision hashes.

    Each unique 8x8 downsampled observation becomes a node, allowing the
    Director to reason about novelty and request backtracking paths.
    """

    def __init__(
        self,
        max_nodes: int = 5000,
        downsample_size: int = 8,
        quantization_step: int = 32,
    ):
        self.graph = nx.DiGraph()
        self.current_node: Optional[bytes] = None
        self.max_nodes = max_nodes
        self.downsample_size = downsample_size
        self.quantization_step = quantization_step

    def get_state_hash(self, observation: Any) -> bytes:
        """Compresses an observation tensor/array to an 8x8 hash."""
        if hasattr(observation, "cpu"):
            observation = observation.cpu().numpy()

        if observation.ndim == 4:
            img = observation[0, 0]
        elif observation.ndim == 3:
            img = observation[0]
        else:
            img = observation

        small = cv2.resize(
            img, (self.downsample_size, self.downsample_size), interpolation=cv2.INTER_AREA
        )
        quantized = (small // self.quantization_step) * self.quantization_step
        return quantized.tobytes()

    def update(self, observation: Any, last_action: Optional[int]) -> Tuple[bytes, bool]:
        """Registers the current observation and optional transition edge."""
        state_hash = self.get_state_hash(observation)
        is_new_state = False

        if not self.graph.has_node(state_hash):
            self.graph.add_node(state_hash, visits=1)
            is_new_state = True
            if self.graph.number_of_nodes() > self.max_nodes:
                self._prune_graph()
        else:
            self.graph.nodes[state_hash]["visits"] += 1

        if self.current_node is not None and last_action is not None:
            if self.current_node != state_hash:
                if not self.graph.has_edge(self.current_node, state_hash):
                    self.graph.add_edge(
                        self.current_node,
                        state_hash,
                        action=last_action,
                        weight=1,
                    )

        self.current_node = state_hash
        return state_hash, is_new_state

    def _prune_graph(self):
        """Removes rarely visited nodes to bound memory usage."""
        to_remove = [n for n, data in self.graph.nodes(data=True) if data.get("visits", 0) <= 1]
        if self.current_node in to_remove:
            to_remove.remove(self.current_node)
        if to_remove:
            self.graph.remove_nodes_from(to_remove)

    def find_path_to_start(self, start_node_hash: bytes) -> Optional[list[int]]:
        """Returns a sequence of actions that leads back to start_node_hash."""
        try:
            path = nx.shortest_path(
                self.graph, source=self.current_node, target=start_node_hash
            )
            actions: list[int] = []
            for i in range(len(path) - 1):
                data = self.graph.get_edge_data(path[i], path[i + 1])
                actions.append(data["action"])
            return actions
        except nx.NetworkXNoPath:
            return None
