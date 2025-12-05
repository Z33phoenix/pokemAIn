import networkx as nx
import numpy as np
import cv2

class GraphMemory:
    """
    Topological Map of the Game World.
    Nodes = Distinct Visual States (Downsampled hashes).
    Edges = Actions taken to move between them.
    """
    def __init__(self, max_nodes=5000):
        self.graph = nx.DiGraph()
        self.current_node = None
        self.max_nodes = max_nodes # Safety Cap
        
    def get_state_hash(self, observation):
        """
        Compresses 84x84 image into a coarse hash string.
        """
        # Safety Check: Ensure we have a valid numpy array (H, W) or (C, H, W)
        if hasattr(observation, 'cpu'): 
            observation = observation.cpu().numpy()
            
        # Handle (1, 84, 84) vs (84, 84)
        if observation.ndim == 3:
            img = observation[0]
        else:
            img = observation
            
        # Resize to 8x8 for coarse location matching
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Quantize: Round to nearest 32 to ignore small sprite animations
        quantized = (small // 32) * 32
        
        return quantized.tobytes()

    def update(self, observation, last_action=None):
        """
        Updates the graph with the current observation.
        """
        state_hash = self.get_state_hash(observation)
        
        # 1. Add Node if new
        if not self.graph.has_node(state_hash):
            self.graph.add_node(state_hash, visits=1)
            # RAM SAFETY: Prune if too big
            if self.graph.number_of_nodes() > self.max_nodes:
                self._prune_graph()
        else:
            self.graph.nodes[state_hash]['visits'] += 1
            
        # 2. Add Edge (Transition)
        if self.current_node is not None and last_action is not None:
            if self.current_node != state_hash:
                if not self.graph.has_edge(self.current_node, state_hash):
                    self.graph.add_edge(self.current_node, state_hash, action=last_action, weight=1)
        
        self.current_node = state_hash
        return state_hash

    def _prune_graph(self):
        """
        Removes 'Noise' nodes (visited only once) to save RAM.
        """
        # Find nodes with only 1 visit
        to_remove = [n for n, data in self.graph.nodes(data=True) if data.get('visits', 0) <= 1]
        
        # Don't delete the node we are currently standing on!
        if self.current_node in to_remove:
            to_remove.remove(self.current_node)
            
        if len(to_remove) > 0:
            self.graph.remove_nodes_from(to_remove)
            # print(f"[GRAPH] Pruned {len(to_remove)} noise nodes.")

    def find_path_to_start(self, start_node_hash):
        try:
            path = nx.shortest_path(self.graph, source=self.current_node, target=start_node_hash)
            actions = []
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                edge_data = self.graph.get_edge_data(u, v)
                actions.append(edge_data['action'])
            return actions
        except nx.NetworkXNoPath:
            return None