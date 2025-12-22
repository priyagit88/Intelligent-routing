import networkx as nx

class RoutingAlgorithm:
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, source, target):
        raise NotImplementedError("Subclasses must implement find_path")

class ShortestPathRouting(RoutingAlgorithm):
    def find_path(self, source, target):
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight='weight')
        except nx.NetworkXNoPath:
            return None

class IntelligentRouting(RoutingAlgorithm):
    def __init__(self, graph, trust_model):
        super().__init__(graph)
        self.trust_model = trust_model

    def calculate_cost(self, u, v, data):
        """
        Custom cost function that considers:
        - Link latency (weight)
        - Destination Node Trust (or next hop trust)
        """
        base_cost = data.get('weight', 1)
        trust_score = self.trust_model.get_trust(v)
        
        # Invert trust so lower trust = higher cost
        # cost = base_weight * (1 + (1 - trust))
        # If trust is 1.0, cost = base_weight
        # If trust is 0.0, cost = base_weight * 2 (or more extreme penalty)
        trust_penalty = (1 - trust_score) * 10 # Multiplier for impact
        
        return base_cost * (1 + trust_penalty)

    def find_path(self, source, target):
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight=self.calculate_cost)
        except nx.NetworkXNoPath:
            return None

class RLRouting(RoutingAlgorithm):
    def __init__(self, graph, agent):
        super().__init__(graph)
        self.agent = agent
    
    def find_path(self, source, target):
        """
        Route packet hop-by-hop using Q-Learning Agent.
        Note: This is different from Dijkstra. We don't return a full path upfront typically in RL routing,
        but for this simulation compatibility, we will simulate the hop-by-hop decisions to generate a 'path'.
        """
        path = [source]
        current = source
        visited = set([source])
        max_hops = 20
        
        while current != target and len(path) < max_hops:
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
                
            next_hop = self.agent.choose_action(current, neighbors)
            
            if next_hop is None:
                break
            
            # Prevent simple loops
            if next_hop in visited:
                 # In pure RL, loops are discouraged by penalty, but we force break for routing utility
                 # Or we can allow it if we trust the agent to eventually exit. 
                 # For simulation speed, we'll avoid immediate loops or re-visiting.
                 # But standard Q-routing might re-visit. Let's block it for now.
                 pass

            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            
        if current == target:
            return path
        return None

class RIPRouting(RoutingAlgorithm):
    """
    Simulates RIP (Routing Information Protocol).
    - Metric: Hop Count (Distance Vector).
    - Ignores Link Latency/Congestion.
    """
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, source, target):
        try:
            # Shortest path with weight=None implies BFS (Hop Count)
            return nx.shortest_path(self.graph, source=source, target=target, weight=None)
        except nx.NetworkXNoPath:
            return None
