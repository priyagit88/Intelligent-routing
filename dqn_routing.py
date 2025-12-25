from routing import RoutingAlgorithm

class DQNRouting(RoutingAlgorithm):
    def __init__(self, graph, agent):
        super().__init__(graph)
        self.agent = agent
        
    def find_path(self, source, target):
        path = [source]
        current = source
        visited = set([source])
        max_hops = 20
        
        while current != target and len(path) < max_hops:
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
                
            # DQNAgent needs target_node in choose_action
            next_hop = self.agent.choose_action(current, target, neighbors)
            
            if next_hop is None:
                break
                
            if next_hop in visited:
                 # Loop prevention
                 pass

            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            
        if current == target:
            return path
        return None
