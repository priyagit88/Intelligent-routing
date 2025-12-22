class TrustModel:
    def __init__(self, initial_trust=1.0, decay_factor=0.95, bonus_factor=0.05):
        self.node_trust = {}
        self.initial_trust = initial_trust
        self.decay_factor = decay_factor # Penalty for bad behavior
        self.bonus_factor = bonus_factor # Bonus for good behavior

    def initialize_node(self, node_id):
        if node_id not in self.node_trust:
            self.node_trust[node_id] = self.initial_trust

    def update_trust(self, node_id, success):
        """
        Updates trust score based on transaction success/failure.
        success: bool (True if packet delivered successfully, False otherwise)
        """
        if node_id not in self.node_trust:
            self.initialize_node(node_id)
        
        current = self.node_trust[node_id]
        if success:
            # Increase trust, capped at 1.0
            self.node_trust[node_id] = min(1.0, current + self.bonus_factor)
        else:
            # Decrease trust
            self.node_trust[node_id] = max(0.0, current * self.decay_factor)
    
    def get_trust(self, node_id):
        return self.node_trust.get(node_id, self.initial_trust)
