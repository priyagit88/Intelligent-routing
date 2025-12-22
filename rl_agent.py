import random
import numpy as np

class QLearningAgent:
    def __init__(self, nodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-Learning Agent for Routing.
        nodes: list of all node IDs
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nodes = nodes
        
        # Q-Table: Q[current_node][next_hop] -> Value
        # We can use a dict of dicts for flexibility
        self.q_table = {node: {} for node in nodes}
        
        # Initialize Q-values
        for u in nodes:
            for v in nodes:
                if u != v:
                    # We only care about neighbors, but we might not know neighbors upfront 
                    # if we assume model-free. However, for routing, next_hop MUST be a neighbor.
                    # We will initialize entries lazily or assume 0.
                    pass

    def get_q_value(self, state, action):
        """State = current node, Action = next hop neighbor"""
        return self.q_table[state].get(action, 0.0)

    def choose_action(self, current_node, neighbors):
        """
        Epsilon-greedy selection of next hop.
        """
        if not neighbors:
            return None
            
        if random.random() < self.epsilon:
            # Explore
            return random.choice(neighbors)
        else:
            # Exploit: Choose neighbor with max Q-value
            q_values = [self.get_q_value(current_node, n) for n in neighbors]
            max_q = max(q_values)
            
            # Tie-breaking
            best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_neighbors):
        """
        Q-Learning Update Rule:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)
        
        # Calculate max Q for next state
        if not next_neighbors:
            max_next_q = 0.0 # Terminal or dead end
        else:
            max_next_q = max([self.get_q_value(next_state, n) for n in next_neighbors])
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
