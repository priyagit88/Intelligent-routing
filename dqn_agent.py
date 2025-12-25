import numpy as np
import random
from collections import deque

class SimpleMLP:
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        
        # Xavier Initialization
        self.W1 = np.random.randn(input_dim, 64) * np.sqrt(1/input_dim)
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 64) * np.sqrt(1/64)
        self.b2 = np.zeros((1, 64))
        self.W3 = np.random.randn(64, output_dim) * np.sqrt(1/64)
        self.b3 = np.zeros((1, output_dim))
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2) # ReLU
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3 # Linear output (Q-values)
    
    def train(self, x, target):
        # Forward pass
        pred = self.forward(x)
        
        # Loss derivative (MSE): dL/dpred = 2 * (pred - target) / N
        # But we only update for specific actions, so target has 'correct' values for other actions?
        # Typically in DQN update: target = pred, but for action 'a', target[a] = y
        # So diff is 0 for non-actions.
        
        m = x.shape[0]
        grad_output = 2 * (pred - target) / m
        
        # Backprop
        # Layer 3
        d_W3 = np.dot(self.a2.T, grad_output)
        d_b3 = np.sum(grad_output, axis=0, keepdims=True)
        d_a2 = np.dot(grad_output, self.W3.T)
        
        # Layer 2
        d_z2 = d_a2 * (self.z2 > 0) # ReLU deriv
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = np.dot(d_z2, self.W2.T)
        
        # Layer 1
        d_z1 = d_a1 * (self.z1 > 0) # ReLU deriv
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # Update
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        
        return np.mean((pred - target)**2)

    def get_weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.W3.copy(), self.b3.copy()]
        
    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [w.copy() for w in weights]

class DQNAgent:
    def __init__(self, num_nodes, alpha=0.001, gamma=0.9, epsilon=0.1, buffer_size=1000, batch_size=32):
        self.num_nodes = num_nodes
        self.input_dim = num_nodes * 2 
        self.output_dim = num_nodes 
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        self.model = SimpleMLP(self.input_dim, self.output_dim, learning_rate=alpha)
        self.target_model = SimpleMLP(self.input_dim, self.output_dim, learning_rate=alpha)
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def _encode_state(self, current_node, target_node):
        state = np.zeros((1, self.input_dim))
        state[0, current_node] = 1
        state[0, self.num_nodes + target_node] = 1
        return state

    def choose_action(self, current_node, target_node, neighbors):
        if not neighbors:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        
        state = self._encode_state(current_node, target_node)
        q_values = self.model.forward(state)[0] # Shape (output_dim,)
        
        # Filter
        neighbor_q = {n: q_values[n] for n in neighbors}
        max_q = max(neighbor_q.values())
        best_nodes = [n for n, q in neighbor_q.items() if q == max_q]
        
        return random.choice(best_nodes)
    
    def learn(self, current_node, action, reward, next_node, target_node, done):
        state = self._encode_state(current_node, target_node) # (1, dim)
        next_state = self._encode_state(next_node, target_node)
        
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        if len(self.replay_buffer) < self.batch_size:
            return
            
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare Batch
        # states -> (batch, input_dim)
        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch]).reshape(-1, 1)
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch]).reshape(-1, 1)
        
        # Target
        current_q = self.model.forward(states)
        next_q_target = self.target_model.forward(next_states)
        max_next_q = np.max(next_q_target, axis=1, keepdims=True)
        
        target_q = current_q.copy()
        for i in range(self.batch_size):
            target_val = rewards[i] + (1 - dones[i]) * self.gamma * max_next_q[i]
            target_q[i, actions[i]] = target_val.item()
            
        self.model.train(states, target_q)
        
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
