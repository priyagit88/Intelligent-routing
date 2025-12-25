from dqn_agent import DQNAgent
import numpy as np

print("Initializing DQN Agent...")
agent = DQNAgent(num_nodes=20)
print("Agent initialized.")

print("Choosing action...")
action = agent.choose_action(current_node=0, target_node=19, neighbors=[1, 2, 3])
print(f"Action chosen: {action}")

print("Learning step...")
agent.learn(current_node=0, action=action, reward=10.0, next_node=action, target_node=19, done=False)
print("Learning step 1 complete.")

# Run a loop to check speed
print("Running 100 learning steps...")
for i in range(100):
    agent.learn(current_node=0, action=action, reward=10.0, next_node=action, target_node=19, done=False)
    if i % 10 == 0:
        print(f"Step {i}")

print("Smoke test complete.")
