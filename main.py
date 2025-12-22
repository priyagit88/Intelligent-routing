import simpy
import networkx as nx
import random
import logging
from utils import setup_logger
import simpy
import networkx as nx
import random
import logging
from utils import setup_logger
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import ShortestPathRouting, IntelligentRouting, RLRouting
from rl_agent import QLearningAgent

logger = setup_logger("Main")

def run_simulation(env, net_sim, trust_model, routing_algo, num_packets=50):
    """
    Generates traffic and attempts to route it using the given algorithm.
    """
    logger.info(f"Starting simulation with {routing_algo.__class__.__name__}")
    
    success_count = 0
    total_latency = 0
    
    # Simple traffic generator
    # We will pick 10 random source-dest pairs/flows
    flows = []
    nodes = net_sim.nodes
    for _ in range(10):
        src, dst = random.sample(nodes, 2)
        flows.append((src, dst))
        
    for i in range(num_packets):
        src, dst = random.choice(flows)
        
        # 1. Routing Decision
        path = routing_algo.find_path(src, dst)
        
        if not path:
            logger.debug(f"Packet {i}: No path found from {src} to {dst}")
            continue
            
        # 2. Simulate Transmission
        # Calculate theoretical latency based on path weights
        path_latency = sum(net_sim.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        
        # 3. Simulate success/failure (Trust/Reliability check)
        success = net_sim.simulate_packet(path, trust_model)
        
        if success:
            success_count += 1
            total_latency += path_latency
            
        # 4. Feedback Loop for RL (if applicable)
        if isinstance(routing_algo, RLRouting):
            # We need to backtrack or update along the path
            # Simplification: The agent learns from the entire path experience
            # Reward formulation:
            # - High penalty for failure (-100)
            # - Negative latency for success (e.g., -Latency) to minimize it
            
            reward = -path_latency if success else -100
            
            # Update Q-Values along the path
            # We iterate properly: Q(u, v)
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                # Next neighbors needed for Q-update (max next Q)
                next_neighbors = list(net_sim.graph.neighbors(v)) if v in net_sim.graph else []
                routing_algo.agent.learn(u, v, reward, v, next_neighbors)
            
        # Yield some time between packets
        yield env.timeout(random.uniform(0.1, 0.5))
        
    logger.info(f"--- Results for {routing_algo.__class__.__name__} ---")
    logger.info(f"Total Packets: {num_packets}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Avg Latency (of successful): {total_latency/success_count if success_count else 0:.2f}")
    return success_count

def main():
    # Setup Environment
    env = simpy.Environment()
    
    # 1. Create Network
    net_sim = NetworkSimulation(env)
    # Use fixed seed for reproducibility across algos
    random.seed(42)
    net_sim.create_topology(num_nodes=15, connectivity=0.2)
    
    # 2. Introduce some "Bad" nodes to make it interesting
    # Let's say node 5 and 7 are unreliable
    # In a real sim, the NetworkSim class would handle the physics of this,
    # and the TrustModel would learn it.
    # For now, we simulate this by pre-seeding the TrustModel or just hoping the simulation picks it up.
    # Actually, the simulate_packet method uses the trust_model's current belief to determine drop probability?
    # NO: simulate_packet should use the "Real" node status, and TrustModel should LEARN it.
    # But my current implementation of simulate_packet uses drop_prob = (1 - trust).
    # This means the "Trust Model" IS the "Ground Truth" for reliability in my simple code.
    # That's circular. I need to separate "Ground Truth Reliability" from "Perceived Trust".
    
    # Let's Refactor slightly here in main or monkey-patch for the demo
    # We will assign "Real Reliability" to nodes in the graph
    bad_nodes = [3, 7]
    for n in net_sim.nodes:
        net_sim.graph.nodes[n]['reliability'] = 0.98 # Good nodes
    
    for n in bad_nodes:
        if n in net_sim.graph.nodes:
            net_sim.graph.nodes[n]['reliability'] = 0.6 # Bad nodes
            logger.info(f"Setting Node {n} as Unreliable (60%)")

    # Monkey patch simulate_packet to use 'reliability' attribute instead of trust score for truth
    def realistic_simulate_packet(path, trust_model):
        if not path: return False
        success = True
        for i in range(len(path) - 1):
             u, v = path[i], path[i+1]
             reliability = net_sim.graph.nodes[v].get('reliability', 1.0)
             if random.random() > reliability:
                 logger.debug(f"Packet dropped at REAL bad node {v}")
                 success = False
                 if trust_model:
                     trust_model.update_trust(v, False) # Algorithm learns
                 break
             else:
                 if trust_model:
                     trust_model.update_trust(v, True)
        return success
    
    net_sim.simulate_packet = realistic_simulate_packet

    # Import Visualization
    from visualization import visualize_network
    visualize_network(net_sim.graph, filename="network_initial.png")

    # 3. Runs
    # A) Standard Routing (Shortest Path unaware of trust)
    print("\n--- Simulation 1: Standard Dijkstra ---")
    
    # Start Congestion Process
    env.process(net_sim.update_congestion())
    
    trust_model_std = TrustModel() # Reset trust
    routing_std = ShortestPathRouting(net_sim.graph)
    proc1 = env.process(run_simulation(env, net_sim, trust_model_std, routing_std, num_packets=50))
    env.run(until=proc1)
    
    # B) Intelligent Routing
    print("\n--- Simulation 2: Intelligent Routing ---")
    env2 = simpy.Environment() # New time
    
    # Re-create net_sim structure for fair comparison or reuse graph?
    # If we reuse graph, it has modified weights from run 1.
    # Better to Reset weights or re-init net_sim.
    # For simplicity, let's just reset weights in the same graph or let it carry over (dynamic env).
    # Ideally, we want to compare ALGORITHMS on SAME conditions.
    # So we should re-seed congestion or use same seed.
    
    # Let's create a NEW simulation instance for fair comparison
    net_sim2 = NetworkSimulation(env2)
    random.seed(42) # Same topology
    net_sim2.create_topology(num_nodes=15, connectivity=0.2)
    # Re-apply bad nodes
    for n in net_sim2.nodes: net_sim2.graph.nodes[n]['reliability'] = 0.98
    for n in bad_nodes: 
        if n in net_sim2.graph.nodes: net_sim2.graph.nodes[n]['reliability'] = 0.6
    
    net_sim2.simulate_packet = realistic_simulate_packet # Patch again
    
    # Start Congestion
    env2.process(net_sim2.update_congestion())
    
    trust_model_int = TrustModel()
    routing_int = IntelligentRouting(net_sim2.graph, trust_model_int)
    proc2 = env2.process(run_simulation(env2, net_sim2, trust_model_int, routing_int, num_packets=50))
    env2.run(until=proc2)

    # C) RL Routing
    print("\n--- Simulation 3: Q-Learning Routing (Training Phase) ---")
    # Need many more packets to learn
    env3 = simpy.Environment()
    net_sim3 = NetworkSimulation(env3)
    random.seed(42)
    net_sim3.create_topology(num_nodes=15, connectivity=0.2)
    for n in net_sim3.nodes: net_sim3.graph.nodes[n]['reliability'] = 0.98
    for n in bad_nodes: 
        if n in net_sim3.graph.nodes: net_sim3.graph.nodes[n]['reliability'] = 0.6
    net_sim3.simulate_packet = realistic_simulate_packet
    env3.process(net_sim3.update_congestion())

    # Initialize Agent
    rl_agent = QLearningAgent(net_sim3.nodes, epsilon=0.5) # High exploration initially
    routing_rl = RLRouting(net_sim3.graph, rl_agent)
    
    # Train heavily
    proc3 = env3.process(run_simulation(env3, net_sim3, None, routing_rl, num_packets=500))
    env3.run(until=proc3)
    
    # Test Phase (Exploit)
    print("\n--- Simulation 3b: Q-Learning Routing (Testing Phase) ---")
    rl_agent.epsilon = 0.05 # Reduce exploration
    env3_test = simpy.Environment()
    # Continue with same network/agent state
    net_sim3.env = env3_test
    # Restart congestion process on new env
    env3_test.process(net_sim3.update_congestion())
    
    proc3_test = env3_test.process(run_simulation(env3_test, net_sim3, None, routing_rl, num_packets=50))
    env3_test.run(until=proc3_test)


    # Visualize Final Trust State
    visualize_network(net_sim2.graph, trust_model=trust_model_int, filename="network_final_trust.png")

if __name__ == "__main__":
    main()
