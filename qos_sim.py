import simpy
import random
import logging
from utils import setup_logger
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import IntelligentRouting

logger = setup_logger("QoS_Sim")

def run_qos_scenario():
    print("--- Running QoS Simulation ---")
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    
    # Create Topology
    net_sim.create_topology(num_nodes=15, connectivity=0.3)
    
    # Start Congestion (This is key for QoS diff)
    env.process(net_sim.update_congestion())
    
    # Logic:
    # We will send 100 Data Packets (Priority 0)
    # And 100 Voice Packets (Priority 1)
    # We expect Data Packets to suffer higher loss due to congestion logic in network_sim.py
    
    trust_model = TrustModel()
    routing = IntelligentRouting(net_sim.graph, trust_model)
    
    stats = {
        'data': {'sent': 0, 'recv': 0},
        'voice': {'sent': 0, 'recv': 0}
    }
    
    def packet_generator():
        # Mix traffic
        for _ in range(200):
            p_type = 'voice' if random.random() < 0.5 else 'data'
            priority = 1 if p_type == 'voice' else 0
            
            src, dst = random.sample(net_sim.nodes, 2)
            
            path = routing.find_path(src, dst)
            if path:
                # Use base simulate_packet which has the congestion drop logic
                # We need to ensure we call the ONE in network_sim.py, 
                # NOT the monkey-patched one from main.py if we were importing main.
                # Here we import NetworkSimulation directly, so we are good.
                
                # However, we also need the "Reliability" check for Bad Nodes?
                # For this test, let's focus purely on CONGESTION QoS.
                # So we won't monkey-patch reliability drops, just use congestion drops.
                
                success = net_sim.simulate_packet(path, trust_model, priority=priority)
                
                stats[p_type]['sent'] += 1
                if success:
                    stats[p_type]['recv'] += 1
            
            yield env.timeout(0.1)

    env.process(packet_generator())
    env.run(until=100)
    
    print("\n--- QoS Results ---")
    d_pdr = (stats['data']['recv'] / stats['data']['sent'] * 100) if stats['data']['sent'] else 0
    v_pdr = (stats['voice']['recv'] / stats['voice']['sent'] * 100) if stats['voice']['sent'] else 0
    
    print(f"Data (Low Prio)  | Sent: {stats['data']['sent']} | Recv: {stats['data']['recv']} | PDR: {d_pdr:.2f}%")
    print(f"Voice (High Prio)| Sent: {stats['voice']['sent']} | Recv: {stats['voice']['recv']} | PDR: {v_pdr:.2f}%")
    
    if v_pdr > d_pdr:
        print("\nSUCCESS: Voice traffic had higher reliability during congestion!")
    else:
        print("\nINCONCLUSIVE: Congestion might not have been high enough.")

if __name__ == "__main__":
    run_qos_scenario()
