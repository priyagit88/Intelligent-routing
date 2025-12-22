import simpy
import random
import logging
from utils import setup_logger
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import IntelligentRouting, ShortestPathRouting
from security import Adversary

logger = setup_logger("Attack_Sim")

def run_attack_scenario(attack_type="blackhole"):
    print(f"\n--- Running Challenge: {attack_type.upper()} Attack ---")
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    net_sim.create_topology(num_nodes=15, connectivity=0.3)
    
    # 1. Setup Adversaries
    # Node 5 and 9 will be malicious
    malicious_nodes = [5, 9]
    adversaries = {n: Adversary(n, attack_type) for n in malicious_nodes}
    
    for adv in adversaries.values():
        env.process(adv.update_behavior(env))
        
    # Monkey-patch simulation to use Adversary logic
    def secure_simulate_packet(path, trust_model, priority=0):
        if not path: return False
        success = True
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Check if V is an adversary
            if v in adversaries:
                # Ask adversary logic
                verdict = adversaries[v].process_packet("data" if priority == 0 else "voice")
                if not verdict:
                    logger.debug(f"Packet dropped by ADVERSARY {v} ({attack_type})")
                    success = False
                    if trust_model: trust_model.update_trust(v, False)
                    break
            
            # Normal Congestion/Reliability checks (simplified)
            # ...
            if trust_model: trust_model.update_trust(v, True)
            
        return success
    
    net_sim.simulate_packet = secure_simulate_packet

    # 2. Run with Intelligent Routing
    trust_model = TrustModel() # Should catch the attackers
    routing = IntelligentRouting(net_sim.graph, trust_model)
    
    stats = {'success': 0, 'total': 0}
    
    def traffic():
        for i in range(200):
            src, dst = random.sample(net_sim.nodes, 2)
            # Avoid picking adversaries as source/dest to keep stats clean
            if src in malicious_nodes or dst in malicious_nodes:
                continue
                
            path = routing.find_path(src, dst)
            if path:
                success = net_sim.simulate_packet(path, trust_model)
                stats['total'] += 1
                if success: stats['success'] += 1
            yield env.timeout(0.1)
            
    env.process(traffic())
    env.run()
    
    pdr = (stats['success'] / stats['total'] * 100) if stats['total'] else 0
    print(f"Result for {attack_type}: PDR = {pdr:.2f}%")
    return pdr

if __name__ == "__main__":
    # Test all 3 types
    run_attack_scenario("blackhole")
    run_attack_scenario("grayhole") # Should kill Data
    run_attack_scenario("on-off")   # Hardest to catch?
