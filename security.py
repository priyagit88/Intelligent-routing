import random
import logging

logger = logging.getLogger("Security")

class Adversary:
    def __init__(self, node_id, attack_type="blackhole"):
        self.node_id = node_id
        self.attack_type = attack_type
        self.active = True
        self.state = "good" # For On-Off attacks
        
    def process_packet(self, packet_type="data"):
        """
        Returns True if packet is forwarded, False if dropped.
        """
        if not self.active:
            return True # Behaves normally if attack is inactive
            
        if self.attack_type == "blackhole":
            return False # Drop everything
            
        elif self.attack_type == "grayhole":
            # Selectively drop 'data' packets, forward 'voice' to stay under radar?
            # Or drop specific % to annoy but not trigger instant ban
            if packet_type == "data":
                return False
            return True
            
        elif self.attack_type == "on-off":
            if self.state == "bad":
                return False
            return True
            
        return True

    def update_behavior(self, env):
        """For dynamic attacks (On-Off)"""
        while True:
            if self.attack_type == "on-off":
                # Behave good for 20s to build trust
                self.state = "good"
                logger.debug(f"Adversary {self.node_id}: Switching to GOOD state")
                yield env.timeout(20)
                
                # Attack for 5s
                self.state = "bad"
                logger.debug(f"Adversary {self.node_id}: Switching to BAD state")
                yield env.timeout(5)
            else:
                yield env.timeout(100) # Static attacks don't change
