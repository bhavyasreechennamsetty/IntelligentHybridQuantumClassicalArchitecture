from metrics.performance import SystemMetrics
import numpy as np
from .rl_agent import QLearningAgent

class Monitor:
    """
    Aggregates metrics and presents system health.
    """
    def __init__(self, tracker):
        self.tracker = tracker

    def check_health(self):
        """
        Returns a health vector/status for the decision engine.
        """
        metrics = self.tracker.get_summary()
        avg_latency = metrics.total_latency
        
        # Simple heuristics for prototype
        load_status = "normal"
        if avg_latency > 0.1: # 100ms threshold
            load_status = "high_latency"
        elif avg_latency < 0.01:
            load_status = "low_latency"
            
        return load_status, metrics

class DecisionEngine:
    """
    The BRAIN of the system.
    Uses Reinforcement Learning to optimize the Classical/Quantum trade-off.
    """
    def __init__(self):
        self.agent = QLearningAgent()
        self.last_state = None
        self.last_action = None
        
    def decide_execution_path(self, data_packet: dict, health_status: str, latency_val: float) -> str:
        """
        Decide whether to process Classical or Quantum using RL.
        """
        data = data_packet['data']
        complexity = np.std(data)
        
        # 1. Observe Current State
        current_state = self.agent.get_state_key(latency_val, complexity)
        
        # 2. Learn from previous step if exists
        if self.last_state is not None:
            # Calculate Reward
            # Penalize High Latency, Reward Quantum usage if Complexity was High (Simulating 'Value')
            reward = 0
            if latency_val < 0.05:
                reward += 10 # Fast is good
            else:
                reward -= 10 # Slow is bad
                
            if self.last_action == 1: # Quantum
                # We assume Quantum is valuable for High Complexity
                # Recover state components (hacky but works for proto)
                prev_complexity_high = "C_HIGH" in self.last_state[1]
                if prev_complexity_high:
                    reward += 20 # Good use of Quantum
                else:
                    reward -= 5 # Waste of Quantum resources
            
            self.agent.learn(self.last_state, self.last_action, reward, current_state)
            
        # 3. Choose New Action
        action_idx = self.agent.choose_action(current_state) # 0 or 1
        
        self.last_state = current_state
        self.last_action = action_idx
        
        return "quantum" if action_idx == 1 else "classical"

    def adjust_parameters(self, health_status: str, current_rate: float, current_dim: int):
        """
        Feedback Loop: Adjust system parameters based on health.
        """
        new_rate = current_rate
        new_dim = current_dim
        
        if health_status == "high_latency":
            # Throttle down
            new_rate = max(1.0, current_rate * 0.9)
            new_dim = max(2, current_dim - 1)
        elif health_status == "low_latency":
            # Scale up
            new_rate = min(50.0, current_rate * 1.1)
            new_dim = min(8, current_dim + 1) # Max 8 qubits for this demo
            
        return new_rate, new_dim
