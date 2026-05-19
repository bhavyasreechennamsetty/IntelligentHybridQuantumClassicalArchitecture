import time
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    total_latency: float = 0.0
    quantum_count: int = 0
    classical_count: int = 0
    current_sampling_rate: float = 0.0
    current_dim: int = 0

class PerformanceTracker:
    """
    Tracks runtime metrics to feed into the orchestration layer.
    """
    def __init__(self):
        self.metrics = SystemMetrics()
        self.latencies = []

    def log_latency(self, latency: float):
        self.latencies.append(latency)
        # Keep window small for rolling average
        if len(self.latencies) > 50:
            self.latencies.pop(0)
        self.metrics.total_latency = sum(self.latencies) / len(self.latencies)

    def increment_quantum(self):
        self.metrics.quantum_count += 1

    def increment_classical(self):
        self.metrics.classical_count += 1
        
    def update_state(self, rate, dim):
        self.metrics.current_sampling_rate = rate
        self.metrics.current_dim = dim

    def get_summary(self):
        return self.metrics
