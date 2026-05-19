import time
import numpy as np
import random
from typing import Generator, Dict, Any

class StreamSimulator:
    """
    Simulates a high-velocity data stream for the hybrid system.
    Generates synthetic data points with tunable noise and features.
    """
    def __init__(self, data_rate_hz: float = 10.0, n_features: int = 50):
        """
        Initialize the stream simulator.
        
        Args:
            data_rate_hz: Target data generation rate in Hz.
            n_features: Number of raw features to generate.
        """
        self.data_rate_hz = data_rate_hz
        self.n_features = n_features
        self.running = False

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields data packets indefinitely.
        
        Returns:
            Generator yielding dictionaries with 'data' and 'timestamp'.
        """
        self.running = True
        interval = 1.0 / self.data_rate_hz
        
        while self.running:
            start_time = time.time()
            
            # Generate synthetic data: specific pattern + random noise
            # Base signal: sine wave pattern across features
            t = time.time()
            base_signal = np.sin(np.linspace(0, 2 * np.pi, self.n_features) + t)
            noise = np.random.normal(0, 0.2, self.n_features) # Gaussian noise
            
            raw_data = base_signal + noise
            
            packet = {
                "timestamp": t,
                "data": raw_data,
                "metadata": {
                    "source": "simulated_sensor_array",
                    "status": "raw"
                }
            }
            
            yield packet
            
            # Rate limiting to match data_rate_hz
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False
