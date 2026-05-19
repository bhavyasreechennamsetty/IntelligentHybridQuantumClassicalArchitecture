from qiskit import transpile
# Try importing Aer, else fallback
try:
    from qiskit_aer import Aer
    HAS_AER = True
except ImportError:
    HAS_AER = False

# Handle Qiskit 1.0 vs Pre-1.0 Primitives
try:
    from qiskit.primitives import StatevectorSampler as Sampler
except ImportError:
    try:
        from qiskit.primitives import Sampler
    except ImportError:
        # Last resort fallback or mock
        print("Warning: Could not import Sampler from qiskit.primitives")
        Sampler = None

from .circuits import QuantumKernelCircuits
import numpy as np
import time

class QuantumExecutor:
    """
    Handles the execution of quantum jobs on simulators (or hardware).
    """
    def __init__(self):
        if Sampler:
            self.sampler = Sampler()
        else:
             self.sampler = None
             
        if HAS_AER:
            self.backend = Aer.get_backend('qasm_simulator')
        else:
            self.backend = None # Will use Sampler primitive which doesn't need explicit backend for basic use

    def execute_task(self, data: np.ndarray, task_type: str = "pattern_recognition"):
        """
        Executes a quantum circuit based on the reduced data.
        
        Args:
            data: The reduced dimensionality data vector.
            task_type: Type of task to run.
            
        Returns:
            dict: Results of the execution (counts/probabilities).
        """
        start_time = time.time()
        n_qubits = len(data)
        
        # 1. Build Circuit
        # Encode data
        qc = QuantumKernelCircuits.feature_map_circuit(n_qubits, data)
        
        # Add task specific logic
        if task_type == "search_optimization":
            oracle = QuantumKernelCircuits.simple_search_oracle(n_qubits)
            qc.compose(oracle, inplace=True)
            
        qc.measure_all()
        
        # 2. Run
        if HAS_AER and self.backend:
            # Optimized for the simulator (Aer Legacy / V1)
            transpiled_qc = transpile(qc, self.backend)
            job = self.backend.run(transpiled_qc, shots=1024)
            result = job.result()
            counts = result.get_counts()
            backend_name = "aer_simulator"
        else:
            # ROI: Use Reference Sampler (Likely V2 in Qiskit 1.0+)
            # V2 input is list of pubs
            job = self.sampler.run([qc])
            result = job.result()
            
            # Try V2 Access pattern
            try:
                # Assuming measure_all() created 'meas' register
                # result[0] is PubResult, data is DataBin
                counts = result[0].data.meas.get_counts()
            except (AttributeError, KeyError, TypeError):
                # Fallback to V1 Access pattern (quasi_dists)
                if hasattr(result, 'quasi_dists'):
                    quasi_dists = result.quasi_dists[0]
                    counts = {bin(k)[2:].zfill(n_qubits): v * 1024 for k, v in quasi_dists.items()}
                else:
                    counts = {}
            
            backend_name = "reference_sampler"
        
        # --- ERROR MITIGATION (Naive) ---
        # Discard states with low counts (likely noise)
        total_shots = sum(counts.values())
        threshold = total_shots * 0.05 # 5% threshold
        mitigated_counts = {k: v for k, v in counts.items() if v > threshold}
        if not mitigated_counts:
            mitigated_counts = counts # Fallback if all look like noise
            
        exec_time = time.time() - start_time
        
        # Calculate a "result" score
        if mitigated_counts:
            most_freq_bitstring = max(mitigated_counts, key=mitigated_counts.get)
        else:
            most_freq_bitstring = "0" * n_qubits
        
        return {
            "status": "success",
            "backend": backend_name,
            "execution_time": exec_time,
            "top_result": most_freq_bitstring,
            "counts": mitigated_counts,
            "mitigation": "threshold_filter"
        }
