from qiskit import QuantumCircuit
import numpy as np

class QuantumKernelCircuits:
    """
    Provides NISQ-friendly circuits.
    Focus on shallow depth and parameterized options.
    """
    
    @staticmethod
    def feature_map_circuit(n_qubits: int, data_params: np.ndarray) -> QuantumCircuit:
        """
        Creates a data encoding circuit (Angle Encoding).
        Shallow depth: O(1) layers of rotations.
        """
        qc = QuantumCircuit(n_qubits)
        # Angle encoding: Ry rotation by data values * pi
        for i in range(min(n_qubits, len(data_params))):
            qc.ry(data_params[i] * np.pi, i)
        return qc

    @staticmethod
    def simple_search_oracle(n_qubits: int) -> QuantumCircuit:
        """
        A very simplified oracle for demonstration.
        In a real Grover's search, this marks a specific state.
        Here we just add some entanglement to simulate 'work'.
        """
        qc = QuantumCircuit(n_qubits)
        # Create superposition
        qc.h(range(n_qubits)) 
        # "Oracle" / Processing - some CZ gates for entanglement
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qc.cz(i, i+1)
        return qc
