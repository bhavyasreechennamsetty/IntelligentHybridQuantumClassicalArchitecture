import numpy as np
from sklearn.decomposition import IncrementalPCA

class DimensionalityReducer:
    """
    Reduces the feature space to a size compatible with NISQ devices.
    Uses Incremental PCA for true online learning on streams.
    """
    def __init__(self, target_dim: int = 4):
        """
        Args:
            target_dim: Number of qubits available or target features.
        """
        self.target_dim = target_dim
        self.ipca = IncrementalPCA(n_components=target_dim)
        self.is_fitted = False
        self.buffer = []
        self.buffer_size = 10 # Need batch > n_components

    def learn(self, data_vector: np.ndarray):
        """
        Online learning step. Buffers data until enough for a partial_fit.
        """
        self.buffer.append(data_vector)
        if len(self.buffer) >= self.buffer_size:
            batch = np.array(self.buffer)
            self.ipca.partial_fit(batch)
            self.buffer = [] # Clear buffer
            self.is_fitted = True

    def reduce(self, data_vector: np.ndarray) -> np.ndarray:
        """
        Projects high-dim vector to low-dim Quantum-ready vector.
        """
        # If not yet fitted, return truncated raw data or random projection
        if not self.is_fitted:
            # Just return first N features normalized as fallback until trained
            return data_vector[:self.target_dim]
            
        reshaped = data_vector.reshape(1, -1)
        reduced = self.ipca.transform(reshaped)
        return reduced.flatten()
        
    def update_target_dim(self, new_dim: int):
        """
        Dynamic adjustment. Resetting model required for sklearn IPCA if dim changes.
        """
        if new_dim != self.target_dim:
            self.target_dim = new_dim
            # We must reset the learner if dimensions change, 
            # as the projection matrix shape changes.
            self.ipca = IncrementalPCA(n_components=new_dim)
            self.is_fitted = False
            self.buffer = []
