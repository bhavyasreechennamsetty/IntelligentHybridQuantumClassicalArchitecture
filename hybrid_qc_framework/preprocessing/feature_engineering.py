import numpy as np

class FeatureEngineer:
    """
    Extracts relevant features from the cleaned stream.
    """
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Augment data with statistical features.
        For this prototype, we pass the cleaned signal and append basic stats.
        """
        mean_val = np.mean(data)
        std_val = np.std(data)
        # We preserve the original signal shape but could append stats
        # For the architecture's sake, we just return the cleaned data 
        # as the 'features' for the reduction step, maybe normalized.
        
        # Normalization to [0, 1] range (useful for quantum embedding later)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 1e-6:
            normalized = (data - min_val) / (max_val - min_val)
        else:
            normalized = data
            
        return normalized
