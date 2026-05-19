import numpy as np
from scipy.ndimage import gaussian_filter1d

class NoiseFilter:
    """
    Responsible for cleaning raw input data using classical signal processing techniques.
    """
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to the input vector.
        """
        # Simple Gaussian filter to smooth out high-freq noise
        return gaussian_filter1d(data, sigma=self.sigma)
