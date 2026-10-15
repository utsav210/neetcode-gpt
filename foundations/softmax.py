import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted_z = z - np.max(z)
        exp_z = np.exp(shifted_z)
        probabilities = exp_z / np.sum(exp_z)
        return np.round(probabilities, 4)
        
