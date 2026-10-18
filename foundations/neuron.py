import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = (x @ w) + b

        if activation == "sigmoid":
            output = 1/(1 + np.exp(-z))
        elif activation == "relu":
            output = max(0.0, z)

        return round(float(output), 5)
        
