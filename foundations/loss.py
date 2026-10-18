import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss_array = y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)
        loss = -np.mean(loss_array)
        return round(loss, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        class_losses = y_true*np.log(y_pred)
        loss = -np.mean(np.sum(class_losses, axis=1))
        return round(loss, 4)
        
