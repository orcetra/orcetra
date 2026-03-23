"""Brier score metric for probabilistic predictions."""
import numpy as np
from .base import Metric

class BrierScore(Metric):
    name = "brier"
    direction = "minimize"
    def compute(self, y_true, y_pred):
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))