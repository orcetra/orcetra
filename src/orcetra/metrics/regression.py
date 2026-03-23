"""Regression metrics."""
import numpy as np
from .base import Metric

class MSE(Metric):
    name = "mse"
    direction = "minimize"
    def compute(self, y_true, y_pred):
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

class MAE(Metric):
    name = "mae"
    direction = "minimize"
    def compute(self, y_true, y_pred):
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

class R2(Metric):
    name = "r2"
    direction = "maximize"
    def compute(self, y_true, y_pred):
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))