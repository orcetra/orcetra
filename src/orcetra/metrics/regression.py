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

class RMSLE(Metric):
    name = "rmsle"
    direction = "minimize"
    def compute(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.clip(np.array(y_pred, dtype=np.float64), 0, None)
        return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))

class R2(Metric):
    name = "r2"
    direction = "maximize"
    def compute(self, y_true, y_pred):
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))

class RMSLE(Metric):
    """Root Mean Squared Logarithmic Error - ideal for right-skewed positive targets."""
    name = "rmsle"
    direction = "minimize"
    def compute(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 0, None)  # Clip negative predictions to 0
        return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))