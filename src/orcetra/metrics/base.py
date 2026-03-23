"""Metric interface."""
from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def direction(self) -> str: ...  # "minimize" or "maximize"
    
    @abstractmethod
    def compute(self, y_true, y_pred) -> float: ...

def get_metric(name: str) -> Metric:
    from .regression import MSE, MAE, R2
    from .classification import Accuracy, F1
    from .brier import BrierScore
    
    metrics = {
        "mse": MSE(), "mae": MAE(), "r2": R2(),
        "accuracy": Accuracy(), "f1": F1(),
        "brier": BrierScore(),
    }
    if name not in metrics:
        raise ValueError(f"Unknown metric: {name}. Available: {list(metrics.keys())}")
    return metrics[name]