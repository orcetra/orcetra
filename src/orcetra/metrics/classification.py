"""Classification metrics."""
from .base import Metric

class Accuracy(Metric):
    name = "accuracy"
    direction = "maximize"
    def compute(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y_true, y_pred))

class F1(Metric):
    name = "f1"
    direction = "maximize"
    def compute(self, y_true, y_pred):
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="weighted"))