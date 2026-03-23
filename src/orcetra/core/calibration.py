"""
Calibration module — the key insight that makes Orcetra faster than vanilla autoresearch.

Markets and models have systematic biases. Calibration corrects these biases
using historical data, giving the AutoResearch loop a warm start.
"""
import numpy as np
from typing import List, Tuple

def calibration_correct(predicted: float, cal_curve: List[Tuple[float, float]] = None) -> float:
    """
    Apply calibration correction based on learned bias curve.
    
    Default curve learned from 177 resolved Polymarket predictions:
    Markets systematically overprice events in the 20-50% range.
    """
    if cal_curve is None:
        cal_curve = [
            (0.05, 0.06), (0.15, 0.10), (0.25, 0.12),
            (0.35, 0.38), (0.45, 0.38), (0.55, 0.50),
            (0.65, 0.67), (0.75, 0.75), (0.85, 0.88), (0.95, 0.97),
        ]
    
    if predicted <= cal_curve[0][0]:
        return cal_curve[0][1]
    if predicted >= cal_curve[-1][0]:
        return cal_curve[-1][1]
    
    for i in range(len(cal_curve) - 1):
        x0, y0 = cal_curve[i]
        x1, y1 = cal_curve[i + 1]
        if x0 <= predicted <= x1:
            t = (predicted - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    
    return predicted

def learn_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> List[Tuple[float, float]]:
    """Learn a calibration curve from historical predictions."""
    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            curve.append((float(bins[i] + bins[i+1]) / 2, float(y_true[mask].mean())))
    return curve