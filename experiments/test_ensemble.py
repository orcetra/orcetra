#!/usr/bin/env python3
"""Smoke test for weighted ensemble."""
import numpy as np
from orcetra.core.loop import StrategyCache
from orcetra.models.ensemble import try_ensemble
from orcetra.core.agent import Proposal
from orcetra.metrics.base import get_metric
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from rich.console import Console

np.random.seed(42)
X = np.random.randn(200, 5)
y = X @ np.array([1, 2, 3, 0.5, -1]) + np.random.randn(200) * 0.5

data_info = {
    "X_train": X[:160], "X_test": X[160:],
    "y_train": y[:160], "y_test": y[160:],
    "task_type": "regression",
}
metric_fn = get_metric("mse")

cache = StrategyCache()
models = [
    (RandomForestRegressor(n_estimators=100, random_state=42), "RF"),
    (GradientBoostingRegressor(n_estimators=100, random_state=42), "GBM"),
    (ExtraTreesRegressor(n_estimators=100, random_state=42), "ET"),
]
for model, name in models:
    p = Proposal(description=name, rationale="test", model=model)
    score = p.evaluate(data_info, metric_fn)
    cache.record(p, score)
    print(f"  {name}: MSE={score:.4f}")

console = Console()
result = try_ensemble(cache, data_info, metric_fn, console)
if result:
    print(f"  Ensemble: MSE={result[0]:.4f} -- {result[1]}")
else:
    print("  Ensemble: None")
