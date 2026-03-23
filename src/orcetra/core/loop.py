"""
AutoResearch prediction loop.

Inspired by Karpathy's autoresearch: an AI agent iteratively improves
a prediction pipeline by proposing modifications, testing them, and
keeping improvements.
"""
import time
import pandas as pd
from typing import Optional
from rich.console import Console
from rich.progress import Progress

from ..data.loader import analyze_and_load
from ..models.registry import get_baselines
from ..metrics.base import get_metric

console = Console()

def run_prediction(
    data_path: str,
    target: str,
    budget: str = "10min",
    metric: str = "auto",
) -> dict:
    """
    Main entry point for automated prediction.
    
    1. Load and analyze data
    2. Run baseline models
    3. AutoResearch loop: iteratively improve
    4. Return best result
    """
    budget_seconds = parse_budget(budget)
    start_time = time.time()
    
    # Step 1: Analyze data
    console.print("[bold]Step 1:[/bold] Analyzing data...")
    data_info = analyze_and_load(data_path, target)
    console.print(f"  Shape: {data_info['shape']}")
    console.print(f"  Task: {data_info['task_type']}")
    console.print(f"  Features: {data_info['n_features']}")
    
    # Step 2: Select metric
    if metric == "auto":
        metric = "mse" if data_info["task_type"] == "regression" else "accuracy"
    metric_fn = get_metric(metric)
    console.print(f"  Metric: {metric_fn.name} ({metric_fn.direction})")
    
    # Step 3: Run baselines
    console.print("\n[bold]Step 2:[/bold] Running baselines...")
    baselines = get_baselines(data_info["task_type"])
    best_score = None
    best_model = None
    
    for model_name, model_fn in baselines.items():
        try:
            score = model_fn(data_info, metric_fn)
            is_better = (
                best_score is None
                or (metric_fn.direction == "minimize" and score < best_score)
                or (metric_fn.direction == "maximize" and score > best_score)
            )
            if is_better:
                best_score = score
                best_model = model_name
            console.print(f"  {model_name}: {score:.4f} {'⭐' if is_better else ''}")
        except Exception as e:
            console.print(f"  {model_name}: [red]failed ({e})[/red]")
    
    # Step 4: AutoResearch loop (iterate while budget remains)
    console.print(f"\n[bold]Step 3:[/bold] AutoResearch loop (budget: {budget})...")
    iteration = 0
    
    while time.time() - start_time < budget_seconds:
        iteration += 1
        # TODO: AI agent proposes improvement
        # For now, try hyperparameter variations
        elapsed = time.time() - start_time
        if elapsed > budget_seconds:
            break
    
    elapsed = time.time() - start_time
    
    return {
        "best_model": best_model,
        "best_score": best_score,
        "metric_name": metric_fn.name,
        "iterations": iteration,
        "elapsed": elapsed,
        "task_type": data_info["task_type"],
    }

def parse_budget(budget: str) -> float:
    """Parse budget string like '10min', '1h', '30s' to seconds."""
    budget = budget.strip().lower()
    if budget.endswith("min"):
        return float(budget[:-3]) * 60
    elif budget.endswith("h"):
        return float(budget[:-1]) * 3600
    elif budget.endswith("s"):
        return float(budget[:-1])
    else:
        return float(budget) * 60  # default to minutes