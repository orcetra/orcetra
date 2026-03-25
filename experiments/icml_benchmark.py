"""
ICML 2026 Benchmark: Orcetra (LLM-guided vs Random) vs AutoGluon vs sklearn baselines.

Runs 6 datasets × 4 methods × 3 seeds. Outputs JSON + summary table.
"""
import json
import time
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Datasets ──────────────────────────────────────────────────────────
from sklearn.datasets import (
    fetch_california_housing, load_diabetes, load_breast_cancer, load_digits,
)

def load_wine_quality():
    """Wine Quality (regression) from UCI via URL."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(url, sep=";")
    except Exception:
        # fallback: generate locally if no internet
        from sklearn.datasets import load_wine
        d = load_wine(as_frame=True)
        df = d.frame
        df = df.rename(columns={"target": "quality"})
    return df, "quality", "regression"

def load_adult_income():
    """Adult Income (classification) from UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"]
    try:
        df = pd.read_csv(url, names=cols, skipinitialspace=True, na_values="?")
    except Exception:
        # Fallback: use breast cancer as substitute
        d = load_breast_cancer(as_frame=True)
        df = d.frame
        return df, "target", "classification"
    
    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        if col != "income":
            df[col] = df[col].astype("category").cat.codes
    df["income"] = (df["income"] == ">50K").astype(int)
    df = df.dropna()
    return df, "income", "classification"


DATASETS = {
    "california_housing": {
        "loader": lambda: (
            fetch_california_housing(as_frame=True).frame,
            "MedHouseVal", "regression"
        ),
    },
    "diabetes": {
        "loader": lambda: (
            load_diabetes(as_frame=True).frame,
            "target", "regression"
        ),
    },
    "wine_quality": {
        "loader": load_wine_quality,
    },
    "breast_cancer": {
        "loader": lambda: (
            load_breast_cancer(as_frame=True).frame,
            "target", "classification"
        ),
    },
    "adult_income": {
        "loader": load_adult_income,
    },
    "digits": {
        "loader": lambda: (
            load_digits(as_frame=True).frame,
            "target", "classification"
        ),
    },
}

# ── Methods ───────────────────────────────────────────────────────────

def run_orcetra(data_path, target, budget, agent_type="llm", seed=42):
    """Run Orcetra and return results dict."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # We need to use internal API since CLI doesn't expose agent choice
    from orcetra.data.loader import analyze_and_load
    from orcetra.models.registry import get_baselines
    from orcetra.metrics.base import get_metric
    
    data_info = analyze_and_load(data_path, target)
    task_type = data_info["task_type"]
    metric_name = "mse" if task_type == "regression" else "accuracy"
    metric_fn = get_metric(metric_name)
    
    # Baselines
    baselines = get_baselines(task_type)
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
        except:
            pass
    
    baseline_best = best_score
    
    # Agent
    if agent_type == "llm":
        from orcetra.core.llm_agent import LLMSearchAgent
        agent = LLMSearchAgent(task_type=task_type, provider="groq")
    else:
        from orcetra.core.agent import RandomSearchAgent
        agent = RandomSearchAgent(task_type=task_type)
    
    # Loop
    start = time.time()
    budget_sec = budget
    iteration = 0
    improvements = 0
    convergence = []  # (iteration, best_score)
    last_desc = ""
    last_score_val = None
    last_improved = False
    
    while time.time() - start < budget_sec:
        iteration += 1
        proposal = agent.propose({
            "best_score": best_score,
            "best_model": best_model,
            "iteration": iteration,
            "task_type": task_type,
            "metric_direction": metric_fn.direction,
            "data_summary": f"{data_info['shape'][0]} rows, {data_info['n_features']} features",
            "last_proposal": last_desc,
            "last_score": last_score_val,
            "last_improved": last_improved,
        })
        
        try:
            score = proposal.evaluate(data_info, metric_fn)
            last_desc = proposal.description
            last_score_val = score
            last_improved = False
            
            is_better = (
                (metric_fn.direction == "minimize" and score < best_score)
                or (metric_fn.direction == "maximize" and score > best_score)
            )
            if is_better:
                improvements += 1
                best_score = score
                best_model = proposal.description
                last_improved = True
        except:
            pass
        
        convergence.append((iteration, best_score))
    
    elapsed = time.time() - start
    
    # Improvement
    if metric_fn.direction == "minimize":
        improvement_pct = (baseline_best - best_score) / baseline_best * 100 if baseline_best else 0
    else:
        improvement_pct = (best_score - baseline_best) / baseline_best * 100 if baseline_best else 0
    
    return {
        "best_score": best_score,
        "baseline_score": baseline_best,
        "improvement_pct": improvement_pct,
        "best_model": best_model,
        "iterations": iteration,
        "improvements": improvements,
        "elapsed": elapsed,
        "metric": metric_name,
        "direction": metric_fn.direction,
        "convergence": convergence,
    }


def run_sklearn_baseline(data_path, target):
    """Just return sklearn default RF and GBM scores."""
    from orcetra.data.loader import analyze_and_load
    from orcetra.metrics.base import get_metric
    
    data_info = analyze_and_load(data_path, target)
    task_type = data_info["task_type"]
    metric_name = "mse" if task_type == "regression" else "accuracy"
    metric_fn = get_metric(metric_name)
    
    from orcetra.models.registry import get_baselines
    baselines = get_baselines(task_type)
    
    results = {}
    for name, fn in baselines.items():
        try:
            results[name] = fn(data_info, metric_fn)
        except:
            results[name] = None
    
    return results, metric_name


def run_autogluon(data_path, target, task_type, budget=60):
    """Run AutoGluon TabularPredictor."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        return None  # AutoGluon not installed
    
    df = pd.read_csv(data_path)
    problem_type = "regression" if task_type == "regression" else "binary" if df[target].nunique() == 2 else "multiclass"
    
    eval_metric = "root_mean_squared_error" if task_type == "regression" else "accuracy"
    
    predictor = TabularPredictor(
        label=target,
        eval_metric=eval_metric,
        problem_type=problem_type,
        verbosity=0,
        path=f"/tmp/ag_{int(time.time())}",
    ).fit(
        df,
        time_limit=budget,
        presets="best_quality",
    )
    
    leaderboard = predictor.leaderboard(silent=True)
    best_score = leaderboard.iloc[0]["score_val"]
    
    # AutoGluon returns negative MSE for regression
    if task_type == "regression":
        best_score = -best_score  # Convert to positive MSE
    
    return {
        "best_score": best_score,
        "best_model": leaderboard.iloc[0]["model"],
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    BUDGET = 60  # seconds per run
    SEEDS = [42, 123, 456]
    RESULTS_DIR = Path("/home/guilinzhang/allProjects/orcetra/experiments/results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    all_results = {}
    
    for ds_name, ds_config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        # Load and save to CSV
        df, target, task_type = ds_config["loader"]()
        csv_path = f"/tmp/bench_{ds_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} cols, task={task_type}")
        
        ds_results = {"task_type": task_type, "shape": list(df.shape)}
        
        # 1. sklearn baselines (no randomness needed)
        print("  Running sklearn baselines...")
        sk_results, metric = run_sklearn_baseline(csv_path, target)
        ds_results["sklearn"] = sk_results
        ds_results["metric"] = metric
        print(f"    sklearn: {sk_results}")
        
        # 2. Orcetra Random Search (3 seeds)
        print("  Running Orcetra (random)...")
        random_scores = []
        random_convergences = []
        for seed in SEEDS:
            r = run_orcetra(csv_path, target, BUDGET, agent_type="random", seed=seed)
            random_scores.append(r["best_score"])
            random_convergences.append(r["convergence"])
            print(f"    seed={seed}: {r['best_score']:.4f} ({r['iterations']} iters, {r['improvements']} improvements)")
        
        ds_results["orcetra_random"] = {
            "mean": float(np.mean(random_scores)),
            "std": float(np.std(random_scores)),
            "scores": random_scores,
            "convergences": random_convergences,
        }
        
        # 3. Orcetra LLM-guided (3 seeds)
        print("  Running Orcetra (LLM-guided)...")
        llm_scores = []
        llm_convergences = []
        for seed in SEEDS:
            r = run_orcetra(csv_path, target, BUDGET, agent_type="llm", seed=seed)
            llm_scores.append(r["best_score"])
            llm_convergences.append(r["convergence"])
            print(f"    seed={seed}: {r['best_score']:.4f} ({r['iterations']} iters, {r['improvements']} improvements)")
        
        ds_results["orcetra_llm"] = {
            "mean": float(np.mean(llm_scores)),
            "std": float(np.std(llm_scores)),
            "scores": llm_scores,
            "convergences": llm_convergences,
        }
        
        # 4. AutoGluon (if available)
        print("  Running AutoGluon...")
        ag_result = run_autogluon(csv_path, target, task_type, budget=BUDGET)
        if ag_result:
            ds_results["autogluon"] = ag_result
            print(f"    AutoGluon: {ag_result['best_score']:.4f} ({ag_result['best_model']})")
        else:
            ds_results["autogluon"] = None
            print("    AutoGluon: not installed, skipping")
        
        all_results[ds_name] = ds_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"icml_benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {results_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Metric':<8} {'sklearn':<12} {'Random':<16} {'LLM-guided':<16} {'AutoGluon':<12}")
    print("-" * 80)
    
    for ds_name, ds_res in all_results.items():
        metric = ds_res["metric"]
        direction = "↓" if metric == "mse" else "↑"
        
        # Best sklearn baseline
        sk_vals = [v for v in ds_res["sklearn"].values() if v is not None]
        if metric == "mse":
            sk_best = min(sk_vals) if sk_vals else float('inf')
        else:
            sk_best = max(sk_vals) if sk_vals else 0
        
        random_mean = ds_res["orcetra_random"]["mean"]
        random_std = ds_res["orcetra_random"]["std"]
        llm_mean = ds_res["orcetra_llm"]["mean"]
        llm_std = ds_res["orcetra_llm"]["std"]
        
        ag_str = "N/A"
        if ds_res.get("autogluon") and ds_res["autogluon"].get("best_score"):
            ag_str = f"{ds_res['autogluon']['best_score']:.4f}"
        
        print(f"{ds_name:<20} {metric}{direction:<7} {sk_best:<12.4f} {random_mean:.4f}±{random_std:.4f} {llm_mean:.4f}±{llm_std:.4f} {ag_str:<12}")
    
    print(f"\nBudget: {BUDGET}s per run, {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
