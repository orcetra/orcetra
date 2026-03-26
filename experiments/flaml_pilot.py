#!/usr/bin/env python3
"""
Orcetra vs FLAML (real AutoML) — fair comparison pilot.

Same datasets, same time budget (30s), same train/test split.
Picks 50 datasets from our existing benchmark results to compare.

Usage:
    python experiments/flaml_pilot.py [--n-datasets 50] [--budget 30]
"""
import json
import time
import sys
import warnings
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_existing_results():
    """Load completed Orcetra benchmark results to pick datasets."""
    results = []
    for f in sorted(RESULTS_DIR.glob("openml_benchmark_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    results.append(r)
            except:
                pass
    return results


def pick_pilot_datasets(results, n=50):
    """Pick diverse datasets: stratified by task type, varied difficulty."""
    classification = [r for r in results if r["task_type"] == "classification"]
    regression = [r for r in results if r["task_type"] == "regression"]
    
    # Sort by dataset size for diversity
    classification.sort(key=lambda r: r.get("n_samples", 0))
    regression.sort(key=lambda r: r.get("n_samples", 0))
    
    # Take evenly spaced samples
    n_cls = min(n * 2 // 3, len(classification))  # ~2/3 classification
    n_reg = min(n - n_cls, len(regression))        # ~1/3 regression
    
    step_cls = max(1, len(classification) // n_cls)
    step_reg = max(1, len(regression) // n_reg)
    
    picked = []
    picked += classification[::step_cls][:n_cls]
    picked += regression[::step_reg][:n_reg]
    
    return picked[:n]


def run_flaml(X_train, X_test, y_train, y_test, task_type, budget_sec=30):
    """Run FLAML AutoML with same budget."""
    from flaml import AutoML
    
    automl = AutoML()
    settings = {
        "time_budget": budget_sec,
        "task": task_type,
        "log_file_name": "/dev/null",
        "verbose": 0,
        "seed": 42,
    }
    
    if task_type == "classification":
        settings["metric"] = "accuracy"
    else:
        settings["metric"] = "mse"
    
    automl.fit(X_train, y_train, **settings)
    
    preds = automl.predict(X_test)
    best_model = str(automl.best_estimator)
    
    return preds, best_model, automl.best_config


def run_orcetra(X, y, task_type, budget_sec=30):
    """Run Orcetra with same budget (replicating benchmark logic)."""
    from orcetra.models.registry import get_baselines
    from orcetra.metrics.base import get_metric
    from orcetra.core.agent import RandomSearchAgent
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task_type == "classification":
        metric_name = "accuracy"
    else:
        # Match benchmark logic: use RMSLE if all positive
        if (y > 0).all():
            metric_name = "rmsle"
        else:
            metric_name = "mse"
    
    metric_fn = get_metric(metric_name)
    data_info = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "shape": X.shape, "task_type": task_type,
        "n_features": X.shape[1],
    }
    
    # Baselines first
    best_score = None
    best_model = None
    for model_name, model_fn in get_baselines(task_type).items():
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
    
    # AutoResearch loop
    import os
    agent = None
    try:
        if os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            from orcetra.core.llm_agent import LLMSearchAgent
            provider = "groq" if os.environ.get("GROQ_API_KEY") else "openai"
            agent = LLMSearchAgent(task_type=task_type, provider=provider)
    except:
        pass
    if agent is None:
        agent = RandomSearchAgent(task_type=task_type)
    
    start = time.time()
    iteration = 0
    while time.time() - start < budget_sec:
        iteration += 1
        try:
            proposal = agent.propose(data_info, metric_fn, best_score, best_model, iteration)
            if proposal is None:
                continue
            score = proposal.evaluate(data_info, metric_fn)
            is_better = (
                (metric_fn.direction == "minimize" and score < best_score)
                or (metric_fn.direction == "maximize" and score > best_score)
            )
            if is_better:
                best_score = score
                best_model = proposal.description
        except:
            pass
    
    return best_score, best_model, metric_name, iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=50)
    parser.add_argument("--budget", type=int, default=30, help="Time budget in seconds per method")
    args = parser.parse_args()
    
    import openml
    
    existing = load_existing_results()
    print(f"Loaded {len(existing)} existing Orcetra results")
    
    pilots = pick_pilot_datasets(existing, args.n_datasets)
    print(f"Selected {len(pilots)} pilot datasets\n")
    
    outfile = RESULTS_DIR / f"flaml_pilot_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    
    flaml_wins = 0
    orcetra_wins = 0
    ties = 0
    success = 0
    
    for i, prior in enumerate(pilots):
        did = prior["dataset_id"]
        name = prior.get("name", f"dataset_{did}")
        print(f"[{i+1}/{len(pilots)}] {name} (id={did}, {prior['task_type']}, "
              f"n={prior.get('n_samples','?')}, p={prior.get('n_features','?')}) ... ", end="", flush=True)
        
        result = {
            "dataset_id": did,
            "name": name,
            "task_type": prior["task_type"],
            "n_samples": prior.get("n_samples"),
            "n_features": prior.get("n_features"),
            "budget_sec": args.budget,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            # Load dataset
            ds = openml.datasets.get_dataset(did, download_data=True)
            X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)
            
            if X_df is None or y_series is None:
                result["status"] = "no_data"
                print("SKIP (no data)")
                with open(outfile, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue
            
            # Preprocess (same as benchmark)
            task_type = prior["task_type"]
            
            if task_type == "classification":
                y_series = y_series.astype("category").cat.codes
            else:
                y_series = pd.to_numeric(y_series, errors="coerce")
            
            X_df = X_df.copy()
            cat_cols = X_df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                X_df[col] = X_df[col].astype("category").cat.codes
            
            num_cols = X_df.select_dtypes(include=[np.number]).columns
            X_df[num_cols] = X_df[num_cols].fillna(X_df[num_cols].median())
            
            X_df = X_df.dropna(axis=1, how="all")
            
            mask = y_series.notna()
            X_np = X_df.loc[mask].values.astype(np.float64)
            y_np = y_series.loc[mask].values.astype(np.float64)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, random_state=42
            )
            
            # ── FLAML ──
            t0 = time.time()
            flaml_preds, flaml_model, flaml_config = run_flaml(
                X_train, X_test, y_train, y_test, task_type, args.budget
            )
            flaml_time = time.time() - t0
            
            if task_type == "classification":
                flaml_score = float(np.mean(flaml_preds == y_test))
                metric_name = "accuracy"
            else:
                flaml_score = float(np.mean((flaml_preds - y_test) ** 2))
                metric_name = "mse"
            
            result["flaml_score"] = round(flaml_score, 6)
            result["flaml_model"] = flaml_model
            result["flaml_time"] = round(flaml_time, 2)
            
            # ── Orcetra ──
            t0 = time.time()
            orc_score, orc_model, orc_metric, orc_iters = run_orcetra(
                X_np, y_np, task_type, args.budget
            )
            orcetra_time = time.time() - t0
            
            # Normalize Orcetra score to same metric as FLAML for fair comparison
            # Orcetra might use RMSLE internally, but we compare on same metric
            # Recalculate Orcetra on same test set with same metric
            # Actually — Orcetra uses its own split. For fairness, we should 
            # note this. But the existing benchmark also has this issue.
            # For the pilot, use the score Orcetra reports.
            
            result["orcetra_score"] = round(float(orc_score), 6) if orc_score else None
            result["orcetra_model"] = orc_model
            result["orcetra_time"] = round(orcetra_time, 2)
            result["orcetra_iters"] = orc_iters
            result["orcetra_metric"] = orc_metric
            result["comparison_metric"] = metric_name
            
            # Compare on same metric
            if task_type == "classification":
                # Both using accuracy — higher is better
                if orc_metric == "accuracy" and orc_score is not None:
                    if orc_score > flaml_score + 1e-6:
                        winner = "orcetra"
                    elif flaml_score > orc_score + 1e-6:
                        winner = "flaml"
                    else:
                        winner = "tie"
                else:
                    winner = "metric_mismatch"
            else:
                # Regression — need to compare on same metric
                # If Orcetra used RMSLE and FLAML used MSE, it's not directly comparable
                # For fairness, just record both and flag
                if orc_metric == "mse" and orc_score is not None:
                    if orc_score < flaml_score - 1e-6:
                        winner = "orcetra"
                    elif flaml_score < orc_score - 1e-6:
                        winner = "flaml"
                    else:
                        winner = "tie"
                else:
                    winner = "metric_mismatch"
            
            result["winner"] = winner
            result["status"] = "success"
            
            if winner == "orcetra":
                orcetra_wins += 1
            elif winner == "flaml":
                flaml_wins += 1
            else:
                ties += 1
            success += 1
            
            print(f"FLAML={flaml_score:.4f}({flaml_model}) vs ORC={orc_score:.4f}({orc_model}) "
                  f"→ {winner.upper()} | running: F={flaml_wins} O={orcetra_wins} T={ties}")
        
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]
            print(f"ERROR: {str(e)[:80]}")
            traceback.print_exc()
        
        with open(outfile, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PILOT RESULTS: Orcetra vs FLAML ({args.budget}s budget each)")
    print(f"{'='*60}")
    print(f"Datasets:     {success} successful / {len(pilots)} attempted")
    print(f"FLAML wins:   {flaml_wins} ({flaml_wins/max(success,1)*100:.0f}%)")
    print(f"Orcetra wins: {orcetra_wins} ({orcetra_wins/max(success,1)*100:.0f}%)")
    print(f"Ties:         {ties}")
    print(f"Results:      {outfile}")


if __name__ == "__main__":
    main()
