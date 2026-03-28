#!/usr/bin/env python3
"""
Orcetra vs FLAML — STRICT 30s budget for BOTH.
Orcetra's 30s includes baseline phase.
Reuses the same 108 datasets from previous pilot runs.
"""
import json
import time
import sys
import os
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"


def load_previous_datasets():
    """Load dataset IDs from previous FLAML pilot runs."""
    seen = {}
    for fname in sorted(RESULTS_DIR.glob("flaml_pilot*.jsonl")):
        for line in open(fname):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    seen[r["dataset_id"]] = r
            except:
                pass
    return list(seen.values())


def run_flaml(X_train, X_test, y_train, y_test, task_type, budget_sec=30):
    from flaml import AutoML
    automl = AutoML()
    flaml_metric = "accuracy" if task_type == "classification" else "mse"
    
    automl.fit(
        X_train, y_train,
        time_budget=budget_sec,
        task=task_type,
        metric=flaml_metric,
        log_file_name="/dev/null",
        verbose=0,
        seed=42,
    )
    preds = automl.predict(X_test)
    return preds, str(automl.best_estimator)


def run_orcetra_strict(X_train, X_test, y_train, y_test, task_type, metric_name, budget_sec=30):
    """Run Orcetra with STRICT total budget (baseline + search combined)."""
    from orcetra.models.registry import get_baselines
    from orcetra.metrics.base import get_metric
    from orcetra.core.agent import RandomSearchAgent
    
    metric_fn = get_metric(metric_name)
    data_info = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "shape": (len(X_train) + len(X_test), X_train.shape[1]),
        "task_type": task_type, "n_features": X_train.shape[1],
    }
    
    # STRICT: total timer starts NOW (includes baselines)
    t_start = time.time()
    
    # Phase 1: Baselines (within budget)
    best_score = None
    best_model = None
    baseline_time = 0
    for model_name, model_fn in get_baselines(task_type).items():
        if time.time() - t_start >= budget_sec:
            break  # budget exhausted during baselines
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
    
    baseline_time = time.time() - t_start
    
    if best_score is None:
        return None, None, 0, baseline_time
    
    # Phase 2: Search with REMAINING budget
    remaining = budget_sec - (time.time() - t_start)
    if remaining <= 1:
        return best_score, best_model, 0, baseline_time
    
    agent = RandomSearchAgent(task_type=task_type)
    iteration = 0
    while time.time() - t_start < budget_sec:
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
    
    return best_score, best_model, iteration, baseline_time


def main():
    import openml
    
    prior_results = load_previous_datasets()
    print(f"Loaded {len(prior_results)} previous datasets to re-run")
    
    BUDGET = 30
    outfile = RESULTS_DIR / f"flaml_strict30s_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    
    flaml_wins = orcetra_wins = ties = success = 0
    
    for i, prior in enumerate(prior_results):
        did = prior["dataset_id"]
        name = prior.get("name", f"dataset_{did}")
        task_type = prior["task_type"]
        
        print(f"[{i+1}/{len(prior_results)}] {name} (id={did}, {task_type}) ... ", 
              end="", flush=True)
        
        result = {
            "dataset_id": did, "name": name, "task_type": task_type,
            "n_samples": prior.get("n_samples"), "n_features": prior.get("n_features"),
            "budget_sec": BUDGET, "timestamp": datetime.now().isoformat(),
            "experiment": "strict_30s",
        }
        
        try:
            ds = openml.datasets.get_dataset(did, download_data=True)
            X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)
            
            if X_df is None or y_series is None:
                result["status"] = "no_data"
                print("SKIP")
                with open(outfile, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue
            
            metric_name = "accuracy" if task_type == "classification" else "mse"
            
            if task_type == "classification":
                y_series = y_series.astype("category").cat.codes
            else:
                y_series = pd.to_numeric(y_series, errors="coerce")
            
            X_df = X_df.copy()
            for col in X_df.select_dtypes(include=["object", "category"]).columns:
                X_df[col] = X_df[col].astype("category").cat.codes
            num_cols = X_df.select_dtypes(include=[np.number]).columns
            X_df[num_cols] = X_df[num_cols].fillna(X_df[num_cols].median())
            X_df = X_df.dropna(axis=1, how="all")
            
            mask = y_series.notna()
            X_np = X_df.loc[mask].values.astype(np.float64)
            y_np = y_series.loc[mask].values.astype(np.float64)
            
            if len(X_np) < 50:
                result["status"] = "too_small"
                print("SKIP")
                with open(outfile, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, random_state=42
            )
            
            # FLAML — strict 30s
            t0 = time.time()
            flaml_preds, flaml_model = run_flaml(
                X_train, X_test, y_train, y_test, task_type, BUDGET
            )
            flaml_time = time.time() - t0
            
            if task_type == "classification":
                flaml_score = float(np.mean(flaml_preds == y_test))
            else:
                flaml_score = float(np.mean((flaml_preds - y_test) ** 2))
            
            result["flaml_score"] = round(flaml_score, 6)
            result["flaml_model"] = flaml_model
            result["flaml_time"] = round(flaml_time, 2)
            
            # Orcetra — STRICT 30s (baseline + search combined)
            t0 = time.time()
            orc_score, orc_model, orc_iters, bl_time = run_orcetra_strict(
                X_train, X_test, y_train, y_test, task_type, metric_name, BUDGET
            )
            orcetra_time = time.time() - t0
            
            if orc_score is None:
                result["status"] = "orcetra_failed"
                print("SKIP")
                with open(outfile, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue
            
            result["orcetra_score"] = round(float(orc_score), 6)
            result["orcetra_model"] = orc_model
            result["orcetra_time"] = round(orcetra_time, 2)
            result["orcetra_iters"] = orc_iters
            result["orcetra_baseline_time"] = round(bl_time, 2)
            result["metric"] = metric_name
            
            # Compare
            EPS = 1e-6
            if task_type == "classification":
                if orc_score > flaml_score + EPS: winner = "orcetra"
                elif flaml_score > orc_score + EPS: winner = "flaml"
                else: winner = "tie"
            else:
                if orc_score < flaml_score - EPS: winner = "orcetra"
                elif flaml_score < orc_score - EPS: winner = "flaml"
                else: winner = "tie"
            
            result["winner"] = winner
            result["status"] = "success"
            
            if winner == "orcetra": orcetra_wins += 1
            elif winner == "flaml": flaml_wins += 1
            else: ties += 1
            success += 1
            
            delta = orc_score - flaml_score if task_type == "classification" else flaml_score - orc_score
            print(f"F={flaml_score:.4f} O={orc_score:.4f} Δ={delta:+.4f} → {winner.upper()} "
                  f"[bl={bl_time:.1f}s] | F:{flaml_wins} O:{orcetra_wins} T:{ties}")
        
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]
            print(f"ERROR: {str(e)[:60]}")
        
        with open(outfile, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n{'='*60}")
    print(f"STRICT 30s RESULTS ({success} datasets)")
    print(f"{'='*60}")
    print(f"  Orcetra: {orcetra_wins} ({orcetra_wins/max(success,1)*100:.0f}%)")
    print(f"  FLAML:   {flaml_wins} ({flaml_wins/max(success,1)*100:.0f}%)")
    print(f"  Tie:     {ties} ({ties/max(success,1)*100:.0f}%)")
    print(f"\nResults: {outfile}")


if __name__ == "__main__":
    main()
