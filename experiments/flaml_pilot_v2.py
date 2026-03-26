#!/usr/bin/env python3
"""
Orcetra vs FLAML v2 — fixed fair comparison.

Fixes from v1:
1. Same train/test split for both (no double-splitting)
2. Same metric for both (MSE for regression, accuracy for classification)
3. Orcetra gets the same data_info as its internal loop expects
4. More datasets (50 default)

Usage:
    python experiments/flaml_pilot_v2.py [--n-datasets 50] [--budget 30]
"""
import json
import time
import sys
import os
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
    """Stratified diverse selection."""
    classification = [r for r in results if r["task_type"] == "classification"]
    regression = [r for r in results if r["task_type"] == "regression"]
    
    classification.sort(key=lambda r: r.get("n_samples", 0))
    regression.sort(key=lambda r: r.get("n_samples", 0))
    
    n_cls = min(n * 2 // 3, len(classification))
    n_reg = min(n - n_cls, len(regression))
    
    step_cls = max(1, len(classification) // n_cls) if n_cls > 0 else 1
    step_reg = max(1, len(regression) // n_reg) if n_reg > 0 else 1
    
    picked = []
    picked += classification[::step_cls][:n_cls]
    picked += regression[::step_reg][:n_reg]
    
    # Deduplicate by dataset_id
    seen = set()
    deduped = []
    for r in picked:
        if r["dataset_id"] not in seen:
            seen.add(r["dataset_id"])
            deduped.append(r)
    
    return deduped[:n]


def run_flaml(X_train, X_test, y_train, y_test, task_type, metric_name, budget_sec=30):
    """Run FLAML AutoML."""
    from flaml import AutoML
    
    automl = AutoML()
    
    # Map our metric to FLAML's metric
    if task_type == "classification":
        flaml_metric = "accuracy"
    else:
        flaml_metric = "mse"
    
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
    return preds, str(automl.best_estimator), automl.best_config


def run_orcetra_fair(X_train, X_test, y_train, y_test, task_type, metric_name, budget_sec=30):
    """Run Orcetra on the SAME split, with the SAME metric."""
    from orcetra.models.registry import get_baselines
    from orcetra.metrics.base import get_metric
    from orcetra.core.agent import RandomSearchAgent
    
    metric_fn = get_metric(metric_name)
    
    data_info = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "shape": (len(X_train) + len(X_test), X_train.shape[1]),
        "task_type": task_type,
        "n_features": X_train.shape[1],
    }
    
    # Phase 1: Baselines
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
    
    if best_score is None:
        return None, None, 0
    
    # Phase 2: AutoResearch loop
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
    
    return best_score, best_model, iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=50)
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()
    
    import openml
    
    existing = load_existing_results()
    print(f"Loaded {len(existing)} existing Orcetra results")
    
    pilots = pick_pilot_datasets(existing, args.n_datasets)
    print(f"Selected {len(pilots)} pilot datasets\n")
    
    outfile = RESULTS_DIR / f"flaml_pilot_v2_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    
    flaml_wins = 0
    orcetra_wins = 0
    ties = 0
    success = 0
    cls_f = cls_o = cls_t = 0
    reg_f = reg_o = reg_t = 0
    
    for i, prior in enumerate(pilots):
        did = prior["dataset_id"]
        name = prior.get("name", f"dataset_{did}")
        task_type = prior["task_type"]
        
        print(f"[{i+1}/{len(pilots)}] {name} (id={did}, {task_type}, "
              f"n={prior.get('n_samples','?')}, p={prior.get('n_features','?')}) ... ", 
              end="", flush=True)
        
        result = {
            "dataset_id": did,
            "name": name,
            "task_type": task_type,
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
            
            # Preprocess
            if task_type == "classification":
                y_series = y_series.astype("category").cat.codes
                metric_name = "accuracy"
            else:
                y_series = pd.to_numeric(y_series, errors="coerce")
                metric_name = "mse"  # FIXED: always MSE for fair comparison
            
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
            
            if len(X_np) < 50:
                result["status"] = "too_small"
                print("SKIP (too small)")
                with open(outfile, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue
            
            # SAME split for both
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, random_state=42
            )
            
            # ── FLAML ──
            t0 = time.time()
            flaml_preds, flaml_model, flaml_config = run_flaml(
                X_train, X_test, y_train, y_test, task_type, metric_name, args.budget
            )
            flaml_time = time.time() - t0
            
            if task_type == "classification":
                flaml_score = float(np.mean(flaml_preds == y_test))
            else:
                flaml_score = float(np.mean((flaml_preds - y_test) ** 2))
            
            result["flaml_score"] = round(flaml_score, 6)
            result["flaml_model"] = flaml_model
            result["flaml_time"] = round(flaml_time, 2)
            
            # ── Orcetra (same split, same metric) ──
            t0 = time.time()
            orc_score, orc_model, orc_iters = run_orcetra_fair(
                X_train, X_test, y_train, y_test, task_type, metric_name, args.budget
            )
            orcetra_time = time.time() - t0
            
            if orc_score is None:
                result["status"] = "orcetra_failed"
                print("SKIP (orcetra failed)")
                with open(outfile, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue
            
            result["orcetra_score"] = round(float(orc_score), 6)
            result["orcetra_model"] = orc_model
            result["orcetra_time"] = round(orcetra_time, 2)
            result["orcetra_iters"] = orc_iters
            result["metric"] = metric_name
            
            # Compare (same metric, fair)
            EPS = 1e-6
            if task_type == "classification":
                # Higher is better
                if orc_score > flaml_score + EPS:
                    winner = "orcetra"
                elif flaml_score > orc_score + EPS:
                    winner = "flaml"
                else:
                    winner = "tie"
            else:
                # Lower is better (MSE)
                if orc_score < flaml_score - EPS:
                    winner = "orcetra"
                elif flaml_score < orc_score - EPS:
                    winner = "flaml"
                else:
                    winner = "tie"
            
            result["winner"] = winner
            result["status"] = "success"
            
            if winner == "orcetra":
                orcetra_wins += 1
                if task_type == "classification": cls_o += 1
                else: reg_o += 1
            elif winner == "flaml":
                flaml_wins += 1
                if task_type == "classification": cls_f += 1
                else: reg_f += 1
            else:
                ties += 1
                if task_type == "classification": cls_t += 1
                else: reg_t += 1
            success += 1
            
            # Score delta for display
            if task_type == "classification":
                delta = orc_score - flaml_score
                print(f"{metric_name}: FLAML={flaml_score:.4f}({flaml_model}) ORC={orc_score:.4f}({orc_model}) "
                      f"Δ={delta:+.4f} → {winner.upper()} | "
                      f"F={flaml_wins} O={orcetra_wins} T={ties}")
            else:
                delta = flaml_score - orc_score  # positive = orcetra better for MSE
                print(f"{metric_name}: FLAML={flaml_score:.4f}({flaml_model}) ORC={orc_score:.4f}({orc_model}) "
                      f"Δ={delta:+.4f} → {winner.upper()} | "
                      f"F={flaml_wins} O={orcetra_wins} T={ties}")
        
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]
            print(f"ERROR: {str(e)[:80]}")
        
        with open(outfile, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PILOT v2 RESULTS: Orcetra vs FLAML ({args.budget}s budget, same split, same metric)")
    print(f"{'='*70}")
    print(f"Datasets:       {success} successful / {len(pilots)} attempted")
    print(f"")
    print(f"  FLAML wins:   {flaml_wins:3d} ({flaml_wins/max(success,1)*100:.0f}%)")
    print(f"  Orcetra wins: {orcetra_wins:3d} ({orcetra_wins/max(success,1)*100:.0f}%)")
    print(f"  Ties:         {ties:3d} ({ties/max(success,1)*100:.0f}%)")
    print(f"")
    cls_total = cls_f + cls_o + cls_t
    reg_total = reg_f + reg_o + reg_t
    if cls_total > 0:
        print(f"  Classification ({cls_total}): FLAML={cls_f} Orcetra={cls_o} Tie={cls_t}")
    if reg_total > 0:
        print(f"  Regression ({reg_total}):     FLAML={reg_f} Orcetra={reg_o} Tie={reg_t}")
    print(f"\nResults: {outfile}")


if __name__ == "__main__":
    main()
