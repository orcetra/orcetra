#!/usr/bin/env python3
"""
Run strict 30s Orcetra vs FLAML on remaining datasets (parallel).
Skips datasets already in flaml_strict30s_*.jsonl.
"""
import json
import time
import sys
import os
import warnings
import traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
BUDGET = 30


def load_done_ids():
    done = set()
    for f in RESULTS_DIR.glob("flaml_strict30s_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    done.add(r["dataset_id"])
            except:
                pass
    return done


def load_candidates():
    candidates = []
    for f in RESULTS_DIR.glob("openml_benchmark_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    candidates.append(r)
            except:
                pass
    return candidates


def run_one_dataset(prior, worker_id=0):
    """Run a single dataset comparison. Returns result dict."""
    import openml
    from orcetra.models.registry import get_baselines
    from orcetra.metrics.base import get_metric
    from orcetra.core.agent import RandomSearchAgent
    from flaml import AutoML

    did = prior["dataset_id"]
    name = prior.get("name", f"dataset_{did}")
    task_type = prior["task_type"]

    result = {
        "dataset_id": did, "name": name, "task_type": task_type,
        "n_samples": prior.get("n_samples"), "n_features": prior.get("n_features"),
        "budget_sec": BUDGET, "timestamp": datetime.now().isoformat(),
        "experiment": "strict_30s_full",
    }

    try:
        ds = openml.datasets.get_dataset(did, download_data=True)
        X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)

        if X_df is None or y_series is None:
            result["status"] = "no_data"
            return result

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
            return result

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42
        )

        # FLAML
        t0 = time.time()
        automl = AutoML()
        automl.fit(X_train, y_train, time_budget=BUDGET,
                   task=task_type, metric="accuracy" if task_type == "classification" else "mse",
                   log_file_name="/dev/null", verbose=0, seed=42)
        flaml_preds = automl.predict(X_test)
        flaml_time = time.time() - t0

        if task_type == "classification":
            flaml_score = float(np.mean(flaml_preds == y_test))
        else:
            flaml_score = float(np.mean((flaml_preds - y_test) ** 2))

        result["flaml_score"] = round(flaml_score, 6)
        result["flaml_model"] = str(automl.best_estimator)
        result["flaml_time"] = round(flaml_time, 2)

        # Orcetra STRICT
        metric_fn = get_metric(metric_name)
        data_info = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "shape": (len(X_train) + len(X_test), X_train.shape[1]),
            "task_type": task_type, "n_features": X_train.shape[1],
        }

        t_start = time.time()
        best_score = None
        best_model = None

        for model_name, model_fn in get_baselines(task_type).items():
            if time.time() - t_start >= BUDGET:
                break
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

        if best_score is not None:
            remaining = BUDGET - (time.time() - t_start)
            if remaining > 1:
                agent = RandomSearchAgent(task_type=task_type)
                iteration = 0
                while time.time() - t_start < BUDGET:
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

        orcetra_time = time.time() - t_start

        if best_score is None:
            result["status"] = "orcetra_failed"
            return result

        result["orcetra_score"] = round(float(best_score), 6)
        result["orcetra_model"] = best_model
        result["orcetra_time"] = round(orcetra_time, 2)
        result["orcetra_baseline_time"] = round(baseline_time, 2)
        result["metric"] = metric_name

        EPS = 1e-6
        if task_type == "classification":
            if best_score > flaml_score + EPS: winner = "orcetra"
            elif flaml_score > best_score + EPS: winner = "flaml"
            else: winner = "tie"
        else:
            if best_score < flaml_score - EPS: winner = "orcetra"
            elif flaml_score < best_score - EPS: winner = "flaml"
            else: winner = "tie"

        result["winner"] = winner
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]

    return result


def worker_fn(tasks, worker_id, outfile):
    """Worker process: run assigned datasets sequentially."""
    for i, prior in enumerate(tasks):
        result = run_one_dataset(prior, worker_id)
        status = result.get("status", "?")
        winner = result.get("winner", "")
        fs = result.get("flaml_score", 0)
        os_ = result.get("orcetra_score", 0)
        print(f"[W{worker_id}] {i+1}/{len(tasks)} {prior['name'][:40]} → {status} {winner.upper() if winner else ''} F={fs:.4f} O={os_:.4f}", flush=True)
        with open(outfile, "a") as f:
            f.write(json.dumps(result) + "\n")


def main():
    done_ids = load_done_ids()
    candidates = load_candidates()
    remaining = [r for r in candidates if r["dataset_id"] not in done_ids]

    # Deduplicate
    seen = set()
    deduped = []
    for r in remaining:
        if r["dataset_id"] not in seen:
            seen.add(r["dataset_id"])
            deduped.append(r)

    print(f"Already done: {len(done_ids)}")
    print(f"Remaining: {len(deduped)}")
    print(f"Starting 4 parallel workers...\n")

    outfile = RESULTS_DIR / f"flaml_strict30s_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

    N_WORKERS = 4
    chunks = [deduped[i::N_WORKERS] for i in range(N_WORKERS)]

    processes = []
    for wid, chunk in enumerate(chunks):
        p = mp.Process(target=worker_fn, args=(chunk, wid, str(outfile)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Summary
    results = []
    for line in open(outfile):
        try:
            r = json.loads(line)
            if r.get("status") == "success":
                results.append(r)
        except:
            pass

    ow = sum(1 for r in results if r["winner"] == "orcetra")
    fw = sum(1 for r in results if r["winner"] == "flaml")
    tw = sum(1 for r in results if r["winner"] == "tie")
    print(f"\n{'='*60}")
    print(f"REMAINING BATCH: {len(results)} datasets")
    print(f"  Orcetra: {ow} ({ow/max(len(results),1)*100:.0f}%)")
    print(f"  FLAML:   {fw} ({fw/max(len(results),1)*100:.0f}%)")
    print(f"  Tie:     {tw} ({tw/max(len(results),1)*100:.0f}%)")
    print(f"\nResults: {outfile}")


if __name__ == "__main__":
    main()
