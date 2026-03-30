#!/usr/bin/env python3
"""
Orcetra vs FLAML vs AutoGluon — strict 30s budget, 4 parallel workers.
Reuses the 513 datasets from previous strict 30s runs.
"""
import json
import time
import sys
import os
import warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
BUDGET = 60


def load_dataset_ids():
    """Load all dataset IDs that succeeded in strict 30s runs."""
    seen = {}
    for f in RESULTS_DIR.glob("flaml_strict30s_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    seen[r["dataset_id"]] = r
            except:
                pass
    return list(seen.values())


def run_autogluon(X_train, X_test, y_train, y_test, task_type, budget_sec=30):
    """Run AutoGluon with strict time budget."""
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil

    label_col = "__target__"
    train_df = pd.DataFrame(X_train)
    train_df[label_col] = y_train
    test_df = pd.DataFrame(X_test)

    problem = "binary" if task_type == "classification" and len(np.unique(y_train)) == 2 else \
              "multiclass" if task_type == "classification" else "regression"

    tmpdir = tempfile.mkdtemp()
    try:
        predictor = TabularPredictor(
            label=label_col,
            problem_type=problem,
            path=tmpdir,
            verbosity=0,
        ).fit(
            train_df,
            time_limit=budget_sec,
            presets="medium_quality",
            ag_args_fit={"num_gpus": 0},
        )
        preds = predictor.predict(test_df).values
        model_name = predictor.model_best
    except Exception as e:
        return None, str(e)[:100]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return preds, model_name


def run_flaml(X_train, X_test, y_train, y_test, task_type, budget_sec=30):
    from flaml import AutoML
    automl = AutoML()
    automl.fit(X_train, y_train, time_budget=budget_sec,
               task=task_type, metric="accuracy" if task_type == "classification" else "mse",
               log_file_name="/dev/null", verbose=0, seed=42)
    return automl.predict(X_test), str(automl.best_estimator)


def run_orcetra_strict(X_train, X_test, y_train, y_test, task_type, metric_name, budget_sec=30):
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

    t_start = time.time()
    best_score = None
    best_model = None

    for model_name, model_fn in get_baselines(task_type).items():
        if time.time() - t_start >= budget_sec:
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

    remaining = budget_sec - (time.time() - t_start)
    if remaining > 1 and best_score is not None:
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

    return best_score, best_model


def score_predictions(preds, y_test, task_type):
    if task_type == "classification":
        return float(np.mean(preds == y_test))
    else:
        return float(np.mean((preds - y_test) ** 2))


def determine_winner(scores, task_type):
    """Given dict of {name: score}, return winner name."""
    EPS = 1e-6
    higher_better = task_type == "classification"
    
    best_name = None
    best_score = None
    for name, score in scores.items():
        if score is None:
            continue
        if best_score is None:
            best_score = score
            best_name = name
        elif higher_better and score > best_score + EPS:
            best_score = score
            best_name = name
        elif not higher_better and score < best_score - EPS:
            best_score = score
            best_name = name

    # Check for ties
    tie_names = []
    for name, score in scores.items():
        if score is not None and abs(score - best_score) <= EPS:
            tie_names.append(name)

    if len(tie_names) > 1:
        return "tie"
    return best_name


def run_one(prior, worker_id=0):
    import openml

    did = prior["dataset_id"]
    name = prior.get("name", f"dataset_{did}")
    task_type = prior["task_type"]
    metric_name = "accuracy" if task_type == "classification" else "mse"

    result = {
        "dataset_id": did, "name": name, "task_type": task_type,
        "n_samples": prior.get("n_samples"), "n_features": prior.get("n_features"),
        "budget_sec": BUDGET, "timestamp": datetime.now().isoformat(),
    }

    try:
        ds = openml.datasets.get_dataset(did, download_data=True)
        X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)

        if X_df is None or y_series is None:
            result["status"] = "no_data"
            return result

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

        scores = {}
        models = {}
        times = {}

        # FLAML
        t0 = time.time()
        fl_preds, fl_model = run_flaml(X_train, X_test, y_train, y_test, task_type, BUDGET)
        times["flaml"] = round(time.time() - t0, 2)
        scores["flaml"] = score_predictions(fl_preds, y_test, task_type)
        models["flaml"] = fl_model

        # AutoGluon
        t0 = time.time()
        ag_preds, ag_model = run_autogluon(X_train, X_test, y_train, y_test, task_type, BUDGET)
        times["autogluon"] = round(time.time() - t0, 2)
        if ag_preds is not None:
            try:
                # AutoGluon may return int labels for classification
                if task_type == "classification":
                    ag_preds_eval = np.array(ag_preds).flatten()
                    y_test_eval = np.array(y_test).flatten()
                    scores["autogluon"] = float(np.mean(ag_preds_eval == y_test_eval))
                else:
                    ag_preds_eval = np.array(ag_preds, dtype=np.float64).flatten()
                    scores["autogluon"] = float(np.mean((ag_preds_eval - y_test) ** 2))
            except Exception as e:
                scores["autogluon"] = None
                models["autogluon"] = f"SCORE_ERR: {str(e)[:80]}"
            if scores.get("autogluon") is not None:
                models["autogluon"] = ag_model
        else:
            scores["autogluon"] = None
            models["autogluon"] = f"FAILED: {ag_model}"

        # Orcetra
        t0 = time.time()
        orc_score, orc_model = run_orcetra_strict(
            X_train, X_test, y_train, y_test, task_type, metric_name, BUDGET
        )
        times["orcetra"] = round(time.time() - t0, 2)
        scores["orcetra"] = float(orc_score) if orc_score is not None else None
        models["orcetra"] = orc_model

        winner = determine_winner(scores, task_type)

        result["scores"] = {k: round(v, 6) if v is not None else None for k, v in scores.items()}
        result["models"] = models
        result["times"] = times
        result["winner"] = winner
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]

    return result


def worker_fn(tasks, worker_id, outfile):
    for i, prior in enumerate(tasks):
        result = run_one(prior, worker_id)
        w = result.get("winner", "")
        s = result.get("scores", {})
        fs = s.get('flaml') or 0
        ag = s.get('autogluon') or 0
        os_ = s.get('orcetra') or 0
        print(f"[W{worker_id}] {i+1}/{len(tasks)} {prior['name'][:35]:35s} "
              f"F={fs:.4f} AG={ag:.4f} O={os_:.4f} "
              f"→ {w.upper()}", flush=True)
        with open(outfile, "a") as f:
            f.write(json.dumps(result) + "\n")


def main():
    datasets = load_dataset_ids()
    print(f"Loaded {len(datasets)} datasets from strict 30s runs")

    # Resume: skip already completed datasets
    done_ids = set()
    for f in RESULTS_DIR.glob("multi_framework_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    done_ids.add(r["dataset_id"])
            except:
                pass
    datasets = [d for d in datasets if d["dataset_id"] not in done_ids]
    print(f"Skipping {len(done_ids)} already completed, running {len(datasets)} remaining")

    outfile = RESULTS_DIR / f"multi_framework_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

    N_WORKERS = 4
    chunks = [datasets[i::N_WORKERS] for i in range(N_WORKERS)]

    # Ensure output file exists before workers write to it
    outfile.touch()

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

    from collections import Counter
    wins = Counter(r["winner"] for r in results)
    print(f"\n{'='*60}")
    print(f"MULTI-FRAMEWORK RESULTS ({len(results)} datasets, {BUDGET}s budget)")
    print(f"{'='*60}")
    for name in ["orcetra", "flaml", "autogluon", "tie"]:
        c = wins.get(name, 0)
        print(f"  {name:12s}: {c:4d} ({c/max(len(results),1)*100:.1f}%)")
    print(f"\nResults: {outfile}")


if __name__ == "__main__":
    main()
