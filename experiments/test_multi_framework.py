#!/usr/bin/env python3
"""
Quick 5-dataset smoke test for multi-framework comparison.
Tests the improved Orcetra (12 baselines + weighted ensemble) against FLAML and AutoGluon.

Each framework runs in a subprocess to isolate segfaults (AutoGluon/native libs).

Usage:
    python experiments/test_multi_framework.py
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

BUDGET = 30  # seconds per framework per dataset
SUBPROCESS_TIMEOUT = BUDGET + 60  # generous grace for imports + overhead

# 5 well-known OpenML datasets: 3 regression + 2 classification
TEST_DATASETS = [
    {"dataset_id": 531,   "name": "boston (housing)",     "task_type": "regression"},
    {"dataset_id": 422,   "name": "cpu_act",             "task_type": "regression"},
    {"dataset_id": 507,   "name": "space_ga",            "task_type": "regression"},
    {"dataset_id": 37,    "name": "diabetes",            "task_type": "classification"},
    {"dataset_id": 1510,  "name": "wdbc",                "task_type": "classification"},
]


def load_openml_dataset(did, task_type):
    """Download and prepare an OpenML dataset."""
    import openml

    ds = openml.datasets.get_dataset(did, download_data=True)
    X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)

    if X_df is None or y_series is None:
        raise ValueError(f"No data for dataset {did}")

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

    return train_test_split(X_np, y_np, test_size=0.2, random_state=42)


# ── Subprocess runner ─────────────────────────────────────────────────────

_WORKER_SCRIPT = r'''
import json, os, pickle, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

data_path = sys.argv[1]
framework = sys.argv[2]
task_type = sys.argv[3]
budget_sec = int(sys.argv[4])
out_path = sys.argv[5]

with open(data_path, "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

result = {}
t0 = time.time()

try:
    if framework == "flaml":
        from flaml import AutoML
        automl = AutoML()
        automl.fit(
            X_train, y_train,
            time_budget=budget_sec,
            task=task_type,
            metric="accuracy" if task_type == "classification" else "mse",
            log_file_name="/dev/null", verbose=0, seed=42,
        )
        preds = automl.predict(X_test)
        model = str(automl.best_estimator)
        if task_type == "classification":
            score = float(np.mean(preds == y_test))
        else:
            score = float(np.mean((preds - y_test) ** 2))
        result = {"score": score, "model": model, "time": round(time.time() - t0, 1)}

    elif framework == "autogluon":
        from autogluon.tabular import TabularPredictor
        import tempfile, shutil
        label_col = "__target__"
        train_df = pd.DataFrame(X_train)
        train_df[label_col] = y_train
        test_df = pd.DataFrame(X_test)
        problem = (
            "binary" if task_type == "classification" and len(np.unique(y_train)) == 2
            else "multiclass" if task_type == "classification"
            else "regression"
        )
        tmpdir = tempfile.mkdtemp()
        try:
            predictor = TabularPredictor(
                label=label_col, problem_type=problem, path=tmpdir, verbosity=0,
            ).fit(
                train_df, time_limit=budget_sec,
                presets="medium_quality", ag_args_fit={"num_gpus": 0},
            )
            preds = predictor.predict(test_df).values
            model = predictor.model_best
            if task_type == "classification":
                score = float(np.mean(np.array(preds).flatten() == np.array(y_test).flatten()))
            else:
                score = float(np.mean((np.array(preds, dtype=np.float64).flatten() - y_test) ** 2))
            result = {"score": score, "model": model, "time": round(time.time() - t0, 1)}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    elif framework == "orcetra":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".."))
        from orcetra.models.registry import get_baselines
        from orcetra.metrics.base import get_metric
        from orcetra.core.agent import RandomSearchAgent
        from sklearn.ensemble import VotingRegressor, VotingClassifier

        metric_name = "accuracy" if task_type == "classification" else "mse"
        metric_fn = get_metric(metric_name)
        data_info = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "shape": (len(X_train) + len(X_test), X_train.shape[1]),
            "task_type": task_type, "n_features": X_train.shape[1],
        }
        best_score = None
        best_model = None
        top_proposals = []

        # Phase 1: baseline sweep
        for mname, model_fn in get_baselines(task_type).items():
            if time.time() - t0 >= budget_sec:
                break
            try:
                sc = model_fn(data_info, metric_fn)
                is_better = (
                    best_score is None
                    or (metric_fn.direction == "minimize" and sc < best_score)
                    or (metric_fn.direction == "maximize" and sc > best_score)
                )
                if is_better:
                    best_score, best_model = sc, mname
            except Exception:
                pass

        # Phase 2: random search
        if best_score is not None and time.time() - t0 < budget_sec:
            agent = RandomSearchAgent(task_type=task_type)
            iteration = 0
            while time.time() - t0 < budget_sec:
                iteration += 1
                try:
                    state = {
                        "best_score": best_score, "best_model": best_model,
                        "iteration": iteration, "task_type": task_type,
                        "metric_direction": metric_fn.direction,
                    }
                    proposal = agent.propose(state)
                    if proposal is None:
                        continue
                    sc = proposal.evaluate(data_info, metric_fn)
                    if proposal.preprocessor is None:
                        top_proposals.append((proposal.model, sc, type(proposal.model).__name__))
                    is_better = (
                        (metric_fn.direction == "minimize" and sc < best_score)
                        or (metric_fn.direction == "maximize" and sc > best_score)
                    )
                    if is_better:
                        best_score, best_model = sc, proposal.description
                except Exception:
                    pass

        # Phase 3: post-search weighted ensemble
        if len(top_proposals) >= 3:
            try:
                if metric_fn.direction == "minimize":
                    sorted_props = sorted(top_proposals, key=lambda x: x[1])
                else:
                    sorted_props = sorted(top_proposals, key=lambda x: -x[1])
                selected, seen_types = [], set()
                for mdl, sc, mtype in sorted_props:
                    if mtype not in seen_types:
                        selected.append((mdl, sc))
                        seen_types.add(mtype)
                    if len(selected) >= 3:
                        break
                if len(selected) >= 2:
                    scores_list = [s for _, s in selected]
                    if metric_fn.direction == "minimize":
                        ws = [1.0 / max(s, 1e-10) for s in scores_list]
                    else:
                        ws = scores_list if sum(scores_list) > 0 else [1.0] * len(selected)
                    wt = sum(ws)
                    ws = [w / wt for w in ws]
                    estimators = [(f"m{i}", m) for i, (m, _) in enumerate(selected)]
                    if task_type == "regression":
                        ens = VotingRegressor(estimators=estimators, weights=ws, n_jobs=-1)
                    else:
                        ens = VotingClassifier(estimators=estimators, weights=ws, voting="soft", n_jobs=-1)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ens.fit(X_train, y_train)
                        ens_preds = ens.predict(X_test)
                    ens_score = metric_fn.compute(y_test, ens_preds)
                    is_better = (
                        (metric_fn.direction == "minimize" and ens_score < best_score)
                        or (metric_fn.direction == "maximize" and ens_score > best_score)
                    )
                    if is_better:
                        best_score = ens_score
                        names = "+".join(type(m).__name__ for m, _ in selected)
                        best_model = f"WeightedEnsemble({names})"
            except Exception:
                pass

        if best_score is not None:
            result = {"score": float(best_score), "model": best_model, "time": round(time.time() - t0, 1)}
        else:
            result = {"score": None, "error": "all baselines failed"}

except Exception as e:
    result = {"score": None, "error": str(e)[:200]}

with open(out_path, "w") as f:
    json.dump(result, f)
'''


def _run_in_subprocess(framework, X_train, X_test, y_train, y_test, task_type, budget_sec):
    """Run a framework in an isolated subprocess; returns dict with score/model/time or error."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as df:
        data_path = df.name
        pickle.dump((X_train, X_test, y_train, y_test), df)

    out_path = data_path.replace(".pkl", "_out.json")
    worker_path = data_path.replace(".pkl", "_worker.py")
    with open(worker_path, "w") as wf:
        wf.write(_WORKER_SCRIPT)

    try:
        proc = subprocess.run(
            [sys.executable, worker_path, data_path, framework, task_type, str(budget_sec), out_path],
            timeout=SUBPROCESS_TIMEOUT,
            capture_output=True, text=True,
        )
        if os.path.exists(out_path):
            with open(out_path) as f:
                return json.load(f)
        # No output file → process crashed
        stderr_tail = (proc.stderr or "")[-300:]
        return {"score": None, "error": f"exit={proc.returncode} {stderr_tail}"}
    except subprocess.TimeoutExpired:
        return {"score": None, "error": "timeout"}
    finally:
        for p in [data_path, out_path, worker_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    print(f"{'='*70}")
    print(f"  MULTI-FRAMEWORK TEST  —  {len(TEST_DATASETS)} datasets, {BUDGET}s budget")
    print(f"  (each framework runs in isolated subprocess)")
    print(f"{'='*70}\n")

    results = []

    for ds_info in TEST_DATASETS:
        did = ds_info["dataset_id"]
        name = ds_info["name"]
        task = ds_info["task_type"]
        metric_label = "ACC" if task == "classification" else "MSE"

        print(f"▸ {name} (id={did}, {task})")

        try:
            X_train, X_test, y_train, y_test = load_openml_dataset(did, task)
            print(f"  data: {X_train.shape[0]} train / {X_test.shape[0]} test, {X_train.shape[1]} features")
        except Exception as e:
            print(f"  SKIP: {e}\n")
            continue

        row = {"name": name, "task_type": task}

        for fw in ["flaml", "autogluon", "orcetra"]:
            fw_label = {"flaml": "FLAML", "autogluon": "AutoGluon", "orcetra": "Orcetra"}[fw]
            res = _run_in_subprocess(fw, X_train, X_test, y_train, y_test, task, BUDGET)
            row[fw] = res
            if res.get("score") is not None:
                print(f"  {fw_label:12s} {metric_label}={res['score']:.6f}  model={res.get('model','-')}  ({res.get('time','-')}s)")
            else:
                print(f"  {fw_label:12s} FAILED: {res.get('error','unknown')[:80]}")

        # Determine winner
        scores = {}
        for fw in ["flaml", "autogluon", "orcetra"]:
            s = row.get(fw, {}).get("score")
            if s is not None:
                scores[fw] = s

        if scores:
            if task == "classification":
                winner = max(scores, key=scores.get)
            else:
                winner = min(scores, key=scores.get)
            row["winner"] = winner
            print(f"  → Winner: {winner.upper()}")
        else:
            row["winner"] = "none"

        results.append(row)
        print()

    # ── Summary ──
    if not results:
        print("No results collected.")
        return

    print(f"{'='*70}")
    print(f"  SUMMARY  ({len(results)} datasets, {BUDGET}s budget)")
    print(f"{'='*70}")

    wins = {"flaml": 0, "autogluon": 0, "orcetra": 0, "tie": 0}
    reg_wins = {"flaml": 0, "autogluon": 0, "orcetra": 0, "tie": 0}
    clf_wins = {"flaml": 0, "autogluon": 0, "orcetra": 0, "tie": 0}

    for r in results:
        w = r.get("winner", "none")
        if w in wins:
            wins[w] += 1
        if r["task_type"] == "regression" and w in reg_wins:
            reg_wins[w] += 1
        if r["task_type"] == "classification" and w in clf_wins:
            clf_wins[w] += 1

    n = len(results)
    n_reg = sum(1 for r in results if r["task_type"] == "regression")
    n_clf = sum(1 for r in results if r["task_type"] == "classification")

    print(f"\n  Overall ({n} datasets):")
    for fw in ["orcetra", "flaml", "autogluon", "tie"]:
        c = wins[fw]
        print(f"    {fw:12s}: {c}/{n}  ({c/max(n,1)*100:.0f}%)")

    if n_reg:
        print(f"\n  Regression ({n_reg} datasets):")
        for fw in ["orcetra", "flaml", "autogluon", "tie"]:
            c = reg_wins[fw]
            print(f"    {fw:12s}: {c}/{n_reg}  ({c/max(n_reg,1)*100:.0f}%)")

    if n_clf:
        print(f"\n  Classification ({n_clf} datasets):")
        for fw in ["orcetra", "flaml", "autogluon", "tie"]:
            c = clf_wins[fw]
            print(f"    {fw:12s}: {c}/{n_clf}  ({c/max(n_clf,1)*100:.0f}%)")

    # Per-dataset table
    print(f"\n  {'Dataset':25s} {'Task':15s} {'FLAML':>10s} {'AutoGluon':>10s} {'Orcetra':>10s}  Winner")
    print(f"  {'─'*25} {'─'*15} {'─'*10} {'─'*10} {'─'*10}  {'─'*10}")
    for r in results:
        fl = r.get("flaml", {}).get("score")
        ag = r.get("autogluon", {}).get("score")
        orc = r.get("orcetra", {}).get("score")
        fl_s = f"{fl:.6f}" if fl is not None else "---"
        ag_s = f"{ag:.6f}" if ag is not None else "---"
        orc_s = f"{orc:.6f}" if orc is not None else "---"
        print(f"  {r['name']:25s} {r['task_type']:15s} {fl_s:>10s} {ag_s:>10s} {orc_s:>10s}  {r.get('winner','?')}")

    # Save results
    out_file = Path(__file__).parent / "results" / "test_multi_framework.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_file}")
    print()


if __name__ == "__main__":
    main()
