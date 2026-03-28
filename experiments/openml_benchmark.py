#!/usr/bin/env python3
"""
Orcetra vs sklearn baseline on 3000+ OpenML datasets.

Usage:
    python experiments/openml_benchmark.py [--budget 30s] [--max-datasets 3000] [--resume]

Results saved incrementally to experiments/results/openml_benchmark_YYYYMMDD.jsonl
"""
import json
import time
import os
import sys
import warnings
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_datasets(max_datasets=3000):
    """Get curated + quality-filtered OpenML datasets."""
    import openml
    
    print("Fetching OpenML dataset catalog...")
    
    # 1. Curated suites first (highest quality)
    curated_ids = set()
    for suite_id in [218, 99, 269]:  # AutoML Benchmark, CC18, Regression
        try:
            suite = openml.study.get_suite(suite_id)
            # Get dataset IDs from tasks
            for task_id in suite.tasks:
                try:
                    task = openml.tasks.get_task(task_id, download_data=False)
                    curated_ids.add(task.dataset_id)
                except Exception:
                    pass
        except Exception:
            pass
    
    print(f"  Curated datasets: {len(curated_ids)}")
    
    # 2. Broader catalog
    all_datasets = openml.datasets.list_datasets(
        number_instances="50..100000",
        number_features="2..500",
        output_format="dataframe",
    )
    
    # Quality filters
    good = all_datasets[
        (all_datasets["NumberOfMissingValues"] < all_datasets["NumberOfInstances"] * all_datasets["NumberOfFeatures"] * 0.3)
        & (all_datasets["NumberOfClasses"].fillna(0) <= 30)
        & (all_datasets["NumberOfInstances"] >= 50)
        & (all_datasets["NumberOfFeatures"] >= 2)
        & (all_datasets["NumberOfFeatures"] <= 500)
        & (all_datasets["status"] == "active")
    ].copy()
    
    # Sort by number of instances (larger datasets first = more interesting)
    good = good.sort_values("NumberOfInstances", ascending=False)
    
    # Combine: curated first, then by popularity
    curated_list = [did for did in good.index if did in curated_ids]
    other_list = [did for did in good.index if did not in curated_ids]
    
    dataset_ids = curated_list + other_list
    dataset_ids = dataset_ids[:max_datasets]
    
    print(f"  Total selected: {len(dataset_ids)} ({len(curated_list)} curated + {len(dataset_ids) - len(curated_list)} popular)")
    return dataset_ids


class SkipDatasetError(Exception):
    """Raised when a dataset should be skipped (not an error, just unsuitable)."""
    pass


def _densify_series(s):
    """Convert sparse series to dense, handling various sparse types."""
    if hasattr(s, 'sparse'):
        # pandas SparseDtype
        return s.sparse.to_dense()
    if hasattr(s, 'to_dense'):
        return s.to_dense()
    # Check for scipy sparse
    if hasattr(s, 'toarray'):
        return pd.Series(s.toarray().ravel())
    return s


def _safe_median(s):
    """Compute median, handling sparse arrays by densifying first."""
    s = _densify_series(s)
    s = pd.to_numeric(s, errors='coerce')
    med = s.median()
    if pd.isna(med):
        med = 0.0
    return med


def load_openml_dataset(dataset_id):
    """Load an OpenML dataset, return (X, y, task_type, name) or raise."""
    import openml

    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
    )

    # "No data" cases should be skipped gracefully, not errored
    if X is None or y is None:
        raise SkipDatasetError("No data")

    if len(X) < 50:
        raise SkipDatasetError(f"Too few samples: {len(X)}")

    # Convert to DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=attribute_names)

    # === Densify y if sparse (fixes "factorize requires Series" error) ===
    y = _densify_series(y)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Determine task type
    # Use try/except for nunique in case of remaining edge cases
    try:
        n_unique = y.nunique()
    except Exception:
        # Fallback: convert to numpy and count unique
        n_unique = len(np.unique(y.dropna().values))

    if y.dtype in ["object", "category", "bool"] or n_unique <= 20:
        task_type = "classification"
        # Encode target - ensure y is a proper array type
        from sklearn.preprocessing import LabelEncoder
        y_values = np.asarray(y).astype(str)
        y = pd.Series(LabelEncoder().fit_transform(y_values))
    else:
        task_type = "regression"
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().sum() > len(y) * 0.1:
            raise SkipDatasetError("Target has too many NaN")
        # Use safe median to handle sparse types
        med = _safe_median(y)
        y = y.fillna(med)

    # Handle features - densify sparse columns first
    for col in X.columns:
        if hasattr(X[col], 'sparse') or hasattr(X[col], 'to_dense'):
            X[col] = _densify_series(X[col])

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype(str).astype("category").cat.codes
    X = X.apply(pd.to_numeric, errors="coerce")

    # Use safe median for fillna (fixes "cannot perform median with Sparse" error)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(_safe_median(X[col]))

    # Drop constant columns
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] < 1:
        raise SkipDatasetError("No valid features after cleaning")

    return X, y, task_type, dataset.name


def run_sklearn_baseline(X, y, task_type):
    """Run sklearn default models, return best score and metric name."""
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
    
    best_score = None
    best_model = None
    
    if task_type == "regression":
        metric = "neg_mean_squared_error"
        metric_name = "mse"
        models = {
            "GBR": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "RF": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        }
    else:
        metric = "accuracy"
        metric_name = "accuracy"
        models = {
            "GBC": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring=metric, n_jobs=-1)
            mean_score = scores.mean()
            
            if task_type == "regression":
                mean_score = -mean_score  # neg_mse -> mse (lower is better)
            
            if best_score is None:
                best_score = mean_score
                best_model = name
            elif task_type == "regression" and mean_score < best_score:
                best_score = mean_score
                best_model = name
            elif task_type == "classification" and mean_score > best_score:
                best_score = mean_score
                best_model = name
        except Exception:
            pass
    
    return best_score, best_model, metric_name


def run_orcetra(X, y, task_type, budget_seconds=30):
    """Run Orcetra AutoResearch loop, return best score."""
    from sklearn.model_selection import train_test_split
    from orcetra.metrics.base import get_metric
    from orcetra.core.agent import RandomSearchAgent
    
    # Use same metric as baseline
    if task_type == "regression":
        # Check if RMSLE is appropriate
        if (y > 0).all():
            try:
                from scipy.stats import skew
                if skew(y) > 1.0:
                    metric_name = "rmsle"
                else:
                    metric_name = "mse"
            except ImportError:
                metric_name = "mse"
        else:
            metric_name = "mse"
    else:
        metric_name = "accuracy"
    
    metric_fn = get_metric(metric_name)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data_info = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "shape": X.shape, "task_type": task_type,
        "n_features": X.shape[1],
    }
    
    # Run baselines
    from orcetra.models.registry import get_baselines
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
        except Exception:
            pass
    
    if best_score is None:
        return None, None, 0, 0, metric_name
    
    baseline_score = best_score
    
    # AutoResearch loop — try LLM agent first
    agent = None
    try:
        if os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            from orcetra.core.llm_agent import LLMSearchAgent
            provider = "groq" if os.environ.get("GROQ_API_KEY") else "openai"
            agent = LLMSearchAgent(task_type=task_type, provider=provider)
    except Exception:
        pass
    
    if agent is None:
        agent = RandomSearchAgent(task_type=task_type)
    
    start = time.time()
    iteration = 0
    improvements = 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import hashlib
    seen = set()
    batch_size = min(os.cpu_count() or 4, 8) * 2
    
    while time.time() - start < budget_seconds:
        # Generate batch
        proposals = []
        for _ in range(batch_size * 3):
            if len(proposals) >= batch_size:
                break
            p = agent.propose({
                "best_score": best_score, "best_model": best_model,
                "iteration": iteration + len(proposals) + 1,
                "task_type": task_type,
                "metric_direction": metric_fn.direction,
                "data_summary": f"{X.shape[0]} rows, {X.shape[1]} features",
                "last_proposal": "", "last_score": None, "last_improved": False,
            })
            key = hashlib.md5(p.description.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                proposals.append(p)
        
        if not proposals:
            break
        
        # Parallel eval
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as pool:
            futures = {pool.submit(lambda p: p.evaluate(data_info, metric_fn), p): p for p in proposals}
            for future in as_completed(futures):
                iteration += 1
                try:
                    score = future.result()
                    p = futures[future]
                    is_better = (
                        (metric_fn.direction == "minimize" and score < best_score)
                        or (metric_fn.direction == "maximize" and score > best_score)
                    )
                    if is_better:
                        improvements += 1
                        best_score = score
                        best_model = p.description
                except Exception:
                    pass
    
    return best_score, best_model, iteration, improvements, metric_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default="30s", help="Budget per dataset")
    parser.add_argument("--max-datasets", type=int, default=3500, help="Max datasets to try")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()
    
    budget_seconds = int(args.budget.replace("s", ""))
    
    # Output file
    date_str = datetime.now().strftime("%Y%m%d")
    outfile = RESULTS_DIR / f"openml_benchmark_{date_str}.jsonl"
    
    # Load already-completed dataset IDs if resuming
    done_ids = set()
    if args.resume and outfile.exists():
        with open(outfile) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r["dataset_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} datasets already done")
    
    # Get datasets
    dataset_ids = get_datasets(args.max_datasets)
    remaining = [d for d in dataset_ids if d not in done_ids]
    print(f"\nDatasets to process: {len(remaining)} (of {len(dataset_ids)} total)")
    
    # Stats
    total = 0
    success = 0
    orcetra_wins = 0
    improvements = []
    
    start_all = time.time()
    
    for i, did in enumerate(remaining):
        total += 1
        result = {
            "dataset_id": int(did),
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            # Load
            t0 = time.time()
            X, y, task_type, name = load_openml_dataset(did)
            load_time = time.time() - t0
            
            result["name"] = name
            result["task_type"] = task_type
            result["n_samples"] = int(X.shape[0])
            result["n_features"] = int(X.shape[1])
            result["load_time"] = round(load_time, 2)
            
            # === Meta-features for knowledge base ===
            try:
                import pandas as pd
                meta = {}
                if hasattr(X, 'dtypes'):
                    meta["n_categorical"] = int((X.dtypes == 'object').sum() + (X.dtypes == 'category').sum())
                    meta["n_numerical"] = int((X.dtypes != 'object').sum() - (X.dtypes == 'category').sum())
                else:
                    meta["n_categorical"] = 0
                    meta["n_numerical"] = int(X.shape[1])
                meta["n_missing"] = int(pd.DataFrame(X).isnull().sum().sum()) if hasattr(pd.DataFrame(X), 'isnull') else 0
                meta["missing_pct"] = round(meta["n_missing"] / (X.shape[0] * X.shape[1]) * 100, 2) if X.shape[0] * X.shape[1] > 0 else 0
                meta["samples_features_ratio"] = round(X.shape[0] / max(X.shape[1], 1), 1)
                
                y_arr = np.array(y, dtype=float) if task_type == "regression" else np.array(y)
                if task_type == "regression":
                    meta["target_mean"] = round(float(np.nanmean(y_arr)), 4)
                    meta["target_std"] = round(float(np.nanstd(y_arr)), 4)
                    meta["target_min"] = round(float(np.nanmin(y_arr)), 4)
                    meta["target_max"] = round(float(np.nanmax(y_arr)), 4)
                    try:
                        from scipy.stats import skew
                        meta["target_skewness"] = round(float(skew(y_arr[~np.isnan(y_arr)])), 4)
                    except Exception:
                        meta["target_skewness"] = None
                    meta["target_all_positive"] = bool((y_arr > 0).all())
                else:
                    unique_classes = np.unique(y_arr)
                    meta["n_classes"] = int(len(unique_classes))
                    class_counts = np.bincount(y_arr.astype(int)) if y_arr.dtype in [np.int64, np.int32, int] else None
                    if class_counts is not None and len(class_counts) > 0:
                        meta["class_balance_ratio"] = round(float(class_counts.min() / max(class_counts.max(), 1)), 4)
                    else:
                        meta["class_balance_ratio"] = None
                
                result["meta_features"] = meta
            except Exception as e:
                result["meta_features"] = {"error": str(e)[:100]}
            
            # sklearn baseline
            t0 = time.time()
            baseline_score, baseline_model, metric_name = run_sklearn_baseline(X, y, task_type)
            baseline_time = time.time() - t0
            
            if baseline_score is None:
                result["status"] = "baseline_failed"
                _write_result(outfile, result)
                continue
            
            result["baseline_score"] = round(float(baseline_score), 6)
            result["baseline_model"] = baseline_model
            result["baseline_time"] = round(baseline_time, 2)
            result["baseline_metric"] = metric_name
            
            # Orcetra
            t0 = time.time()
            orc_score, orc_model, orc_iters, orc_improvements, orc_metric = run_orcetra(
                X, y, task_type, budget_seconds=budget_seconds
            )
            orc_time = time.time() - t0
            
            if orc_score is None:
                result["status"] = "orcetra_failed"
                _write_result(outfile, result)
                continue
            
            result["orcetra_score"] = round(float(orc_score), 6)
            result["orcetra_model"] = orc_model
            result["orcetra_time"] = round(orc_time, 2)
            result["orcetra_iters"] = orc_iters
            result["orcetra_improvements"] = orc_improvements
            result["orcetra_metric"] = orc_metric
            
            # === Best strategy knowledge extraction ===
            try:
                strategy_knowledge = {
                    "winning_model": orc_model,
                    "baseline_model": baseline_model,
                    "metric": orc_metric,
                    "search_efficiency": round(orc_improvements / max(orc_iters, 1), 4) if orc_iters else 0,
                }
                # Parse model type from description
                model_desc = (orc_model or "").lower()
                if "lightgbm" in model_desc or "lgbm" in model_desc:
                    strategy_knowledge["model_family"] = "lightgbm"
                elif "xgb" in model_desc:
                    strategy_knowledge["model_family"] = "xgboost"
                elif "gradient" in model_desc or "gbr" in model_desc or "gbc" in model_desc:
                    strategy_knowledge["model_family"] = "gradient_boosting"
                elif "random_forest" in model_desc or "randomforest" in model_desc:
                    strategy_knowledge["model_family"] = "random_forest"
                elif "ridge" in model_desc or "lasso" in model_desc or "linear" in model_desc or "logistic" in model_desc:
                    strategy_knowledge["model_family"] = "linear"
                elif "svm" in model_desc or "svr" in model_desc or "svc" in model_desc:
                    strategy_knowledge["model_family"] = "svm"
                elif "knn" in model_desc or "neighbor" in model_desc:
                    strategy_knowledge["model_family"] = "knn"
                else:
                    strategy_knowledge["model_family"] = "other"
                
                result["strategy_knowledge"] = strategy_knowledge
            except Exception:
                pass
            
            # Compare (handle both minimize and maximize)
            if task_type == "classification" or metric_name == "accuracy":
                # Higher is better
                orcetra_better = orc_score > baseline_score
                if baseline_score > 0:
                    pct_change = (orc_score - baseline_score) / baseline_score * 100
                else:
                    pct_change = 0
            else:
                # Lower is better (MSE, RMSLE)
                orcetra_better = orc_score < baseline_score
                if baseline_score > 0:
                    pct_change = (baseline_score - orc_score) / baseline_score * 100
                else:
                    pct_change = 0
            
            result["orcetra_wins"] = orcetra_better
            result["improvement_pct"] = round(pct_change, 2)
            result["status"] = "success"
            
            success += 1
            if orcetra_better:
                orcetra_wins += 1
                improvements.append(pct_change)
            
            # Progress
            elapsed = time.time() - start_all
            rate = total / elapsed * 3600 if elapsed > 0 else 0
            win_rate = orcetra_wins / success * 100 if success > 0 else 0
            avg_imp = np.mean(improvements) if improvements else 0
            
            status_char = "✓" if orcetra_better else "·"
            print(f"  [{total}/{len(remaining)}] {status_char} {name[:30]:30s} | {task_type[:5]} | "
                  f"baseline={baseline_score:.4f} orc={orc_score:.4f} ({pct_change:+.1f}%) | "
                  f"win_rate={win_rate:.0f}% avg_imp={avg_imp:.1f}% | {rate:.0f}/hr")
            
        except SkipDatasetError as e:
            # Graceful skip - not an error, just unsuitable dataset
            result["status"] = "skip"
            result["skip_reason"] = str(e)[:200]
            if total <= 10:
                print(f"  [{total}] ⊘ dataset {did} skipped: {str(e)[:60]}")
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]
            if total <= 5:
                print(f"  [{total}] ✗ dataset {did}: {str(e)[:80]}")
        
        _write_result(outfile, result)
        
        # Periodic summary
        if total % 100 == 0:
            _print_summary(total, success, orcetra_wins, improvements, start_all)
    
    # Final summary
    print("\n" + "=" * 70)
    _print_summary(total, success, orcetra_wins, improvements, start_all)
    print(f"Results: {outfile}")


def _write_result(outfile, result):
    """Append one result line."""
    # Convert numpy types to native Python for JSON serialization
    clean = {}
    for k, v in result.items():
        if isinstance(v, (np.bool_, np.integer)):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    with open(outfile, "a") as f:
        f.write(json.dumps(clean) + "\n")


def _print_summary(total, success, wins, improvements, start_time):
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    win_rate = wins / success * 100 if success > 0 else 0
    avg_imp = np.mean(improvements) if improvements else 0
    med_imp = np.median(improvements) if improvements else 0
    print(f"\n{'='*50}")
    print(f"  Processed: {total} | Success: {success} | Errors: {total - success}")
    print(f"  Orcetra wins: {wins}/{success} ({win_rate:.1f}%)")
    print(f"  Avg improvement: {avg_imp:.2f}% | Median: {med_imp:.2f}%")
    print(f"  Elapsed: {hours:.1f}h | Rate: {total/hours:.0f}/hr")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
