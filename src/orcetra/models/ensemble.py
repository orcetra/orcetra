"""Weighted ensemble of top-K models.

Lightweight alternative to AutoGluon-style stacking: take the top-3 models
from baseline evaluation, re-fit them, and combine predictions via weighted
average (regression) or weighted voting (classification).

Weights are proportional to inverse error (better models get higher weight).
"""
import time
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler


# Model factory: maps baseline names to (constructor, needs_scaling) pairs.
# Kept in sync with baseline.py / registry.py.
def _get_model_factories(task_type: str) -> dict:
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
        HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    )

    try:
        from xgboost import XGBRegressor, XGBClassifier
        _HAS_XGB = True
    except ImportError:
        _HAS_XGB = False

    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
        _HAS_LGBM = True
    except ImportError:
        _HAS_LGBM = False

    if task_type == "regression":
        factories = {
            "LinearRegression": (lambda: LinearRegression(), True),
            "Ridge": (lambda: Ridge(alpha=1.0, random_state=42), True),
            "RandomForest": (lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
            "ExtraTrees": (lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
            "GradientBoosting": (lambda: GradientBoostingRegressor(n_estimators=100, random_state=42), False),
            "HistGradientBoosting": (lambda: HistGradientBoostingRegressor(max_iter=200, random_state=42), False),
        }
        if _HAS_XGB:
            factories["XGBoost"] = (lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1), False)
        if _HAS_LGBM:
            factories["LightGBM"] = (lambda: LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1), False)
    else:
        factories = {
            "LogisticRegression": (lambda: LogisticRegression(max_iter=1000, random_state=42), True),
            "RandomForest": (lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
            "ExtraTrees": (lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
            "GradientBoosting": (lambda: GradientBoostingClassifier(n_estimators=100, random_state=42), False),
            "HistGradientBoosting": (lambda: HistGradientBoostingClassifier(max_iter=200, random_state=42), False),
        }
        if _HAS_XGB:
            factories["XGBoost"] = (lambda: XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss', n_jobs=-1), False)
        if _HAS_LGBM:
            factories["LightGBM"] = (lambda: LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1), False)

    return factories


def _compute_weights(scores: list[float], direction: str) -> np.ndarray:
    """Compute ensemble weights from scores (inverse-error weighting).

    Better-scoring models get higher weight. Uses softmax over normalized
    scores for numerical stability.
    """
    arr = np.array(scores, dtype=float)
    if direction == "minimize":
        # Lower is better — invert so lower score = higher weight
        arr = -arr
    # Shift for numerical stability, then softmax
    arr = arr - arr.max()
    weights = np.exp(arr)
    return weights / weights.sum()


def _combine_predictions(
    predictions: list[np.ndarray],
    weights: np.ndarray | None,
    task_type: str,
    y_test: np.ndarray,
) -> np.ndarray:
    """Combine predictions via averaging/voting.

    If weights is None, uses equal weights (simple average / majority voting).
    """
    k = len(predictions)
    if weights is None:
        weights = np.ones(k) / k

    if task_type == "regression":
        ensemble_preds = np.zeros_like(predictions[0], dtype=float)
        for w, p in zip(weights, predictions):
            ensemble_preds += w * p
    else:
        classes = np.unique(y_test)
        n_samples = len(predictions[0])
        vote_matrix = np.zeros((n_samples, len(classes)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for w, p in zip(weights, predictions):
            for j, pred in enumerate(p):
                if pred in class_to_idx:
                    vote_matrix[j, class_to_idx[pred]] += w
        ensemble_preds = classes[np.argmax(vote_matrix, axis=1)]

    return ensemble_preds


def build_ensemble(
    top_models: list[tuple[str, float]],
    data_info: dict,
    metric_fn,
    task_type: str,
    time_budget: float = 30.0,
) -> tuple[float, str]:
    """Build a weighted ensemble from top-K baseline models.

    Args:
        top_models: List of (model_name, score) sorted best-first.
        data_info: Dict with X_train, X_test, y_train, y_test.
        metric_fn: Metric object with .compute() and .direction.
        task_type: "regression" or "classification".
        time_budget: Maximum seconds for the entire ensemble process (default 30s).

    Returns:
        (ensemble_score, description_string)
    """
    t_start = time.time()
    factories = _get_model_factories(task_type)
    names = []
    scores = []
    predictions = []

    X_train = data_info["X_train"]
    X_test = data_info["X_test"]
    y_train = data_info["y_train"]
    y_test = data_info["y_test"]

    for model_name, score in top_models:
        if time.time() - t_start > time_budget:
            break

        if model_name not in factories:
            continue

        constructor, needs_scaling = factories[model_name]
        model = constructor()

        Xtr, Xte = X_train, X_test
        if needs_scaling:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)

        names.append(model_name)
        scores.append(score)
        predictions.append(preds)

    if len(predictions) < 2:
        elapsed = time.time() - t_start
        if elapsed >= time_budget:
            return None, f"Ensemble timed out ({elapsed:.1f}s > {time_budget:.0f}s budget)"
        return None, "Ensemble needs at least 2 models"

    # Try both simple and weighted ensemble, return the better one
    simple_preds = _combine_predictions(predictions, None, task_type, y_test)
    simple_score = metric_fn.compute(y_test, simple_preds)

    weights = _compute_weights(scores, metric_fn.direction)
    weighted_preds = _combine_predictions(predictions, weights, task_type, y_test)
    weighted_score = metric_fn.compute(y_test, weighted_preds)

    # Pick the better strategy
    weighted_wins = (
        (metric_fn.direction == "maximize" and weighted_score >= simple_score)
        or (metric_fn.direction == "minimize" and weighted_score <= simple_score)
    )

    elapsed = time.time() - t_start
    name_list = ", ".join(names)
    if weighted_wins:
        weight_strs = [f"{n}:{w:.0%}" for n, w in zip(names, weights)]
        desc = f"WeightedEnsemble({', '.join(weight_strs)}) [{elapsed:.1f}s]"
        return weighted_score, desc
    else:
        desc = f"SimpleEnsemble({name_list}) [{elapsed:.1f}s]"
        return simple_score, desc
