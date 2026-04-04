"""Weighted ensemble of top-K diverse models.

Standalone module for building ensembles from search results.
Supports both weighted and simple (equal-weight) strategies,
returning whichever scores better.

Importable for both the CLI loop and benchmark scripts.
"""
import time
import warnings
import numpy as np
from sklearn.ensemble import VotingRegressor, VotingClassifier


def compute_weights(scores: list[float], direction: str) -> list[float]:
    """Compute ensemble weights from scores (softmax over normalized scores).

    Better-scoring models get higher weight. Uses softmax for numerical stability.

    Args:
        scores: List of metric scores for each model.
        direction: "minimize" or "maximize".

    Returns:
        List of normalized weights summing to 1.0.
    """
    arr = np.array(scores, dtype=float)
    if direction == "minimize":
        arr = -arr
    arr = arr - arr.max()
    weights = np.exp(arr)
    weights = weights / weights.sum()
    return weights.tolist()


def select_diverse_models(proposals_with_scores, direction: str, top_k: int = 3):
    """Select top-K diverse models from a list of (proposal, score) pairs.

    Diversity is enforced by picking at most one model per class name.

    Args:
        proposals_with_scores: List of (proposal, score) tuples.
            Each proposal must have .model and .preprocessor attributes.
        direction: "minimize" or "maximize".
        top_k: Maximum number of models to select.

    Returns:
        List of (proposal, score) tuples, or empty list if fewer than 2 eligible.
    """
    # Only use proposals without preprocessors (ensemble re-fits on raw data)
    eligible = [(p, s) for p, s in proposals_with_scores if p.preprocessor is None]
    if len(eligible) < 2:
        return []

    if direction == "minimize":
        sorted_proposals = sorted(eligible, key=lambda x: x[1])
    else:
        sorted_proposals = sorted(eligible, key=lambda x: -x[1])

    selected = []
    seen_types = set()
    for proposal, score in sorted_proposals:
        model_type = type(proposal.model).__name__
        if model_type not in seen_types:
            selected.append((proposal, score))
            seen_types.add(model_type)
        if len(selected) >= top_k:
            break

    if len(selected) < 2:
        return []
    return selected


def build_ensemble(selected, data_info, metric_fn, time_budget: float = 30.0):
    """Build a weighted ensemble from selected (proposal, score) pairs.

    Tries both simple (equal-weight) and inverse-error-weighted strategies,
    returning whichever scores better.

    Args:
        selected: List of (proposal, score) tuples (from select_diverse_models).
        data_info: Dict with X_train, X_test, y_train, y_test, task_type.
        metric_fn: Metric object with .compute() and .direction.
        time_budget: Maximum seconds for the ensemble process (default 30s).

    Returns:
        (ensemble_score, description_string) or None if ensemble failed.
    """
    t_start = time.time()
    task_type = data_info.get("task_type", "regression")
    direction = metric_fn.direction
    scores = [s for _, s in selected]

    # Build both simple and weighted ensembles, return the better one
    simple_weights = [1.0 / len(selected)] * len(selected)
    weighted = compute_weights(scores, direction)

    results = []
    for label, weights in [("SimpleEnsemble", simple_weights), ("WeightedEnsemble", weighted)]:
        if time.time() - t_start > time_budget:
            break

        try:
            estimators = [(f"m{i}", p.model) for i, (p, _) in enumerate(selected)]

            if task_type == "regression":
                ensemble = VotingRegressor(estimators=estimators, weights=weights, n_jobs=1)
            else:
                ensemble = VotingClassifier(estimators=estimators, weights=weights, voting="soft", n_jobs=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ensemble.fit(data_info["X_train"], data_info["y_train"])
                preds = ensemble.predict(data_info["X_test"])

            ens_score = metric_fn.compute(data_info["y_test"], preds)
            names = "+".join(type(p.model).__name__ for p, _ in selected)
            w_str = ",".join(f"{w:.2f}" for w in weights)
            desc = f"{label}({names}, w=[{w_str}])"
            results.append((ens_score, desc))
        except Exception:
            continue

    if not results:
        return None

    # Pick the better result
    if len(results) == 1:
        return results[0]

    score_a, desc_a = results[0]
    score_b, desc_b = results[1]
    if direction == "minimize":
        return results[0] if score_a <= score_b else results[1]
    else:
        return results[0] if score_a >= score_b else results[1]


def try_ensemble(cache, data_info, metric_fn, console, time_budget: float = 30.0):
    """High-level entry point: select diverse models from cache and build ensemble.

    This is the main function called from the prediction loop.

    Args:
        cache: StrategyCache with .proposals list of (proposal, score).
        data_info: Dict with X_train, X_test, y_train, y_test, task_type.
        metric_fn: Metric object with .compute() and .direction.
        console: Rich console for output.
        time_budget: Maximum seconds for the ensemble process.

    Returns:
        (ensemble_score, description_string) or None.
    """
    if len(cache.proposals) < 3:
        return None

    selected = select_diverse_models(cache.proposals, metric_fn.direction, top_k=3)
    if not selected:
        return None

    console.print(f"\n[bold]Step 4:[/bold] Trying ensemble of top-{len(selected)} diverse models...")
    for p, s in selected:
        console.print(f"  {p.description}: {s:.4f}")

    result = build_ensemble(selected, data_info, metric_fn, time_budget=time_budget)
    if result is None:
        console.print("  [red]Ensemble failed[/red]")
    return result
