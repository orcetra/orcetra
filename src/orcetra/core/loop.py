"""
AutoResearch prediction loop with parallel evaluation and strategy caching.

Inspired by Karpathy's autoresearch: an AI agent iteratively improves
a prediction pipeline by proposing modifications, testing them, and
keeping improvements.
"""
import time
import hashlib
import pandas as pd
import numpy as np
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress

from ..data.loader import analyze_and_load
from ..models.registry import get_baselines
from ..metrics.base import get_metric

console = Console()

# ── Parallel evaluation helper ────────────────────────────────────────

def _evaluate_proposal(proposal, data_info, metric_fn):
    """Evaluate a single proposal (picklable for parallel exec)."""
    try:
        score = proposal.evaluate(data_info, metric_fn)
        return (proposal, score, None)
    except Exception as e:
        return (proposal, None, str(e))


# ── Post-search Weighted Ensemble ─────────────────────────────────────

def _try_weighted_ensemble(cache, data_info, metric_fn, current_best, console):
    """Build a weighted ensemble from the top-3 diverse models found during search.
    
    Returns (score, description) if ensemble was built, None otherwise.
    """
    import warnings
    from sklearn.ensemble import VotingRegressor, VotingClassifier
    
    if len(cache.proposals) < 3:
        return None
    
    task_type = data_info.get("task_type", "regression")
    direction = metric_fn.direction
    
    # Only use proposals without preprocessors (ensemble re-fits on raw data)
    eligible = [(p, s) for p, s in cache.proposals if p.preprocessor is None]
    if len(eligible) < 3:
        return None
    
    # Sort proposals by score (best first)
    if direction == "minimize":
        sorted_proposals = sorted(eligible, key=lambda x: x[1])
    else:
        sorted_proposals = sorted(eligible, key=lambda x: -x[1])
    
    # Pick top-3 diverse models (different model classes)
    selected = []
    seen_types = set()
    for proposal, score in sorted_proposals:
        # Use the model class name as diversity key
        model_type = type(proposal.model).__name__
        if model_type not in seen_types:
            selected.append((proposal, score))
            seen_types.add(model_type)
        if len(selected) >= 3:
            break
    
    if len(selected) < 2:
        return None
    
    console.print(f"\n[bold]Step 4:[/bold] Trying weighted ensemble of top-{len(selected)} models...")
    for p, s in selected:
        console.print(f"  {p.description}: {s:.4f}")
    
    # Build ensemble with inverse-error weighting
    # For minimize: weight = 1/score (lower score = higher weight)
    # For maximize: weight = score (higher score = higher weight)
    scores = [s for _, s in selected]
    if direction == "minimize":
        # Avoid division by zero
        min_score = min(scores)
        if min_score <= 0:
            weights = [1.0] * len(selected)
        else:
            weights = [1.0 / s for s in scores]
    else:
        total = sum(scores)
        if total <= 0:
            weights = [1.0] * len(selected)
        else:
            weights = scores
    
    # Normalize weights
    w_total = sum(weights)
    weights = [w / w_total for w in weights]
    
    try:
        estimators = [(f"m{i}", p.model) for i, (p, _) in enumerate(selected)]
        
        if task_type == "regression":
            ensemble = VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)
        else:
            ensemble = VotingClassifier(estimators=estimators, weights=weights, voting="soft", n_jobs=-1)
        
        X_train = data_info["X_train"]
        X_test = data_info["X_test"]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble.fit(X_train, data_info["y_train"])
            preds = ensemble.predict(X_test)
        
        ens_score = metric_fn.compute(data_info["y_test"], preds)
        names = "+".join(type(p.model).__name__ for p, _ in selected)
        w_str = ",".join(f"{w:.2f}" for w in weights)
        description = f"WeightedEnsemble({names}, w=[{w_str}])"
        
        return (ens_score, description)
    except Exception as e:
        console.print(f"  [red]Ensemble failed: {e}[/red]")
        return None


# ── Strategy Cache ────────────────────────────────────────────────────

class StrategyCache:
    """Dedup cache keyed by (model_class, sorted_params). Stores all results."""
    
    def __init__(self):
        self._seen = {}  # key -> score
        self.all_results = []  # list of (description, score, improved)
        self.proposals = []  # list of (proposal, score) for ensemble building
    
    def _key(self, proposal):
        """Hash based on model class + description (covers params)."""
        return hashlib.md5(proposal.description.encode()).hexdigest()
    
    def is_duplicate(self, proposal) -> bool:
        return self._key(proposal) in self._seen
    
    def record(self, proposal, score, improved=False):
        k = self._key(proposal)
        self._seen[k] = score
        self.all_results.append({
            "description": proposal.description,
            "score": score,
            "improved": improved,
        })
        self.proposals.append((proposal, score))
    
    @property
    def tried_count(self):
        return len(self._seen)
    
    def top_k(self, k=5, direction="minimize"):
        """Return top-k results sorted by score."""
        reverse = direction == "maximize"
        valid = [r for r in self.all_results if r["score"] is not None]
        return sorted(valid, key=lambda x: x["score"], reverse=reverse)[:k]


# ── Main Loop ─────────────────────────────────────────────────────────

def run_prediction(
    data_path: str,
    target: str,
    budget: str = "10min",
    metric: str = "auto",
    parallel: int = 0,  # 0 = auto-detect
) -> dict:
    """
    Main entry point for automated prediction.
    
    1. Load and analyze data
    2. Run baseline models
    3. AutoResearch loop: parallel batch evaluation with strategy cache
    4. Return best result
    """
    import os
    budget_seconds = parse_budget(budget)
    start_time = time.time()
    
    # Auto-detect parallelism
    if parallel <= 0:
        parallel = min(os.cpu_count() or 4, 8)
    
    # Step 1: Analyze data
    console.print("[bold]Step 1:[/bold] Analyzing data...")
    data_info = analyze_and_load(data_path, target)
    console.print(f"  Shape: {data_info['shape']}")
    console.print(f"  Task: {data_info['task_type']}")
    console.print(f"  Features: {data_info['n_features']}")
    
    # Step 2: Select metric
    if metric == "auto":
        if data_info["task_type"] == "regression":
            y_all = pd.concat([data_info["y_train"], data_info["y_test"]])
            if (y_all > 0).all():
                try:
                    from scipy.stats import skew
                    if skew(y_all) > 1.0:
                        metric = "rmsle"
                        console.print(f"  [cyan]Auto-selected RMSLE (positive right-skewed target)[/cyan]")
                    else:
                        metric = "mse"
                except ImportError:
                    metric = "mse"
            else:
                metric = "mse"
        else:
            metric = "accuracy"
    metric_fn = get_metric(metric)
    console.print(f"  Metric: {metric_fn.name} ({metric_fn.direction})")
    
    # Step 3: Run baselines
    console.print("\n[bold]Step 2:[/bold] Running baselines...")
    baselines = get_baselines(data_info["task_type"])
    best_score = None
    best_model = None
    baseline_best_score = None
    
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
                baseline_best_score = score
            console.print(f"  {model_name}: {score:.4f} {'⭐' if is_better else ''}")
        except Exception as e:
            console.print(f"  {model_name}: [red]failed ({e})[/red]")
    
    console.print(f"\n[bold yellow]Best baseline: {best_model} = {best_score:.4f}[/bold yellow]")
    
    # Step 4: AutoResearch loop — parallel batches with strategy cache
    console.print(f"\n[bold]Step 3:[/bold] AutoResearch loop (budget: {budget}, {parallel}x parallel)...")
    
    # Initialize agent
    agent = None
    agent_type = "random"
    try:
        if os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            from .llm_agent import LLMSearchAgent
            provider = "groq" if os.environ.get("GROQ_API_KEY") else "openai"
            agent = LLMSearchAgent(task_type=data_info["task_type"], provider=provider)
            agent_type = "llm"
            console.print(f"  Agent: [bold cyan]LLM-guided[/bold cyan] ({provider})")
    except Exception:
        pass
    
    if agent is None:
        from .agent import RandomSearchAgent
        agent = RandomSearchAgent(task_type=data_info["task_type"])
        console.print(f"  Agent: [dim]Random search[/dim] (set GROQ_API_KEY for LLM-guided)")
    
    cache = StrategyCache()
    iteration = 0
    improvements = 0
    batch_num = 0
    
    last_proposal_desc = ""
    last_score = None
    last_improved = False
    
    # Batch size: generate N proposals, deduplicate, evaluate in parallel
    batch_size = parallel * 2  # Generate more than workers to account for dedup
    
    while time.time() - start_time < budget_seconds:
        batch_num += 1
        
        # Generate batch of proposals (skip duplicates)
        proposals = []
        attempts = 0
        while len(proposals) < batch_size and attempts < batch_size * 3:
            attempts += 1
            state = {
                "best_score": best_score,
                "best_model": best_model,
                "iteration": iteration + len(proposals) + 1,
                "task_type": data_info["task_type"],
                "metric_direction": metric_fn.direction,
                "data_summary": f"{data_info['shape'][0]} rows, {data_info['n_features']} features",
                "last_proposal": last_proposal_desc,
                "last_score": last_score,
                "last_improved": last_improved,
            }
            proposal = agent.propose(state)
            if not cache.is_duplicate(proposal):
                proposals.append(proposal)
        
        if not proposals:
            console.print(f"  [yellow]Batch {batch_num}: all proposals duplicated, stopping[/yellow]")
            break
        
        # Parallel evaluation
        results = []
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_evaluate_proposal, p, data_info, metric_fn): p
                for p in proposals
            }
            for future in as_completed(futures):
                results.append(future.result())
        
        # Process results
        batch_improvements = 0
        for proposal, score, error in results:
            iteration += 1
            
            if error or score is None:
                if iteration <= 5:
                    console.print(f"  [red]#{iteration} {proposal.description}: failed ({error[:50] if error else 'unknown'})[/red]")
                continue
            
            cache.record(proposal, score)
            
            is_better = (
                (metric_fn.direction == "minimize" and score < best_score)
                or (metric_fn.direction == "maximize" and score > best_score)
            )
            
            if is_better:
                improvements += 1
                batch_improvements += 1
                old_best = best_score
                best_score = score
                best_model = proposal.description
                last_improved = True
                last_proposal_desc = proposal.description
                last_score = score
                
                beats_baseline = (
                    (metric_fn.direction == "minimize" and score < baseline_best_score)
                    or (metric_fn.direction == "maximize" and score > baseline_best_score)
                )
                improvement_pct = abs((score - old_best) / old_best * 100) if old_best != 0 else 0
                
                if beats_baseline:
                    console.print(f"  [bold green]#{iteration} 🎯 {proposal.description}: {score:.4f} (+{improvement_pct:.1f}% vs baseline!)[/bold green]")
                else:
                    console.print(f"  [green]#{iteration} ⭐ {proposal.description}: {score:.4f} (+{improvement_pct:.1f}%)[/green]")
            else:
                last_improved = False
                last_proposal_desc = proposal.description
                last_score = score
        
        # Batch summary (compact)
        if batch_num <= 3 or batch_num % 5 == 0:
            elapsed_so_far = time.time() - start_time
            console.print(f"  [dim]  batch {batch_num}: {len(proposals)} evaluated, {batch_improvements} improved, best={best_score:.4f} ({elapsed_so_far:.0f}s)[/dim]")
    
    console.print(f"  Completed {iteration} strategies ({cache.tried_count} unique), {improvements} improvements")
    
    # Step 5: Post-search weighted ensemble — try top-3 diverse models
    ensemble_score = _try_weighted_ensemble(cache, data_info, metric_fn, best_score, console)
    if ensemble_score is not None:
        ens_score, ens_desc = ensemble_score
        is_better = (
            (metric_fn.direction == "minimize" and ens_score < best_score)
            or (metric_fn.direction == "maximize" and ens_score > best_score)
        )
        if is_better:
            improvement_pct = abs((ens_score - best_score) / best_score * 100) if best_score != 0 else 0
            console.print(f"  [bold green]\U0001f3c6 Ensemble beat best single model! {ens_desc}: {ens_score:.4f} (+{improvement_pct:.1f}%)[/bold green]")
            best_score = ens_score
            best_model = ens_desc
            improvements += 1
        else:
            console.print(f"  [dim]Ensemble ({ens_desc}: {ens_score:.4f}) did not beat best single model[/dim]")
    
    elapsed = time.time() - start_time
    
    # Show top-5 from cache
    top5 = cache.top_k(5, metric_fn.direction)
    if top5:
        console.print(f"\n[bold]Top 5 strategies:[/bold]")
        for i, r in enumerate(top5):
            marker = "🏆" if i == 0 else " "
            console.print(f"  {marker} {r['score']:.4f} — {r['description']}")
    
    # Final summary
    improvement_over_baseline = None
    if baseline_best_score is not None:
        if metric_fn.direction == "minimize":
            improvement_pct = (baseline_best_score - best_score) / baseline_best_score * 100
            beats_baseline = best_score < baseline_best_score
        else:
            improvement_pct = (best_score - baseline_best_score) / baseline_best_score * 100
            beats_baseline = best_score > baseline_best_score
        improvement_over_baseline = improvement_pct
        
        if beats_baseline:
            console.print(f"\n[bold green]🎯 Found better model! {improvement_pct:+.1f}% improvement over baseline[/bold green]")
        else:
            console.print(f"\n[bold yellow]📊 Best result matches baseline performance[/bold yellow]")
    
    console.print(f"\n[bold]Final Result:[/bold] {best_model} = {best_score:.4f}")
    
    return {
        "best_model": best_model,
        "best_score": best_score,
        "baseline_score": baseline_best_score,
        "improvement_over_baseline": improvement_over_baseline,
        "metric_name": metric_fn.name,
        "iterations": iteration,
        "unique_strategies": cache.tried_count,
        "improvements": improvements,
        "elapsed": elapsed,
        "task_type": data_info["task_type"],
        "top_5": cache.top_k(5, metric_fn.direction),
    }


def parse_budget(budget: str) -> float:
    """Parse budget string like '10min', '1h', '30s' to seconds."""
    budget = budget.strip().lower()
    if budget.endswith("min"):
        return float(budget[:-3]) * 60
    elif budget.endswith("h"):
        return float(budget[:-1]) * 3600
    elif budget.endswith("s"):
        return float(budget[:-1])
    else:
        return float(budget) * 60  # default to minutes
