#!/usr/bin/env python3
"""Analyze multi_framework_*.jsonl results and summarize win rates."""
import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    """Load and deduplicate results from all multi_framework JSONL files."""
    by_id = {}
    for f in sorted(RESULTS_DIR.glob("multi_framework_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get("status") == "success":
                    by_id[r["dataset_id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    return list(by_id.values())


def print_summary(label, results):
    budget = results[0]["budget_sec"] if results else 60
    wins = Counter(r["winner"] for r in results)
    n = len(results)
    print(f"\n{'=' * 60}")
    print(f"{label} ({n} datasets, {budget}s budget)")
    print(f"{'=' * 60}")
    for name in ["orcetra", "flaml", "autogluon", "tie"]:
        c = wins.get(name, 0)
        pct = c / n * 100 if n else 0
        print(f"  {name:12s}: {c:4d} ({pct:.1f}%)")


def analyze_regression_losses(results):
    """Show regression datasets where Orcetra loses, grouped by winner."""
    reg = [r for r in results if r["task_type"] == "regression"]
    losses = [r for r in reg if r["winner"] not in ("orcetra", "tie")]

    if not losses:
        print("\nOrcetra wins or ties on all regression datasets!")
        return

    # Group by winner
    by_winner = {}
    for r in losses:
        by_winner.setdefault(r["winner"], []).append(r)

    print(f"\n{'=' * 80}")
    print(f"REGRESSION LOSSES — Orcetra loses on {len(losses)}/{len(reg)} regression datasets")
    print(f"{'=' * 80}")

    for winner, datasets in sorted(by_winner.items(), key=lambda x: -len(x[1])):
        print(f"\n  Lost to {winner.upper()} ({len(datasets)} datasets):")
        print(f"  {'Dataset':<35s} {'Samples':>8s} {'Feats':>6s} "
              f"{'Orcetra':>10s} {'Winner':>10s} {'Gap%':>7s} {'OrcModel':<25s} {'WinModel':<25s}")
        print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*7} {'-'*25} {'-'*25}")

        for r in sorted(datasets, key=lambda x: x.get("name", "")):
            scores = r.get("scores", {})
            orc = scores.get("orcetra")
            win = scores.get(winner)
            models = r.get("models", {})

            # For regression (lower MSE = better), gap = how much worse orcetra is
            if orc is not None and win is not None and win != 0:
                gap_pct = (orc - win) / abs(win) * 100
            else:
                gap_pct = 0

            print(f"  {r.get('name', '?'):<35s} {r.get('n_samples', '?'):>8} {r.get('n_features', '?'):>6} "
                  f"{orc:>10.4f} {win:>10.4f} {gap_pct:>+6.1f}% "
                  f"{str(models.get('orcetra', '?'))[:25]:<25s} {str(models.get(winner, '?'))[:25]:<25s}")

    # Summary stats: size/feature distribution of wins vs losses
    wins = [r for r in reg if r["winner"] == "orcetra"]
    print(f"\n{'=' * 80}")
    print("PATTERN ANALYSIS — Wins vs Losses")
    print(f"{'=' * 80}")
    for label, subset in [("Orcetra WINS", wins), ("Orcetra LOSSES", losses)]:
        if not subset:
            continue
        samples = [r.get("n_samples", 0) for r in subset if r.get("n_samples")]
        feats = [r.get("n_features", 0) for r in subset if r.get("n_features")]
        print(f"\n  {label} ({len(subset)} datasets):")
        if samples:
            print(f"    Samples:  median={sorted(samples)[len(samples)//2]:,}  "
                  f"min={min(samples):,}  max={max(samples):,}")
        if feats:
            print(f"    Features: median={sorted(feats)[len(feats)//2]:,}  "
                  f"min={min(feats):,}  max={max(feats):,}")

        # What models beat orcetra?
        if label == "Orcetra LOSSES":
            win_models = Counter()
            for r in subset:
                w = r["winner"]
                m = r.get("models", {}).get(w, "?")
                win_models[m] += 1
            print(f"    Winning models: {dict(win_models.most_common(10))}")

        # What models did orcetra use?
        orc_models = Counter(str(r.get("models", {}).get("orcetra", "?")) for r in subset)
        print(f"    Orcetra models: {dict(orc_models.most_common(10))}")

    print()


def main():
    results = load_results()
    if not results:
        print("No successful results found in multi_framework_*.jsonl files.")
        return

    # Overall
    print_summary("MULTI-FRAMEWORK RESULTS", results)

    # By task type
    for task in ["classification", "regression"]:
        subset = [r for r in results if r["task_type"] == task]
        if subset:
            print_summary(f"  {task.upper()}", subset)

    # Detailed regression loss analysis
    analyze_regression_losses(results)


if __name__ == "__main__":
    main()
