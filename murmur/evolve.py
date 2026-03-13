"""Self-evolution loop for prediction strategies (autoresearch-style)."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from groq import Groq

from .evaluator import format_backtest_results, quick_evaluate
from .fetcher import list_resolved_events_sync
from .models import BacktestResult, EvolutionRound, Event
from .strategy import (
    DEFAULT_ANALYSIS_CODE,
    STRATEGY_TEMPLATE,
    PredictionStrategy,
    load_strategy_from_code,
    get_default_strategy,
)

RESULTS_DIR = Path("results")
EVOLUTION_LOG_FILE = RESULTS_DIR / "evolution_log.json"
BEST_STRATEGY_FILE = RESULTS_DIR / "best_strategy.py"


class EvolutionEngine:
    """Evolves prediction strategies through LLM-driven code modification."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = "llama-4-scout-17b-16e-instruct"
        self.tried_changes: set[str] = set()  # Track tried modifications
        self.evolution_log: list[EvolutionRound] = []
        self.best_score: float = float("inf")
        self.best_strategy_code: str = ""

        # Ensure results directory exists
        RESULTS_DIR.mkdir(exist_ok=True)

        # Load existing evolution log if present
        self._load_evolution_log()

    def _load_evolution_log(self):
        """Load existing evolution log from file."""
        if EVOLUTION_LOG_FILE.exists():
            try:
                with open(EVOLUTION_LOG_FILE) as f:
                    data = json.load(f)
                    self.evolution_log = [
                        EvolutionRound(**r) for r in data.get("rounds", [])
                    ]
                    self.tried_changes = set(data.get("tried_changes", []))
                    self.best_score = data.get("best_score", float("inf"))
                    self.best_strategy_code = data.get("best_strategy_code", "")
            except Exception as e:
                print(f"Error loading evolution log: {e}")

    def _save_evolution_log(self):
        """Save evolution log to file."""
        try:
            data = {
                "rounds": [r.model_dump() for r in self.evolution_log],
                "tried_changes": list(self.tried_changes),
                "best_score": self.best_score,
                "best_strategy_code": self.best_strategy_code,
                "last_updated": datetime.utcnow().isoformat(),
            }

            # Handle datetime serialization
            def serialize(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(EVOLUTION_LOG_FILE, "w") as f:
                json.dump(data, f, indent=2, default=serialize)

            # Also save best strategy as a standalone file
            if self.best_strategy_code:
                with open(BEST_STRATEGY_FILE, "w") as f:
                    f.write(self.best_strategy_code)

        except Exception as e:
            print(f"Error saving evolution log: {e}")

    def get_initial_strategy_code(self) -> str:
        """Get the initial strategy code for evolution."""
        return STRATEGY_TEMPLATE.format(
            version=1,
            analysis_code=DEFAULT_ANALYSIS_CODE,
        )

    def evolve(
        self,
        rounds: int = 10,
        resolved_events: Optional[list[Event]] = None,
        verbose: bool = True,
    ) -> tuple[str, BacktestResult]:
        """Run the evolution loop."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")

        # Fetch resolved events for backtesting if not provided
        if resolved_events is None:
            if verbose:
                print("Fetching resolved events for backtesting...")
            resolved_events = list_resolved_events_sync(limit=100)

        if len(resolved_events) < 5:
            raise ValueError(
                f"Not enough resolved events for backtesting ({len(resolved_events)}). "
                "Need at least 5 events."
            )

        if verbose:
            print(f"Using {len(resolved_events)} resolved events for backtesting")

        # Initialize with current best or default strategy
        if self.best_strategy_code:
            current_code = self.best_strategy_code
        else:
            current_code = self.get_initial_strategy_code()
            self.best_strategy_code = current_code

        # Get initial baseline
        current_strategy = load_strategy_from_code(current_code)
        if current_strategy is None:
            current_strategy = get_default_strategy()
            current_code = self.get_initial_strategy_code()

        if verbose:
            print("Evaluating initial strategy...")

        current_result = quick_evaluate(current_strategy, resolved_events)
        self.best_score = current_result.brier_score

        if verbose:
            print(f"Initial Brier Score: {current_result.brier_score:.4f}")
            print(f"Market baseline: {current_result.market_brier_score:.4f}")
            print()

        # Evolution loop
        for round_num in range(1, rounds + 1):
            if verbose:
                print(f"=== Evolution Round {round_num}/{rounds} ===")

            # Generate new strategy variant
            new_code, changes = self._evolve_strategy(
                current_code, current_result, round_num
            )

            if new_code is None or changes in self.tried_changes:
                if verbose:
                    print("Skipping duplicate or invalid change, generating new variant...")
                continue

            self.tried_changes.add(changes)

            # Load and evaluate new strategy
            new_strategy = load_strategy_from_code(new_code)
            if new_strategy is None:
                if verbose:
                    print("Failed to load new strategy, skipping...")
                continue

            if verbose:
                print(f"Change: {changes[:100]}...")
                print("Evaluating...")

            try:
                new_result = quick_evaluate(new_strategy, resolved_events)
            except Exception as e:
                if verbose:
                    print(f"Evaluation failed: {e}")
                continue

            improvement = current_result.brier_score - new_result.brier_score
            kept = improvement > 0

            # Log this round
            round_log = EvolutionRound(
                round_number=round_num,
                strategy_code=new_code,
                backtest_result=new_result,
                changes_made=changes,
                improvement=improvement,
                kept=kept,
            )
            self.evolution_log.append(round_log)

            if verbose:
                print(f"New Brier: {new_result.brier_score:.4f}")
                print(f"Improvement: {improvement:+.4f}")
                print(f"Beat market rate: {new_result.beat_market_rate:.1%}")

            if kept:
                if verbose:
                    print("KEEPING new strategy!")
                current_code = new_code
                current_result = new_result
                current_strategy = new_strategy

                if new_result.brier_score < self.best_score:
                    self.best_score = new_result.brier_score
                    self.best_strategy_code = new_code
            else:
                if verbose:
                    print("Reverting to previous strategy")

            # Save progress
            self._save_evolution_log()

            if verbose:
                print()

        # Final summary
        if verbose:
            print("=" * 60)
            print("EVOLUTION COMPLETE")
            print("=" * 60)
            print(f"Rounds: {rounds}")
            print(f"Best Brier Score: {self.best_score:.4f}")
            print(f"Improvements kept: {sum(1 for r in self.evolution_log[-rounds:] if r.kept)}")
            print()
            print(format_backtest_results(current_result))

        return self.best_strategy_code, current_result

    def _evolve_strategy(
        self, current_code: str, current_result: BacktestResult, round_num: int
    ) -> tuple[Optional[str], str]:
        """Use LLM to evolve the strategy code."""
        # Build failure analysis from worst predictions
        failure_analysis = self._analyze_failures(current_result)

        prompt = f"""You are evolving a prediction market strategy. Your goal is to improve its Brier score (lower is better).

CURRENT STRATEGY CODE:
```python
{current_code}
```

CURRENT PERFORMANCE:
- Brier Score: {current_result.brier_score:.4f}
- Market Baseline: {current_result.market_brier_score:.4f}
- Beat Market Rate: {current_result.beat_market_rate:.1%}
- Accuracy: {current_result.accuracy_50:.1%}

FAILURE ANALYSIS (worst predictions):
{failure_analysis}

PREVIOUSLY TRIED CHANGES (DO NOT REPEAT):
{chr(10).join(list(self.tried_changes)[-10:])}

RULES:
1. Change ONLY ONE thing per round - this is critical!
2. Focus on the analysis_results computation in the EVOLVED ANALYSIS LOGIC section
3. You can modify:
   - How momentum is calculated
   - How news sentiment is weighted
   - Add new signals (e.g., volume analysis, time decay)
   - Change how signals are combined
   - Add edge case handling
4. Keep changes small and testable
5. DO NOT repeat previously tried changes

Output the complete modified strategy code (the full class) and describe the change.

Format:
CHANGE: <brief description of what you changed>
```python
<complete strategy class code>
```"""

        try:
            client = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000,
            )

            content = response.choices[0].message.content.strip()

            # Extract change description
            change_line = ""
            for line in content.split("\n"):
                if line.startswith("CHANGE:"):
                    change_line = line.replace("CHANGE:", "").strip()
                    break

            # Extract code
            import re
            code_match = re.search(r"```python\s*([\s\S]*?)```", content)
            if code_match:
                new_code = code_match.group(1).strip()

                # Add required imports if not present
                imports = """
import json
import os
import re
from typing import Optional
from groq import Groq
from murmur.models import Event, PricePoint, NewsSignal, OrderBook, Prediction
from murmur.strategy import PredictionStrategy
"""
                if "import" not in new_code:
                    new_code = imports + "\n" + new_code

                return new_code, change_line

        except Exception as e:
            print(f"Error in LLM evolution: {e}")

        return None, ""

    def _analyze_failures(self, result: BacktestResult, top_n: int = 5) -> str:
        """Analyze worst predictions to guide evolution."""
        if not result.per_event_results:
            return "No event results available"

        # Sort by Brier score (worst first)
        worst = sorted(
            result.per_event_results, key=lambda x: x.brier_score, reverse=True
        )[:top_n]

        lines = []
        for i, r in enumerate(worst, 1):
            lines.append(
                f"{i}. '{r.event_title[:50]}...'\n"
                f"   Predicted: {r.predicted_prob:.2f}, Actual: {r.actual_outcome:.0f}, "
                f"Market: {r.market_prob:.2f}\n"
                f"   Brier: {r.brier_score:.4f}, Reason: {r.reasoning[:100]}..."
            )

        return "\n".join(lines)


def run_evolution(
    rounds: int = 10, verbose: bool = True
) -> tuple[str, BacktestResult]:
    """Run the evolution loop (convenience function)."""
    engine = EvolutionEngine()
    return engine.evolve(rounds=rounds, verbose=verbose)


def load_best_strategy() -> Optional[PredictionStrategy]:
    """Load the best evolved strategy from disk."""
    if BEST_STRATEGY_FILE.exists():
        try:
            with open(BEST_STRATEGY_FILE) as f:
                code = f.read()
            return load_strategy_from_code(code)
        except Exception as e:
            print(f"Error loading best strategy: {e}")

    return None
