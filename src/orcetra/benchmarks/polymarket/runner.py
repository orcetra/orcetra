"""Polymarket benchmark — showcase Orcetra on real prediction market data.

This is Orcetra's "Erdős problem set" — a real-world benchmark that demonstrates
the engine's predictive capabilities on verified market outcomes.
"""
import os
import csv
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DATA_FILE = Path(__file__).parent / "data.csv"


def run_benchmark():
    """Run the Polymarket benchmark using 1,974+ verified predictions."""
    console.print("[bold]🎯 Orcetra Polymarket Benchmark[/bold]")
    console.print()

    if not DATA_FILE.exists():
        console.print("[red]Benchmark data not found. Run the pipeline first.[/red]")
        return

    # Load data
    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    beats = sum(1 for r in rows if r["beat_market"] == "True")
    beat_rate = beats / total if total else 0

    # By category
    from collections import defaultdict
    by_tag = defaultdict(lambda: {"total": 0, "beat": 0, "our_brier": [], "mkt_brier": []})
    for r in rows:
        tag = r["tag"]
        by_tag[tag]["total"] += 1
        if r["beat_market"] == "True":
            by_tag[tag]["beat"] += 1
        by_tag[tag]["our_brier"].append(float(r["our_brier"]))
        by_tag[tag]["mkt_brier"].append(float(r["mkt_brier"]))

    # Summary
    console.print(f"  Total predictions: [bold]{total}[/bold]")
    console.print(f"  Beat rate: [bold green]{beat_rate:.1%}[/bold green] ({beats}/{total})")
    console.print()

    # Category table
    table = Table(title="Performance by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Beat Rate", justify="right", style="green")
    table.add_column("Count", justify="right")
    table.add_column("Our Brier", justify="right")
    table.add_column("Market Brier", justify="right")

    for tag in sorted(by_tag.keys(), key=lambda t: -by_tag[t]["beat"] / max(by_tag[t]["total"], 1)):
        stats = by_tag[tag]
        rate = stats["beat"] / stats["total"] if stats["total"] else 0
        avg_our = sum(stats["our_brier"]) / len(stats["our_brier"])
        avg_mkt = sum(stats["mkt_brier"]) / len(stats["mkt_brier"])
        table.add_row(
            tag,
            f"{rate:.1%}",
            str(stats["total"]),
            f"{avg_our:.4f}",
            f"{avg_mkt:.4f}",
        )

    console.print(table)
    console.print()
    console.print("[dim]Data source: Polymarket resolved predictions (crypto 5-min binary excluded)[/dim]")
    console.print("[dim]Live dashboard: https://orcetra.ai/dashboard.html[/dim]")
