import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
BATCH_FILE = RESULTS_DIR / "batch_predictions.json"
CHECK_LOG = RESULTS_DIR / "check_log.json"
WATCHLIST_FILE = RESULTS_DIR / "watchlist.json"


def _load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default if default is not None else {}


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


@click.group()
@click.version_option()
def main():
    """Orcetra — AI-powered automated prediction engine.

    Give it data, it finds the best forecast.
    """
    pass


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column to predict")
@click.option("--budget", "-b", default="10min", help="Time budget for AutoResearch loop")
@click.option("--metric", "-m", default="auto", help="Evaluation metric (auto/mse/mae/accuracy/f1/brier)")
@click.option("--output", "-o", default=None, help="Output path for predictions")
def predict(data_path, target, budget, metric, output):
    """Run automated prediction on a dataset.

    Example: orcetra predict housing.csv --target price --budget 30min
    """
    from .core.loop import run_prediction
    from . import __version__
    console.print(f"[bold blue]🎯 Orcetra v{__version__}[/bold blue]")
    console.print(f"  Data: {data_path}")
    console.print(f"  Target: {target}")
    console.print(f"  Budget: {budget}")
    console.print(f"  Metric: {metric}")
    console.print()

    result = run_prediction(data_path, target, budget=budget, metric=metric)

    # Print results summary
    console.print(f"\n[bold green]✅ Best result:[/bold green]")
    console.print(f"  Model: {result['best_model']}")
    console.print(f"  {result['metric_name']}: {result['best_score']:.4f}")
    console.print(f"  Iterations: {result['iterations']}")
    console.print(f"  Time: {result['elapsed']:.1f}s")


@main.command()
@click.argument("benchmark_name", default="polymarket")
def benchmark(benchmark_name):
    """Run a built-in benchmark.

    Example: orcetra benchmark polymarket
    """
    console.print(f"[bold]Running benchmark: {benchmark_name}[/bold]")
    if benchmark_name == "polymarket":
        from .benchmarks.polymarket.runner import run_benchmark
        run_benchmark()
    else:
        console.print(f"[red]Unknown benchmark: {benchmark_name}[/red]")


# ── Stats Command ────────────────────────────────────────────────────

@main.command()
@click.option("--category", "-c", default=None, help="Filter by category (e.g. sports, politics)")
@click.option("--limit", "-n", default=10, help="Number of recent results to show")
def stats(category, limit):
    """Show prediction performance analytics.

    Displays beat rate, Brier scores, P&L simulation, category breakdown,
    calibration buckets, and recent resolved predictions.

    Example: orcetra stats
             orcetra stats --category politics
    """
    log = _load_json(CHECK_LOG, {"checks": []})
    checks = log.get("checks", [])

    if not checks:
        console.print("[yellow]No resolved predictions yet. Run auto_check.py first.[/yellow]")
        return

    # Filter out crypto binary bets
    checks = [c for c in checks if not (c.get("tag") == "crypto" and "Up or Down" in c.get("question", ""))]

    if category:
        checks = [c for c in checks if c.get("tag", "").lower() == category.lower()]
        if not checks:
            console.print(f"[yellow]No results for category '{category}'[/yellow]")
            return

    total = len(checks)
    beat = sum(1 for c in checks if c["beat_market"])
    avg_our = sum(c["our_brier"] for c in checks) / total
    avg_mkt = sum(c["mkt_brier"] for c in checks) / total
    bets = [c for c in checks if c.get("bet_made")]
    wins = sum(1 for c in bets if c.get("pnl", 0) > 0)
    total_pnl = sum(c.get("pnl", 0) for c in bets)

    # Header panel
    title = f"Orcetra Performance — {category or 'All Categories'}"
    beat_color = "green" if beat / total > 0.55 else "yellow" if beat / total > 0.45 else "red"
    brier_color = "green" if avg_our < avg_mkt else "red"

    header = (
        f"[bold]Resolved predictions:[/bold] {total}\n"
        f"[bold]Beat market rate:[/bold]     [{beat_color}]{beat}/{total} ({beat/total:.1%})[/{beat_color}]\n"
        f"[bold]Avg Brier (ours):[/bold]     [{brier_color}]{avg_our:.4f}[/{brier_color}]\n"
        f"[bold]Avg Brier (market):[/bold]   {avg_mkt:.4f}\n"
        f"[bold]Brier advantage:[/bold]      [{brier_color}]{avg_mkt - avg_our:+.4f}[/{brier_color}]"
    )
    if bets:
        win_rate = wins / len(bets) if bets else 0
        header += (
            f"\n\n[bold]Betting simulation ($2/bet):[/bold]\n"
            f"  Bets placed: {len(bets)} | Won: {wins} ({win_rate:.0%}) | P&L: ${total_pnl * 2:+.2f}"
        )
    console.print(Panel(header, title=title, border_style="blue"))

    # Category breakdown table
    if not category:
        by_tag = {}
        for c in checks:
            t = c.get("tag", "unknown")
            if t not in by_tag:
                by_tag[t] = {"n": 0, "beat": 0, "our_b": 0.0, "mkt_b": 0.0}
            by_tag[t]["n"] += 1
            by_tag[t]["beat"] += 1 if c["beat_market"] else 0
            by_tag[t]["our_b"] += c["our_brier"]
            by_tag[t]["mkt_b"] += c["mkt_brier"]

        cat_table = Table(title="By Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Resolved", justify="right")
        cat_table.add_column("Beat Rate", justify="right")
        cat_table.add_column("Our Brier", justify="right")
        cat_table.add_column("Mkt Brier", justify="right")
        cat_table.add_column("Edge", justify="right")

        for t in sorted(by_tag, key=lambda x: -by_tag[x]["n"]):
            s = by_tag[t]
            rate = s["beat"] / s["n"]
            our_b = s["our_b"] / s["n"]
            mkt_b = s["mkt_b"] / s["n"]
            edge = mkt_b - our_b
            rate_color = "green" if rate > 0.55 else "yellow" if rate > 0.45 else "red"
            edge_color = "green" if edge > 0 else "red"
            cat_table.add_row(
                t, str(s["n"]),
                f"[{rate_color}]{rate:.0%}[/{rate_color}]",
                f"{our_b:.4f}", f"{mkt_b:.4f}",
                f"[{edge_color}]{edge:+.4f}[/{edge_color}]",
            )
        console.print(cat_table)

    # Calibration buckets
    buckets = {}
    for c in checks:
        bucket = round(c["our_prediction"] * 10) / 10  # round to nearest 0.1
        bucket = max(0.0, min(1.0, bucket))
        key = f"{bucket:.0%}"
        if key not in buckets:
            buckets[key] = {"n": 0, "actual_sum": 0.0, "predicted_sum": 0.0}
        buckets[key]["n"] += 1
        buckets[key]["actual_sum"] += c["actual"]
        buckets[key]["predicted_sum"] += c["our_prediction"]

    if buckets:
        cal_table = Table(title="Calibration (predicted vs actual)")
        cal_table.add_column("Predicted", style="cyan")
        cal_table.add_column("Count", justify="right")
        cal_table.add_column("Actual Rate", justify="right")
        cal_table.add_column("Calibration", justify="right")

        for key in sorted(buckets.keys()):
            b = buckets[key]
            actual_rate = b["actual_sum"] / b["n"]
            avg_pred = b["predicted_sum"] / b["n"]
            diff = actual_rate - avg_pred
            diff_color = "green" if abs(diff) < 0.1 else "yellow" if abs(diff) < 0.2 else "red"
            cal_table.add_row(
                key, str(b["n"]),
                f"{actual_rate:.0%}",
                f"[{diff_color}]{diff:+.0%}[/{diff_color}]",
            )
        console.print(cal_table)

    # Recent results
    recent = sorted(checks, key=lambda c: c.get("checked_at", ""), reverse=True)[:limit]
    if recent:
        rec_table = Table(title=f"Recent Results (last {limit})")
        rec_table.add_column("Question", max_width=40)
        rec_table.add_column("Tag", style="dim")
        rec_table.add_column("Ours", justify="right")
        rec_table.add_column("Market", justify="right")
        rec_table.add_column("Actual", justify="right")
        rec_table.add_column("Result", justify="center")

        for c in recent:
            result_icon = "[green]WIN[/green]" if c["beat_market"] else "[red]LOSS[/red]"
            rec_table.add_row(
                c["question"][:40],
                c.get("tag", "?"),
                f"{c['our_prediction']:.2f}",
                f"{c['market_price']:.2f}",
                f"{c['actual']:.0f}",
                result_icon,
            )
        console.print(rec_table)


# ── Search Command ───────────────────────────────────────────────────

@main.command()
@click.argument("query", default="")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.option("--min-price", default=0.0, help="Minimum YES price (0-1)")
@click.option("--max-price", default=1.0, help="Maximum YES price (0-1)")
@click.option("--min-volume", default=0.0, help="Minimum volume in USD")
@click.option("--limit", "-n", default=20, help="Max results to show")
@click.option("--predicted/--no-predicted", default=False, help="Only show markets we have predictions for")
def search(query, category, min_price, max_price, min_volume, limit, predicted):
    """Search and filter prediction markets.

    Search across active markets and existing predictions by keyword,
    category, price range, or volume.

    Examples:
        orcetra search trump
        orcetra search --category politics --min-volume 10000
        orcetra search "oil price" --predicted
    """
    batch = _load_json(BATCH_FILE, {"predictions": {}})
    preds = batch.get("predictions", {})

    if not preds:
        console.print("[yellow]No predictions loaded. Run batch_tracker.py predict first.[/yellow]")
        return

    results = []
    query_lower = query.lower()

    for key, p in preds.items():
        # Keyword filter
        if query_lower and query_lower not in p.get("question", "").lower() \
                and query_lower not in p.get("event_title", "").lower() \
                and query_lower not in p.get("tag", "").lower():
            continue

        # Category filter
        if category and p.get("tag", "").lower() != category.lower():
            continue

        # Price filter
        mkt = p.get("market_price", 0)
        if mkt < min_price or mkt > max_price:
            continue

        # Volume filter
        if p.get("volume", 0) < min_volume:
            continue

        # Predicted-only filter
        if predicted and p.get("our_prediction") is None:
            continue

        edge = p.get("our_prediction", mkt) - mkt
        results.append({**p, "_key": key, "_edge": edge})

    # Sort by absolute edge (biggest divergence first)
    results.sort(key=lambda x: abs(x["_edge"]), reverse=True)
    results = results[:limit]

    if not results:
        console.print("[yellow]No markets match your filters.[/yellow]")
        return

    table = Table(title=f"Search Results ({len(results)} matches)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", max_width=45)
    table.add_column("Tag", style="dim")
    table.add_column("Market", justify="right", style="yellow")
    table.add_column("Ours", justify="right", style="bold")
    table.add_column("Edge", justify="right")
    table.add_column("Vol", justify="right", style="dim")
    table.add_column("Status", justify="center")

    for i, r in enumerate(results, 1):
        edge = r["_edge"]
        edge_color = "green" if edge > 0.02 else "red" if edge < -0.02 else "yellow"
        status = "[green]Resolved[/green]" if r.get("resolved") else "[blue]Active[/blue]"
        vol = r.get("volume", 0)
        vol_str = f"${vol / 1000:.0f}k" if vol >= 1000 else f"${vol:.0f}"

        table.add_row(
            str(i),
            r.get("question", "?")[:45],
            r.get("tag", "?"),
            f"{r.get('market_price', 0):.2f}",
            f"{r.get('our_prediction', 0):.2f}",
            f"[{edge_color}]{edge:+.2f}[/{edge_color}]",
            vol_str,
            status,
        )
    console.print(table)


# ── Watchlist Command ────────────────────────────────────────────────

@main.group()
def watchlist():
    """Manage your personal market watchlist.

    Track specific markets and see how your watched predictions evolve.

    Examples:
        orcetra watchlist add <condition_id_or_keyword>
        orcetra watchlist show
        orcetra watchlist remove <condition_id>
    """
    pass


@watchlist.command("add")
@click.argument("query")
def watchlist_add(query):
    """Add a market to your watchlist by condition ID or keyword search."""
    wl = _load_json(WATCHLIST_FILE, {"markets": []})
    batch = _load_json(BATCH_FILE, {"predictions": {}})
    preds = batch.get("predictions", {})

    # Try exact match on condition_id first
    matched = None
    for key, p in preds.items():
        if key == query or p.get("condition_id") == query:
            matched = (key, p)
            break

    # Keyword search fallback
    if not matched:
        query_lower = query.lower()
        for key, p in preds.items():
            if query_lower in p.get("question", "").lower():
                matched = (key, p)
                break

    if not matched:
        console.print(f"[red]No market found matching '{query}'. Use 'orcetra search' to find markets.[/red]")
        return

    key, pred = matched

    # Check if already on watchlist
    existing_ids = {m["condition_id"] for m in wl["markets"]}
    cid = pred.get("condition_id", key)
    if cid in existing_ids:
        console.print(f"[yellow]Already on watchlist: {pred['question'][:60]}[/yellow]")
        return

    from datetime import datetime, timezone
    wl["markets"].append({
        "condition_id": cid,
        "question": pred.get("question", ""),
        "tag": pred.get("tag", ""),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "market_price_at_add": pred.get("market_price"),
        "our_prediction_at_add": pred.get("our_prediction"),
    })
    _save_json(WATCHLIST_FILE, wl)
    console.print(f"[green]Added to watchlist:[/green] {pred['question'][:60]}")
    console.print(f"  Market: {pred.get('market_price', 0):.2f} | Ours: {pred.get('our_prediction', 0):.2f}")


@watchlist.command("remove")
@click.argument("query")
def watchlist_remove(query):
    """Remove a market from your watchlist by condition ID or keyword."""
    wl = _load_json(WATCHLIST_FILE, {"markets": []})
    query_lower = query.lower()

    removed = None
    new_markets = []
    for m in wl["markets"]:
        if m["condition_id"] == query or query_lower in m.get("question", "").lower():
            removed = m
        else:
            new_markets.append(m)

    if not removed:
        console.print(f"[red]No watchlist item matching '{query}'[/red]")
        return

    wl["markets"] = new_markets
    _save_json(WATCHLIST_FILE, wl)
    console.print(f"[green]Removed:[/green] {removed['question'][:60]}")


@watchlist.command("show")
def watchlist_show():
    """Show all markets on your watchlist with current status."""
    wl = _load_json(WATCHLIST_FILE, {"markets": []})
    batch = _load_json(BATCH_FILE, {"predictions": {}})
    preds = batch.get("predictions", {})
    log = _load_json(CHECK_LOG, {"checks": []})
    resolved_ids = {c.get("condition_id") for c in log.get("checks", [])}

    if not wl["markets"]:
        console.print("[yellow]Watchlist is empty. Use 'orcetra watchlist add <keyword>' to add markets.[/yellow]")
        return

    table = Table(title=f"Your Watchlist ({len(wl['markets'])} markets)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", max_width=40)
    table.add_column("Tag", style="dim")
    table.add_column("Added Price", justify="right")
    table.add_column("Current Price", justify="right")
    table.add_column("Our Pred", justify="right", style="bold")
    table.add_column("Move", justify="right")
    table.add_column("Status", justify="center")

    for i, m in enumerate(wl["markets"], 1):
        cid = m["condition_id"]
        current = preds.get(cid, {})
        added_price = m.get("market_price_at_add", 0)
        current_price = current.get("market_price", added_price)
        our_pred = m.get("our_prediction_at_add", 0)
        move = current_price - added_price
        move_color = "green" if move > 0.01 else "red" if move < -0.01 else "dim"

        if cid in resolved_ids:
            status = "[green]Resolved[/green]"
        elif current.get("resolved"):
            status = "[green]Resolved[/green]"
        else:
            status = "[blue]Active[/blue]"

        table.add_row(
            str(i),
            m.get("question", "?")[:40],
            m.get("tag", "?"),
            f"{added_price:.2f}",
            f"{current_price:.2f}",
            f"{our_pred:.2f}",
            f"[{move_color}]{move:+.2f}[/{move_color}]",
            status,
        )
    console.print(table)


# ── Export Command ───────────────────────────────────────────────────

@main.command()
@click.option("--format", "-f", "fmt", type=click.Choice(["csv", "json"]), default="csv", help="Output format")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
@click.option("--resolved-only/--all", default=False, help="Only export resolved predictions")
@click.option("--category", "-c", default=None, help="Filter by category")
def export(fmt, output, resolved_only, category):
    """Export predictions to CSV or JSON.

    Export your prediction data for analysis in external tools like
    Excel, Jupyter, or R.

    Examples:
        orcetra export -f csv -o predictions.csv
        orcetra export -f json --resolved-only
        orcetra export --category politics -o politics.csv
    """
    batch = _load_json(BATCH_FILE, {"predictions": {}})
    log = _load_json(CHECK_LOG, {"checks": []})
    preds = batch.get("predictions", {})

    if not preds:
        console.print("[yellow]No predictions to export. Run batch_tracker.py predict first.[/yellow]")
        return

    # Build resolved lookup
    resolved_map = {}
    for c in log.get("checks", []):
        cid = c.get("condition_id", "")
        if cid:
            resolved_map[cid] = c

    rows = []
    for key, p in preds.items():
        cid = p.get("condition_id", key)

        if category and p.get("tag", "").lower() != category.lower():
            continue

        resolved_info = resolved_map.get(cid, {})
        is_resolved = bool(resolved_info)

        if resolved_only and not is_resolved:
            continue

        row = {
            "condition_id": cid,
            "question": p.get("question", ""),
            "tag": p.get("tag", ""),
            "market_price": p.get("market_price"),
            "our_prediction": p.get("our_prediction"),
            "edge": round((p.get("our_prediction", 0) or 0) - (p.get("market_price", 0) or 0), 4),
            "confidence": p.get("confidence"),
            "volume": p.get("volume"),
            "end_date": p.get("end_date", ""),
            "predicted_at": p.get("predicted_at", ""),
            "resolved": is_resolved,
            "actual": resolved_info.get("actual"),
            "our_brier": resolved_info.get("our_brier"),
            "mkt_brier": resolved_info.get("mkt_brier"),
            "beat_market": resolved_info.get("beat_market"),
        }
        rows.append(row)

    if not rows:
        console.print("[yellow]No predictions match your filters.[/yellow]")
        return

    if fmt == "csv":
        import csv
        import io
        fieldnames = list(rows[0].keys())
        if output:
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            console.print(f"[green]Exported {len(rows)} predictions to {output}[/green]")
        else:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            click.echo(buf.getvalue())

    elif fmt == "json":
        out_data = json.dumps(rows, indent=2, default=str)
        if output:
            Path(output).write_text(out_data)
            console.print(f"[green]Exported {len(rows)} predictions to {output}[/green]")
        else:
            click.echo(out_data)


if __name__ == "__main__":
    main()