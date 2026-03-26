#!/usr/bin/env python3
"""Orcetra CLI - Prediction Market Crowd Simulation with Self-Evolving AI Agents."""

import argparse
import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from _deprecated_murmur.fetcher import (
    PolymarketFetcher,
    list_active_events_sync,
    list_resolved_events_sync,
)
from _deprecated_murmur.news import NewsCollector
from _deprecated_murmur.strategy import get_default_strategy, LLMStrategy
from _deprecated_murmur.evaluator import evaluate_strategy_sync, format_backtest_results
from _deprecated_murmur.evolve import run_evolution, load_best_strategy

console = Console()


async def predict_event(url_or_slug: str, use_evolved: bool = False) -> None:
    """Fetch data and run prediction for a single event."""
    fetcher = PolymarketFetcher()
    news_collector = NewsCollector()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Fetch event
            task = progress.add_task("Fetching event data...", total=None)
            event = await fetcher.fetch_event(url_or_slug)

            if event is None:
                console.print(f"[red]Error: Could not find event '{url_or_slug}'[/red]")
                return

            progress.update(task, description="Event found!")

            # Display event info
            console.print()
            console.print(Panel(
                f"[bold]{event.title}[/bold]\n\n"
                f"{event.description[:300]}{'...' if len(event.description) > 300 else ''}\n\n"
                f"[dim]End Date: {event.end_date or 'Unknown'}[/dim]\n"
                f"[dim]Volume: ${event.volume:,.0f} | Liquidity: ${event.liquidity:,.0f}[/dim]",
                title="Event",
                border_style="blue",
            ))

            # Show current market prices
            if event.tokens:
                table = Table(title="Current Market Prices")
                table.add_column("Outcome", style="cyan")
                table.add_column("Price", justify="right", style="green")
                table.add_column("Implied Probability", justify="right")

                for token in event.tokens:
                    table.add_row(
                        token.outcome,
                        f"${token.price:.3f}",
                        f"{token.price:.1%}",
                    )
                console.print(table)

            # Fetch additional context
            progress.update(task, description="Fetching price history...")
            price_history = []
            orderbook = None

            if event.tokens and event.tokens[0].token_id:
                token_id = event.tokens[0].token_id
                price_history = await fetcher.fetch_price_history(token_id)
                progress.update(task, description="Fetching order book...")
                orderbook = await fetcher.fetch_orderbook(token_id)

            # Fetch news
            progress.update(task, description="Searching for news signals...")
            news_signals = await news_collector.search_news(event.title[:100], days_back=7)

            if news_signals:
                progress.update(task, description="Analyzing news signals...")
                news_signals = await news_collector.summarize_signals(
                    news_signals, event.title
                )

            # Get strategy
            progress.update(task, description="Running prediction strategy...")

            if use_evolved:
                strategy = load_best_strategy()
                if strategy is None:
                    console.print("[yellow]No evolved strategy found, using default[/yellow]")
                    strategy = get_default_strategy()
            else:
                strategy = get_default_strategy()

            # Run prediction
            prediction = strategy.predict(event, price_history, news_signals, orderbook)

        # Display prediction results
        console.print()

        # Prediction panel
        prob_color = "green" if prediction.probability > 0.5 else "red"
        market_diff = prediction.probability - (event.tokens[0].price if event.tokens else 0.5)
        diff_sign = "+" if market_diff > 0 else ""

        console.print(Panel(
            f"[bold {prob_color}]{prediction.probability:.1%}[/bold {prob_color}] probability of YES\n\n"
            f"Confidence: [bold]{prediction.confidence:.1%}[/bold]\n"
            f"Market price: {event.tokens[0].price:.1%}\n"
            f"Difference: [{'green' if market_diff > 0 else 'red'}]{diff_sign}{market_diff:.1%}[/]",
            title="Prediction",
            border_style="green" if prediction.probability > 0.5 else "red",
        ))

        # Reasoning
        console.print(Panel(
            prediction.reasoning,
            title="Reasoning",
            border_style="yellow",
        ))

        # 7-day trajectory
        if prediction.trajectory_7d:
            table = Table(title="7-Day Price Trajectory")
            for i, price in enumerate(prediction.trajectory_7d, 1):
                table.add_column(f"Day {i}", justify="center")
            table.add_row(*[f"{p:.1%}" for p in prediction.trajectory_7d])
            console.print(table)

        # News signals summary
        if news_signals:
            console.print()
            table = Table(title=f"News Signals ({len(news_signals)} found)")
            table.add_column("Source", style="cyan", max_width=20)
            table.add_column("Title", max_width=50)
            table.add_column("Sentiment", justify="center")
            table.add_column("Relevance", justify="right")

            for signal in sorted(news_signals, key=lambda x: x.relevance, reverse=True)[:5]:
                sentiment_color = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "yellow",
                }.get(signal.sentiment or "neutral", "white")

                table.add_row(
                    signal.source[:20],
                    signal.title[:50],
                    f"[{sentiment_color}]{signal.sentiment or 'neutral'}[/]",
                    f"{signal.relevance:.0%}",
                )

            console.print(table)

    finally:
        await fetcher.close()
        await news_collector.close()


def list_events(active: bool = True, limit: int = 20) -> None:
    """List active or resolved events."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Fetching {'active' if active else 'resolved'} events...",
            total=None,
        )

        if active:
            events = list_active_events_sync(limit=limit)
        else:
            events = list_resolved_events_sync(limit=limit)

        progress.update(task, completed=True)

    if not events:
        console.print("[yellow]No events found[/yellow]")
        return

    table = Table(title=f"{'Active' if active else 'Resolved'} Events ({len(events)})")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", max_width=50)
    table.add_column("Slug", style="cyan", max_width=30)
    table.add_column("YES Price", justify="right", style="green")
    table.add_column("Volume", justify="right")

    for i, event in enumerate(events, 1):
        yes_price = "N/A"
        if event.tokens:
            for token in event.tokens:
                if token.outcome.lower() in ["yes", "true", "1"]:
                    yes_price = f"{token.price:.1%}"
                    break
            else:
                yes_price = f"{event.tokens[0].price:.1%}"

        table.add_row(
            str(i),
            event.title[:50],
            event.slug[:30] if event.slug else "N/A",
            yes_price,
            f"${event.volume:,.0f}",
        )

    console.print(table)


def run_backtest(limit: int = 20) -> None:
    """Run backtest on resolved events."""
    console.print(f"[bold]Running backtest on {limit} resolved events...[/bold]")

    console.print("Fetching resolved events...")
    events = list_resolved_events_sync(limit=limit)
    console.print(f"Found {len(events)} resolved events")

    # Filter to events that have outcome_price set
    valid = [e for e in events if e.outcome_price is not None]
    console.print(f"  {len(valid)} with valid outcomes")

    if len(valid) < 3:
        console.print(f"[red]Not enough valid events ({len(valid)}). Need at least 3.[/red]")
        return

    console.print("Running predictions (this may take a minute)...")
    strategy = get_default_strategy()
    result = evaluate_strategy_sync(strategy, valid, fetch_context=False)

    console.print(format_backtest_results(result))


def run_evolve(rounds: int = 10) -> None:
    """Run evolution loop."""
    console.print(f"[bold]Starting evolution loop ({rounds} rounds)...[/bold]")
    console.print()

    try:
        best_code, result = run_evolution(rounds=rounds, verbose=True)

        console.print()
        console.print("[green]Evolution complete! Best strategy saved to results/best_strategy.py[/green]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Orcetra - Prediction Market Crowd Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py https://polymarket.com/event/will-trump-win-2024
  python predict.py will-trump-win-2024
  python predict.py --list
  python predict.py --backtest
  python predict.py --evolve --rounds 20
        """,
    )

    parser.add_argument(
        "event",
        nargs="?",
        help="Polymarket event URL or slug to predict",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List active events",
    )

    parser.add_argument(
        "--resolved",
        action="store_true",
        help="List resolved events (use with --list)",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on resolved events",
    )

    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Run evolution loop to improve strategy",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of evolution rounds (default: 10)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of events to list (default: 20)",
    )

    parser.add_argument(
        "--evolved",
        action="store_true",
        help="Use evolved strategy for prediction",
    )

    args = parser.parse_args()

    # Handle different modes
    if args.list:
        list_events(active=not args.resolved, limit=args.limit)
    elif args.backtest:
        run_backtest()
    elif args.evolve:
        run_evolve(rounds=args.rounds)
    elif args.event:
        asyncio.run(predict_event(args.event, use_evolved=args.evolved))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
