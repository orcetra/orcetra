import click
from rich.console import Console

console = Console()

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
@click.argument("benchmark_name", default="openml")
def benchmark(benchmark_name):
    """Run a built-in benchmark.

    Example: orcetra benchmark openml
    """
    console.print(f"[bold]Running benchmark: {benchmark_name}[/bold]")
    if benchmark_name == "openml":
        from .benchmarks.openml.runner import run_benchmark
        run_benchmark()
    else:
        console.print(f"[red]Unknown benchmark: {benchmark_name}[/red]")

if __name__ == "__main__":
    main()