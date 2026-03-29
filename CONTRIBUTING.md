# Orcetra — Project Overview & Contributing Guide

## What is Orcetra?

Orcetra is an automated prediction engine with two modules:

### Module 1: AutoML Engine (ICML paper target)
- Given any tabular dataset, automatically finds the best model + hyperparameters
- **Current results**: 78.4% win rate vs FLAML on 513 OpenML datasets (strict 30s budget)
- 60s three-way benchmark (vs FLAML + AutoGluon) running now
- Key insight: diverse baseline pool + random search beats cost-frugal optimization on small-medium datasets

### Module 2: Polymarket Prediction (application/demo)
- Monitors prediction markets, applies calibration, generates predictions
- 2,955 verified predictions, 67% overall beat rate
- Categories: sports (68%), politics (77%), golf (90%), crypto (52% — needs work), economy (69%)
- Live dashboard: https://orcetra.ai/dashboard.html

## Current Status (March 28, 2026)

### Done ✅
- AutoML baseline pool (RF, GBC, XGB, LightGBM, HistGBM, LogReg, ExtraTrees, etc.)
- Random search agent
- OpenML benchmark: 513 datasets, 78.4% win rate vs FLAML
- Polymarket pipeline: 22K+ tracked markets, auto-check + dashboard
- GitHub org: github.com/orcetra/orcetra

### In Progress 🔄
- 60s three-way benchmark: Orcetra vs FLAML vs AutoGluon (running overnight)
- Polymarket optimization: stopping crypto short-term + sports subcategory analysis

### Roadmap
- [ ] LLM-guided search agent (replace RandomSearch with Groq/OpenAI-guided proposals)
- [ ] Automated feature engineering module (Kai's AutoResearch approach)
- [ ] MLE-bench evaluation (75 Kaggle competitions)
- [ ] Multi-budget analysis (10s/30s/60s/120s pareto curves)
- [ ] Cross-market prediction (beyond Polymarket)
- [ ] User-facing tool ("Cursor for ML")
- [ ] Paper draft for ICML workshop (deadline: Apr 24)

## Architecture

```
src/orcetra/
├── core/           → Search agents (Random, LLM-guided)
├── models/         → Model registry & baselines
├── metrics/        → Evaluation metrics
└── meta/           → Strategy knowledge base (future)

experiments/
├── openml_benchmark.py           → Baseline benchmark (670 datasets)
├── flaml_strict_30s.py           → Orcetra vs FLAML (strict budget)
├── flaml_strict_30s_remaining.py → Parallel runner for remaining datasets
├── multi_framework_benchmark.py  → Three-way: Orcetra vs FLAML vs AutoGluon
└── results/                      → All benchmark outputs (.jsonl)

batch_tracker.py    → Polymarket prediction pipeline
auto_check.py       → Outcome verification + scoring
live_tracker.py     → LLM-powered deep market analysis
```

## How to Get Started

### Run AutoML on a dataset
```python
from orcetra.models.registry import get_baselines
from orcetra.metrics.base import get_metric

metric = get_metric("accuracy")
baselines = get_baselines("classification")
for name, fn in baselines.items():
    score = fn(data_info, metric)
```

### Run Polymarket pipeline
```bash
python batch_tracker.py predict    # Generate predictions
python auto_check.py               # Check resolved markets
```

### Run benchmarks
```bash
python experiments/flaml_strict_30s.py              # vs FLAML
python experiments/multi_framework_benchmark.py     # vs FLAML + AutoGluon
```

## Team & Roles

| Name | Focus Area | Notes |
|---|---|---|
| **Guilin** | Core engine, benchmarks, architecture | PM + primary dev |
| **Kai** | Feature engineering, AutoResearch, MLE-bench | Has working House Price pipeline (90 strategies/110s) |
| **Luyi** | Market research, cross-market analysis, user interviews | Silicon Valley network, weekly events |
| **Xiyang** | Deep research (TBD) | CMU Heinz, to be onboarded |
| **Beibei** | Advisor | CMU professor, paper guidance |

## Contributing

1. Pick an issue from [GitHub Issues](https://github.com/orcetra/orcetra/issues)
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes, test locally
4. Submit a PR with description of what changed and why

### Branch naming
- `feature/xxx` — new functionality
- `fix/xxx` — bug fixes
- `experiment/xxx` — benchmark experiments
- `paper/xxx` — paper-related content

### Commit messages
- `feat: xxx` — new feature
- `fix: xxx` — bug fix
- `docs: xxx` — documentation
- `exp: xxx` — experiment results
