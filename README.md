# 🎯 Orcetra — Automated Prediction Engine

**AutoML that beats FLAML on 78.4% of datasets. Verified on 513 OpenML benchmarks with equal 30-second compute budget.**

[Live Dashboard](https://orcetra.ai/dashboard.html) · [Website](https://orcetra.ai) · [Paper (coming soon)](#)

---

## What Is Orcetra?

Orcetra is an automated prediction engine that combines **intelligent model search** with **meta-learning** to outperform established AutoML frameworks. It works on any tabular dataset — no manual feature engineering required.

## Benchmark Results

### Orcetra vs FLAML (513 OpenML datasets, strict 30s budget each)

| | Orcetra Wins | FLAML Wins | Tie |
|---|---|---|---|
| **Overall** | **402 (78.4%)** | 75 (14.6%) | 36 (7.0%) |
| Classification (382) | **299 (78.3%)** | 49 (12.8%) | 34 (8.9%) |
| Regression (131) | **103 (78.6%)** | 26 (19.8%) | 2 (1.5%) |

### By Dataset Size

| Scale | Orcetra | FLAML | Tie |
|---|---|---|---|
| Small (<5K samples) | **73 (86%)** | 7 (8%) | 5 |
| Medium (5-50K) | **260 (79%)** | 48 (15%) | 21 |
| Large (>50K) | **69 (70%)** | 20 (20%) | 10 |

### Compute Fairness

Both systems receive exactly 30 seconds. Orcetra's baseline evaluation phase counts toward the budget.

| | Median Time |
|---|---|
| Orcetra | **30.0s** |
| FLAML | **31.3s** |

### Polymarket Prediction Benchmark

| Category | Beat Rate | Predictions |
|---|---|---|
| **Overall** | **66.7%** | **2,932** |
| Sports | 81% | — |
| Politics | 100% | — |
| Economy | 100% | — |

*Live tracking of 22,000+ active predictions across 8 categories.*

## How It Works

```
Phase 1: Baseline Pool     → Evaluate RF, GBC, XGB, LightGBM, LogReg, etc.
Phase 2: AutoResearch      → LLM-guided or random search for better configs
Phase 3: Meta-Learning     → Strategy knowledge base from prior dataset wins
Phase 4: Validation        → Cross-validated final evaluation
```

**Key insight**: Instead of just searching hyperparameters (like FLAML/Auto-sklearn), Orcetra searches across **model families, preprocessing strategies, and feature transformations** simultaneously.

## Quick Start

```bash
git clone https://github.com/orcetra/orcetra.git
cd orcetra
pip install -r requirements.txt
```

### Run on any dataset

```python
from orcetra.core.agent import RandomSearchAgent
from orcetra.models.registry import get_baselines
from orcetra.metrics.base import get_metric

# Your data
metric = get_metric("accuracy")  # or "mse" for regression
baselines = get_baselines("classification")

# Phase 1: Baselines
for name, fn in baselines.items():
    score = fn(data_info, metric)

# Phase 2: AutoResearch loop
agent = RandomSearchAgent(task_type="classification")
proposal = agent.propose(data_info, metric, best_score, best_model, iteration)
```

### Run Polymarket predictions

```bash
python batch_tracker.py predict    # Generate predictions
python auto_check.py               # Verify against outcomes
python scripts/gen_dashboard.py    # Update dashboard
```

## Architecture

```
src/orcetra/
├── core/           → Search agents (Random, LLM-guided)
├── models/         → Model registry & baselines
├── metrics/        → Evaluation metrics (accuracy, MSE, Brier)
└── meta/           → Strategy knowledge base

experiments/
├── openml_benchmark.py      → Full OpenML benchmark suite
├── flaml_pilot_v2.py        → Head-to-head vs FLAML
└── flaml_strict_30s.py      → Fair compute-budget comparison

batch_tracker.py    → Polymarket prediction pipeline
auto_check.py       → Outcome verification
live_tracker.py     → LLM-powered deep analysis
```

## Roadmap

- [x] Baseline model pool (RF, GBC, XGB, LightGBM, HistGBM, LogReg, etc.)
- [x] Random search agent
- [x] OpenML benchmark framework (513 datasets, 78.4% win rate vs FLAML)
- [x] FLAML head-to-head comparison (strict 30s, compute-fair)
- [x] Polymarket live prediction pipeline (22K+ markets)
- [ ] LLM-guided search agent (Groq/OpenAI)
- [ ] Meta-learning knowledge base
- [ ] MLE-bench evaluation
- [ ] Automated feature engineering module
- [ ] Multi-framework comparison (Auto-sklearn, H2O, AutoGluon)

## Team

| Name | Role | Affiliation |
|---|---|---|
| **Beibei Li** | Chief Scientist | Carnegie Mellon University |
| **Xiyang Hu** | Researcher | Carnegie Mellon University |
| **Kai Zhao** | Researcher | — |
| **Guilin Zhang** | Researcher | George Washington University |

## License

MIT License

---

*"The best model isn't the one you know — it's the one you discover."*
