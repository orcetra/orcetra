# Orcetra

**Give it data, it finds the best model.** AutoML engine that wins 57% on 513 OpenML datasets vs AutoGluon (21.6%) and FLAML (10.9%).

[![PyPI](https://img.shields.io/pypi/v/orcetra)](https://pypi.org/project/orcetra/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Install

```bash
pip install orcetra
```

## Quick Start

```bash
orcetra predict housing.csv --target price --budget 60s
```

```python
from orcetra.core.loop import run_prediction

result = run_prediction("housing.csv", target="price", budget="60s")
print(result["best_model"], result["best_score"])
```

## Benchmark Results

**513 OpenML datasets, 60-second budget, CPU only.**

| | Orcetra | AutoGluon | FLAML | Tie |
|---|---|---|---|---|
| **Overall (513)** | **57.1%** | 21.6% | 10.9% | 10.3% |
| Classification (382) | **58.4%** | 17.5% | 11.3% | 12.8% |
| Regression (131) | **53.4%** | 33.6% | 9.9% | 3.1% |

### Orcetra vs FLAML only (30-second budget)

| | Orcetra | FLAML | Tie |
|---|---|---|---|
| **Overall (513)** | **78.4%** | 14.6% | 7.0% |

## How It Works

```
1. Analyze    → Detect task type, features, data shape
2. Baseline   → Evaluate RF, GBC, XGB, LightGBM, LogReg, etc.
3. Search     → Parallel strategy search with dedup cache
4. Result     → Best model + score + improvement over baseline
```

**Why it's fast:** Calibration-based warm start + meta-search space (searches what works for similar data, not the full space) + optimized for short budgets (30-60s).

## Optional Extras

```bash
pip install orcetra[xgboost]   # XGBoost support
pip install orcetra[llm]       # LLM-guided search (Groq/OpenAI)
pip install orcetra[all]       # Everything
```

## Architecture

```
src/orcetra/
├── core/           # Search loop + agents (Random, LLM-guided)
│   ├── loop.py     # Main prediction loop
│   ├── agent.py    # RandomSearchAgent
│   ├── llm_agent.py # LLM-guided search
│   └── calibration.py
├── data/           # Data loading + splitting
├── models/         # Model registry + baselines
├── metrics/        # Evaluation (accuracy, MSE, R², etc.)
├── benchmarks/     # OpenML benchmark suite
└── cli.py          # CLI entry point
```

## Roadmap

- [x] Baseline model pool (RF, GBC, XGB, LightGBM, HistGBM, LogReg)
- [x] Parallel random search with strategy cache
- [x] OpenML benchmark (513 datasets, 3-way comparison)
- [x] PyPI package (`pip install orcetra`)
- [ ] CatBoost baseline ([#10](https://github.com/orcetra/orcetra/issues/10))
- [ ] Top-3 ensemble ([#11](https://github.com/orcetra/orcetra/issues/11))
- [ ] Feature engineering module ([#1](https://github.com/orcetra/orcetra/issues/1))
- [ ] Meta-learning knowledge base ([#8](https://github.com/orcetra/orcetra/issues/8))
- [ ] GitHub Actions CI ([#18](https://github.com/orcetra/orcetra/issues/18))

## Contributing

PRs welcome. Main branch is protected — open a PR and request review.

```bash
git clone https://github.com/orcetra/orcetra.git
cd orcetra
pip install -e ".[all]"
```

## License

MIT
