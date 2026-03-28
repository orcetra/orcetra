# OpenML Benchmark Resume Plan

## Current Coverage Summary

| Metric | Value |
|--------|-------|
| Total entries in JSONL | 670 |
| Unique dataset IDs | 670 |
| Duplicates | 0 |
| Target datasets | 3500 |
| Remaining | 2830 |

### Status Breakdown
- **success**: 477 (71.2%)
- **error**: 193 (28.8%)

### Error Analysis
| Error Type | Count |
|------------|-------|
| No data | 166 |
| Other | 21 |
| Target issues | 6 |

Most errors are "No data" issues from OpenML datasets that don't have valid target attributes—these are expected and unavoidable.

## How Resume Works

The `--resume` flag in `openml_benchmark.py`:
1. Loads all `dataset_id` values from the existing JSONL file into a `done_ids` set (line 323-331)
2. Fetches the full dataset list via `get_datasets(max_datasets)` (line 335)
3. Filters to only process datasets not in `done_ids` (line 336)
4. Skipping is based on dataset_id regardless of status (success/error)

### Dataset Selection (`get_datasets()`)
1. **Curated first**: Fetches datasets from OpenML suites 218 (AutoML Benchmark), 99 (CC18), 269 (Regression)
2. **Quality filter**: 50-100k instances, 2-500 features, <30% missing, ≤30 classes, active status
3. **Sort order**: By number of instances descending (larger datasets first)
4. **Truncate**: Limited to `--max-datasets` (default 3500)

### Budget Control
- `--budget` controls per-dataset Orcetra search time (default 30s)
- Does NOT affect sklearn baseline or data loading time

## Resume Command

```bash
cd /home/guilinzhang/allProjects/orcetra

# Basic resume (recommended)
python experiments/openml_benchmark.py --resume --max-datasets 3500 --budget 30s

# With nohup for long runs
nohup python experiments/openml_benchmark.py --resume --max-datasets 3500 --budget 30s > benchmark_resume.log 2>&1 &

# Monitor progress
tail -f benchmark_resume.log
```

## Time Estimates

### Observed Performance
| Metric | Value |
|--------|-------|
| Wall time elapsed | 18.5h |
| Throughput | 36.2 datasets/hour |
| Mean time/dataset | 101.4s |
| Median time/dataset | 48.7s |

### Projected Completion
| Scenario | Remaining Time | Total Wall Time |
|----------|---------------|-----------------|
| Median (optimistic) | ~38h | ~57h total |
| Mean (realistic) | ~80h | ~98h total |
| Throughput-based | ~78h | ~97h total |

**Recommendation**: Expect ~3-4 days of continuous runtime to reach 3500 datasets.

## Risks and Caveats

### 1. Date-based Filename
The output file is `openml_benchmark_YYYYMMDD.jsonl`. If you run across midnight, a new file is created and `--resume` won't find previous results.

**Mitigation**: Either:
- Complete in one session
- Manually rename/merge files before resuming
- Modify code to use fixed filename

### 2. Dataset Availability Changes
OpenML dataset availability can change. Some datasets may become unavailable or change status.

**Impact**: Low—already handled gracefully as errors.

### 3. API Rate Limits
Long runs may hit OpenML API rate limits or transient failures.

**Impact**: Currently handled with try/except, logged as errors.

### 4. Memory Usage
Large datasets (up to 100k samples) may cause memory pressure over long runs.

**Mitigation**: Monitor with `htop`; consider restarting if memory creeps up.

### 5. No Checkpointing Within Dataset
If the process crashes mid-dataset, that dataset's partial work is lost. Results are only written after completing each dataset.

**Impact**: Low—individual datasets take at most ~1h.

## Post-Run Analysis

After completion, analyze results with:

```bash
python -c "
import json
from collections import Counter

with open('experiments/results/openml_benchmark_20260326.jsonl') as f:
    data = [json.loads(l) for l in f]

success = [d for d in data if d.get('status') == 'success']
wins = sum(1 for d in success if d.get('orcetra_wins'))
print(f'Total: {len(data)}, Success: {len(success)}, Orcetra wins: {wins} ({100*wins/len(success):.1f}%)')
"
```
