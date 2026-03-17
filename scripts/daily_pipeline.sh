#!/usr/bin/env bash
# Murmur daily pipeline: predict → check → enrich → track prices → dashboard → alert
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/pipeline.log"
TS=$(date -Iseconds)
echo "=== $TS ===" >> "$LOG"

# 1. Fetch new markets + rule-based predict
echo "[1/5] Batch predict..." >> "$LOG"
python3 batch_tracker.py predict >> "$LOG" 2>&1 || true

# 2. Auto-check resolved predictions
echo "[2/5] Auto check..." >> "$LOG"
python3 auto_check.py >> "$LOG" 2>&1 || true

# 3. Track price snapshots (for price movement analysis)
echo "[3/5] Price tracking..." >> "$LOG"
python3 scripts/track_prices.py >> "$LOG" 2>&1 || true

# 4. Regenerate dashboard
echo "[4/5] Update dashboard..." >> "$LOG"
python3 scripts/gen_dashboard.py >> "$LOG" 2>&1 || true

# 5. Alert check — notify if beat rate drops below threshold
echo "[5/5] Alert check..." >> "$LOG"
python3 -c "
import json
with open('results/check_log.json') as f:
    checks = json.load(f)['checks']
n = len(checks)
if n >= 20:
    beat = sum(1 for c in checks if c['beat_market'])
    rate = beat / n
    # Alert if rate drops below 70% (was 84%)
    if rate < 0.70:
        import subprocess
        msg = f'⚠️ Murmur alert: beat rate dropped to {rate:.1%} ({beat}/{n}). Check pipeline.'
        subprocess.run(['openclaw', 'system', 'event', '--text', msg, '--mode', 'now'], check=False)
        print(f'ALERT SENT: {rate:.1%}')
    else:
        print(f'OK: beat rate {rate:.1%} ({beat}/{n})')
else:
    print(f'Skipped alert: only {n} checks (need 20+)')
" >> "$LOG" 2>&1 || true

# 6. Auto-commit and push updated data (triggers Cloudflare Pages rebuild)
echo "[6/6] Git push..." >> "$LOG"
if git diff --quiet site/ dashboard/ results/check_log.json results/batch_predictions.json 2>/dev/null; then
    echo "No changes to push" >> "$LOG"
else
    git add site/ dashboard/ results/check_log.json results/batch_predictions.json 2>/dev/null
    git commit -m "data: auto-update $(date +%Y-%m-%d_%H:%M)" --no-verify 2>/dev/null >> "$LOG" || true
    git push 2>/dev/null >> "$LOG" || true
    echo "Pushed updates" >> "$LOG"
fi

echo "=== Done $(date -Iseconds) ===" >> "$LOG"
