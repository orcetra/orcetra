#!/usr/bin/env bash
# Orcetra daily pipeline: predict → check → enrich → track prices → dashboard → alert
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

# Exclude sports from alert (Phase 1 finding: negative EV, not a bet category)
NO_ALERT_TAGS = {'sports'}
bet_checks = [c for c in checks if c.get('tag') not in NO_ALERT_TAGS]
all_n = len(checks)
n = len(bet_checks)

if n >= 20:
    beat = sum(1 for c in bet_checks if c['beat_market'])
    rate = beat / n

    # Also compute overall for logging
    all_beat = sum(1 for c in checks if c['beat_market'])
    all_rate = all_beat / all_n if all_n else 0

    # Alert if bet-category rate drops below 70%
    if rate < 0.70:
        import subprocess
        msg = f'⚠️ Orcetra alert: bet-category beat rate dropped to {rate:.1%} ({beat}/{n}). Overall: {all_rate:.1%} ({all_beat}/{all_n}). Check pipeline.'
        subprocess.run(['openclaw', 'system', 'event', '--text', msg, '--mode', 'now'], check=False)
        print(f'ALERT SENT: bet={rate:.1%}, overall={all_rate:.1%}')
    else:
        print(f'OK: bet={rate:.1%} ({beat}/{n}), overall={all_rate:.1%} ({all_beat}/{all_n})')
else:
    print(f'Skipped alert: only {n} bet-category checks (need 20+)')
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
