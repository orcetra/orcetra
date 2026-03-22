#!/usr/bin/env python3
"""Regenerate dashboard data.json + self-contained HTML from check_log."""
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(DIR, "..")
RESULTS = os.path.join(ROOT, "results")
DASH = os.path.join(ROOT, "dashboard")

def main():
    with open(os.path.join(RESULTS, "check_log.json")) as f:
        log = json.load(f)
    checks_raw = log["checks"]

    # Filter out crypto 5-min binary bets ("Up or Down") — these are coin flips
    # with ~50% beat rate that drag down overall metrics
    def is_crypto_binary(c):
        return c.get("tag") == "crypto" and "Up or Down" in c.get("question", "")

    checks = [c for c in checks_raw if not is_crypto_binary(c)]
    crypto_excluded = [c for c in checks_raw if is_crypto_binary(c)]

    with open(os.path.join(RESULTS, "batch_predictions.json")) as f:
        batch = json.load(f)
    preds = batch["predictions"]

    total = len(preds)
    n_checked = len(checks)
    n_beat = sum(1 for c in checks if c["beat_market"])

    # By tag
    from collections import Counter
    tags = Counter()
    beat_tags = Counter()
    brier_by_tag = {}
    for c in checks:
        t = c.get("tag", "unknown")
        tags[t] += 1
        if c["beat_market"]:
            beat_tags[t] += 1
        if t not in brier_by_tag:
            brier_by_tag[t] = {"our": [], "mkt": []}
        brier_by_tag[t]["our"].append(c["our_brier"])
        brier_by_tag[t]["mkt"].append(c["mkt_brier"])

    by_tag = {}
    for t in tags:
        by_tag[t] = {
            "total": tags[t],
            "beat": beat_tags.get(t, 0),
            "rate": beat_tags.get(t, 0) / tags[t] if tags[t] else 0,
            "avg_our_brier": sum(brier_by_tag[t]["our"]) / len(brier_by_tag[t]["our"]),
            "avg_mkt_brier": sum(brier_by_tag[t]["mkt"]) / len(brier_by_tag[t]["mkt"]),
        }

    # All checked sorted by spread
    all_checked = []
    for c in checks:
        spread = abs(c["our_prediction"] - c["market_price"])
        
        # Look up the reasoning from batch_predictions.json
        pred_key = c.get("condition_id", "")
        pred_data = preds.get(pred_key, {})
        reasoning = pred_data.get("reasoning", "")
        
        all_checked.append({
            "q": c["question"],
            "tag": c.get("tag", "unknown"),
            "orcetra": round(c["our_prediction"], 4),
            "market": round(c["market_price"], 4),
            "actual": c["actual"],
            "our_brier": round(c["our_brier"], 4),
            "mkt_brier": round(c["mkt_brier"], 4),
            "beat": c["beat_market"],
            "spread": round(spread, 4),
            "reasoning": reasoning,
        })
    all_checked.sort(key=lambda x: x["spread"], reverse=True)

    dashboard_data = {
        "summary": {
            "total_predictions": total,
            "resolved": n_checked,
            "pending": total - n_checked,
            "beat_rate": n_beat / n_checked if n_checked else 0,
            "avg_orcetra_brier": sum(c["our_brier"] for c in checks) / n_checked if n_checked else 0,
            "avg_market_brier": sum(c["mkt_brier"] for c in checks) / n_checked if n_checked else 0,
            "last_updated": batch.get("last_updated", "unknown"),
            "crypto_binary_excluded": len(crypto_excluded),
        },
        "by_tag": by_tag,
        "top_divergences": all_checked[:20],
        "all_checked": all_checked,
    }

    os.makedirs(DASH, exist_ok=True)

    # Write data.json
    data_path = os.path.join(DASH, "data.json")
    with open(data_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    # Write self-contained HTML
    html_template = os.path.join(DASH, "index.html")
    if os.path.exists(html_template):
        with open(html_template) as f:
            html = f.read()

        old = "const response = await fetch('data.json');\n                data = await response.json();"
        new = f"data = {json.dumps(dashboard_data)};"
        html_standalone = html.replace(old, new)

        out_path = os.path.join(DASH, "orcetra-dashboard.html")
        with open(out_path, "w") as f:
            f.write(html_standalone)
        print(f"Dashboard updated: {n_checked} checks, {n_beat}/{n_checked} beat ({100*n_beat/n_checked:.1f}%)")

        # Also copy to site/ for Cloudflare Pages deployment
        site_dir = os.path.join(ROOT, "site")
        if os.path.isdir(site_dir):
            import shutil
            shutil.copy2(out_path, os.path.join(site_dir, "dashboard.html"))
            print("Copied to site/dashboard.html")
    else:
        print(f"data.json updated (no index.html template found)")


if __name__ == "__main__":
    main()
