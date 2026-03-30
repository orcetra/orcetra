#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p out

echo "🎬 Rendering 3 Orcetra LinkedIn videos..."
echo ""

echo "▶ [1/3] BeatTheMarket — Polymarket performance"
npx remotion render BeatTheMarket out/video1-beat-the-market.mp4 \
  --codec h264 \
  --crf 18 \
  --log error
echo "   ✓ out/video1-beat-the-market.mp4"
echo ""

echo "▶ [2/3] SixtySecondAutoML — FLAML benchmark"
npx remotion render SixtySecondAutoML out/video2-sixty-second-automl.mp4 \
  --codec h264 \
  --crf 18 \
  --log error
echo "   ✓ out/video2-sixty-second-automl.mp4"
echo ""

echo "▶ [3/3] NumbersSpeak — pure data impact"
npx remotion render NumbersSpeak out/video3-numbers-speak.mp4 \
  --codec h264 \
  --crf 18 \
  --log error
echo "   ✓ out/video3-numbers-speak.mp4"
echo ""

echo "✅ All 3 videos rendered to out/"
ls -lh out/*.mp4
