import React from "react";
import { useCurrentFrame, interpolate, Sequence } from "remotion";

// ── Data ─────────────────────────────────────────────────────────────

const CLAUDE_LINES = [
  { t: 0, text: "$ claude 'Build a model for Kaggle House Prices'", color: "#8b5cf6" },
  { t: 30, text: "I'll start with XGBoost and feature engineering...", color: "#a78bfa" },
  { t: 60, text: "import pandas as pd", color: "#94a3b8" },
  { t: 66, text: "import xgboost as xgb", color: "#94a3b8" },
  { t: 72, text: "from sklearn.model_selection import cross_val_score", color: "#94a3b8" },
  { t: 90, text: "# Loading and cleaning data...", color: "#64748b" },
  { t: 120, text: "df = pd.read_csv('train.csv')", color: "#94a3b8" },
  { t: 126, text: "df['SalePrice'] = np.log1p(df['SalePrice'])", color: "#94a3b8" },
  { t: 140, text: "# Feature engineering...", color: "#64748b" },
  { t: 170, text: "df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']", color: "#94a3b8" },
  { t: 190, text: "# Trying XGBoost...", color: "#64748b" },
  { t: 220, text: "model = xgb.XGBRegressor(n_estimators=500)", color: "#94a3b8" },
  { t: 250, text: "scores = cross_val_score(model, X, y, cv=5)", color: "#94a3b8" },
  { t: 280, text: "RMSLE: 0.1398", color: "#fbbf24" },
  { t: 310, text: "# Hmm, let me try different params...", color: "#64748b" },
  { t: 340, text: "model2 = xgb.XGBRegressor(n_estimators=800,", color: "#94a3b8" },
  { t: 346, text: "    learning_rate=0.03, max_depth=5)", color: "#94a3b8" },
  { t: 370, text: "scores2 = cross_val_score(model2, X, y, cv=5)", color: "#94a3b8" },
  { t: 400, text: "RMSLE: 0.1302", color: "#fbbf24" },
  { t: 430, text: "# Let me add LightGBM for comparison...", color: "#64748b" },
  { t: 460, text: "import lightgbm as lgb", color: "#94a3b8" },
  { t: 490, text: "model3 = lgb.LGBMRegressor(n_estimators=1000)", color: "#94a3b8" },
  { t: 520, text: "scores3 = cross_val_score(model3, X, y, cv=5)", color: "#94a3b8" },
  { t: 550, text: "RMSLE: 0.1287", color: "#fbbf24" },
  { t: 580, text: "# Stacking ensemble...", color: "#64748b" },
  { t: 620, text: "from sklearn.ensemble import StackingRegressor", color: "#94a3b8" },
  { t: 660, text: "stack = StackingRegressor(estimators=[...])", color: "#94a3b8" },
  { t: 700, text: "Final RMSLE: 0.1242 ✓", color: "#22c55e" },
];

const ORCETRA_LINES = [
  { t: 0, text: "$ orcetra predict train.csv --target SalePrice", color: "#00ff88" },
  { t: 30, text: "🎯 Orcetra v0.2.0", color: "#00d4ff" },
  { t: 36, text: "  Data: train.csv (1460 × 81)", color: "#94a3b8" },
  { t: 42, text: "  Auto-selected RMSLE (right-skewed target)", color: "#06b6d4" },
  { t: 60, text: "", color: "" },
  { t: 62, text: "Step 1: Analyzing data... ✓", color: "#94a3b8" },
  { t: 80, text: "Step 2: Running baselines...", color: "#94a3b8" },
  { t: 90, text: "  LinearRegression: 0.1968", color: "#64748b" },
  { t: 96, text: "  RandomForest:     0.1528 ⭐", color: "#64748b" },
  { t: 102, text: "  GradientBoosting: 0.1437 ⭐", color: "#fbbf24" },
  { t: 120, text: "", color: "" },
  { t: 122, text: "Step 3: AutoResearch loop (8x parallel)...", color: "#94a3b8" },
  { t: 130, text: "  Agent: LLM-guided (groq)", color: "#a78bfa" },
  { t: 150, text: "  #5  🎯 GBM(n=200,lr=0.05,d=5): 0.1361", color: "#22c55e" },
  { t: 170, text: "  #16 🎯 GBM(n=800,lr=0.03,d=5): 0.1328", color: "#22c55e" },
  { t: 190, text: "  #29 🎯 XGB(n=500,lr=0.05,d=8): 0.1302", color: "#22c55e" },
  { t: 210, text: "  #45 🎯 XGB(n=850,lr=0.02,d=3): 0.1269", color: "#22c55e" },
  { t: 230, text: "  #57 🎯 XGB(n=1000,lr=0.016):   0.1249", color: "#22c55e" },
  { t: 260, text: "", color: "" },
  { t: 262, text: "  80 strategies (67 unique), 10 improvements", color: "#94a3b8" },
  { t: 280, text: "", color: "" },
  { t: 282, text: "Top 5 strategies:", color: "#ffffff" },
  { t: 290, text: "  🏆 0.1249 — XGB(n=1000,lr=0.016,d=3)", color: "#00ff88" },
  { t: 298, text: "     0.1269 — XGB(n=850,lr=0.02,d=3)", color: "#94a3b8" },
  { t: 306, text: "     0.1302 — XGB(n=500,lr=0.05,d=8)", color: "#94a3b8" },
  { t: 320, text: "", color: "" },
  { t: 322, text: "🎯 +13.1% improvement over baseline", color: "#00ff88" },
  { t: 340, text: "Final: XGB = 0.1249  (60 seconds)", color: "#00ff88" },
];

// ── Timer ────────────────────────────────────────────────────────────

const Timer: React.FC<{ frame: number; maxSeconds: number; label: string; color: string }> = ({
  frame, maxSeconds, label, color,
}) => {
  const seconds = Math.min(frame / 30, maxSeconds);
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const display = mins > 0 ? `${mins}:${String(secs).padStart(2, "0")}` : `${secs}s`;
  return (
    <div style={{ fontSize: 28, color, fontFamily: "monospace", fontWeight: "bold" }}>
      {label} {display}
    </div>
  );
};

// ── Terminal Panel ───────────────────────────────────────────────────

const TerminalPanel: React.FC<{
  lines: typeof CLAUDE_LINES;
  frame: number;
  title: string;
  titleColor: string;
}> = ({ lines, frame, title, titleColor }) => {
  const visibleLines = lines.filter((l) => frame >= l.t);
  // Show last ~18 lines to simulate scrolling
  const displayLines = visibleLines.slice(-18);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#0d1117",
        borderRadius: 12,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Title bar */}
      <div
        style={{
          height: 44,
          backgroundColor: "#161b22",
          display: "flex",
          alignItems: "center",
          paddingLeft: 16,
          gap: 8,
          borderBottom: `2px solid ${titleColor}40`,
        }}
      >
        <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#ff5f57" }} />
        <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#febc2e" }} />
        <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#28c840" }} />
        <div style={{ marginLeft: 12, fontSize: 16, color: titleColor, fontWeight: "bold" }}>
          {title}
        </div>
      </div>
      {/* Terminal body */}
      <div style={{ flex: 1, padding: "12px 16px", overflow: "hidden" }}>
        {displayLines.map((line, i) => {
          const age = frame - line.t;
          const opacity = interpolate(age, [0, 8], [0, 1], { extrapolateRight: "clamp" });
          return (
            <div
              key={i}
              style={{
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                fontSize: 17,
                lineHeight: "26px",
                color: line.color,
                opacity,
                whiteSpace: "pre",
              }}
            >
              {line.text}
            </div>
          );
        })}
        {/* Blinking cursor */}
        <span
          style={{
            display: "inline-block",
            width: 10,
            height: 20,
            backgroundColor: frame % 30 < 15 ? "#00ff88" : "transparent",
            marginLeft: 2,
          }}
        />
      </div>
    </div>
  );
};

// ── Score Badge ──────────────────────────────────────────────────────

const ScoreBadge: React.FC<{
  frame: number;
  showAt: number;
  score: string;
  label: string;
  color: string;
  x: number;
}> = ({ frame, showAt, score, label, color, x }) => {
  const opacity = interpolate(frame, [showAt, showAt + 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const scale = interpolate(frame, [showAt, showAt + 15], [0.8, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        position: "absolute",
        bottom: 40,
        left: x,
        opacity,
        transform: `scale(${scale})`,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 18, color: "#94a3b8", marginBottom: 4 }}>{label}</div>
      <div
        style={{
          fontSize: 52,
          fontWeight: "bold",
          color,
          fontFamily: "monospace",
          textShadow: `0 0 20px ${color}40`,
        }}
      >
        {score}
      </div>
    </div>
  );
};

// ── Main Composition ─────────────────────────────────────────────────

export const SplitScreenDemo: React.FC = () => {
  const frame = useCurrentFrame();

  // VS badge
  const vsOpacity = interpolate(frame, [10, 30], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#000000",
        display: "flex",
        flexDirection: "column",
        padding: 24,
        gap: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          height: 60,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 40,
        }}
      >
        <Timer frame={frame} maxSeconds={180} label="⏱" color="#ff6b6b" />
        <div style={{ fontSize: 20, color: "#475569" }}>Kaggle House Prices — Who finishes first?</div>
        <Timer frame={frame} maxSeconds={60} label="⏱" color="#00ff88" />
      </div>

      {/* Split panels */}
      <div
        style={{
          flex: 1,
          display: "flex",
          gap: 16,
          position: "relative",
        }}
      >
        {/* Left: Claude */}
        <div style={{ flex: 1 }}>
          <TerminalPanel
            lines={CLAUDE_LINES}
            frame={frame}
            title="Human + Claude"
            titleColor="#8b5cf6"
          />
        </div>

        {/* VS divider */}
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 10,
            opacity: vsOpacity,
          }}
        >
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 28,
              backgroundColor: "#1e293b",
              border: "2px solid #334155",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 22,
              fontWeight: "bold",
              color: "#94a3b8",
            }}
          >
            VS
          </div>
        </div>

        {/* Right: Orcetra */}
        <div style={{ flex: 1 }}>
          <TerminalPanel
            lines={ORCETRA_LINES}
            frame={frame}
            title="Orcetra (1 command)"
            titleColor="#00ff88"
          />
        </div>
      </div>

      {/* Bottom scores */}
      <div style={{ height: 120, position: "relative" }}>
        <ScoreBadge
          frame={frame}
          showAt={700}
          score="0.1242"
          label="Human + Claude (3 min)"
          color="#8b5cf6"
          x={340}
        />
        <ScoreBadge
          frame={frame}
          showAt={700}
          score="0.1249"
          label="Orcetra (60 sec)"
          color="#00ff88"
          x={1140}
        />
      </div>
    </div>
  );
};
