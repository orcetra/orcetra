import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Audio,
  staticFile,
  spring,
} from "remotion";

// ── Terminal lines for left panel ─────────────────────────────────────
const TERMINAL_LINES = [
  { t: 0,   text: "$ orcetra evolve --budget 60 --data train.csv", color: "#00ff88" },
  { t: 20,  text: "", color: "" },
  { t: 22,  text: "🎯  Orcetra — Evolution Engine v0.2.0", color: "#00d4ff" },
  { t: 34,  text: "    Budget: 60s  |  Parallel: 8 workers", color: "#64748b" },
  { t: 48,  text: "", color: "" },
  { t: 50,  text: "── Baseline sweep ───────────────────────────────", color: "#1e3a5f" },
  { t: 58,  text: "  LGB baseline   0.3791  XGB  0.3812", color: "#94a3b8" },
  { t: 70,  text: "  RF  baseline   0.3908", color: "#94a3b8" },
  { t: 82,  text: "", color: "" },
  { t: 84,  text: "── Evolve loop (8x parallel) ────────────────────", color: "#1e3a5f" },
  { t: 94,  text: "  #03  XGB(n=200,lr=0.1,d=6)     0.3624  ↑", color: "#00d4ff" },
  { t: 106, text: "  #07  LGB(n=300,lr=0.05,d=8)    0.3551  ↑↑", color: "#00d4ff" },
  { t: 118, text: "  #12  XGB(n=500,lr=0.03,d=5)    0.3498  ↑↑↑", color: "#00ff88" },
  { t: 130, text: "  #18  XGB+poly_feat              0.3447  ↑↑↑", color: "#00ff88" },
  { t: 142, text: "  #24  ensemble(XGB+LGB)          0.3401  ↑↑↑↑", color: "#00ff88" },
  { t: 154, text: "  #31  ensemble+stack             0.3388  ↑↑↑↑", color: "#00ff88" },
  { t: 166, text: "  #38  tuned_ensemble_v2          0.3371  ★", color: "#00ff88" },
  { t: 178, text: "  #45  tuned_ensemble_v3          0.3358  ★★", color: "#00ff88" },
  { t: 192, text: "  ...", color: "#475569" },
  { t: 206, text: "  #67  adversarial_boost_v4       0.3291  ★★★", color: "#00ff88" },
  { t: 220, text: "", color: "" },
  { t: 222, text: "── Head-to-head: 406 OpenML datasets ───────────", color: "#1e3a5f" },
  { t: 232, text: "  vs FLAML:     wins 336/406 = 82.8%  ✓", color: "#00ff88" },
  { t: 244, text: "  vs AutoGluon: wins 229/406 = 56.4%  ✓", color: "#00ff88" },
  { t: 258, text: "", color: "" },
  { t: 260, text: "── Final ────────────────────────────────────────", color: "#1e3a5f" },
  { t: 270, text: "  67 strategies · 12 improvements found", color: "#94a3b8" },
  { t: 282, text: "  Best: adversarial_boost_v4", color: "#00ff88" },
  { t: 294, text: "  Score: 0.8619   Time: 58.7s  ✓", color: "#00ff88" },
  { t: 310, text: "  Cost: $0.00  (zero LLM calls)", color: "#64748b" },
  { t: 330, text: "$ █", color: "#00ff88" },
];

// ── Race chart ────────────────────────────────────────────────────────
// Each competitor has a "score curve" over frame time (0-900)
// Score drives bar width. We define keyframes for each.
interface Competitor {
  name: string;
  shortName: string;
  color: string;
  // bar score over frames (0→1 mapped to bar %)
  scoreAt: (rf: number) => number;
}

const COMPETITORS: Competitor[] = [
  {
    name: "AutoGluon",
    shortName: "AutoGluon",
    color: "#8b5cf6",
    scoreAt: (rf) => {
      // Starts strong, slows mid, ends behind Orcetra
      if (rf < 0) return 0;
      if (rf < 80)  return interpolate(rf, [0, 80],  [0, 0.61], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 220) return interpolate(rf, [80, 220], [0.61, 0.74], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 380) return interpolate(rf, [220, 380], [0.74, 0.80], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      return interpolate(rf, [380, 560], [0.80, 0.843], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    },
  },
  {
    name: "FLAML",
    shortName: "FLAML",
    color: "#f59e0b",
    scoreAt: (rf) => {
      if (rf < 0) return 0;
      if (rf < 50)  return interpolate(rf, [0, 50],  [0, 0.45], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 180) return interpolate(rf, [50, 180], [0.45, 0.74], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 340) return interpolate(rf, [180, 340], [0.74, 0.80], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      return interpolate(rf, [340, 560], [0.80, 0.824], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    },
  },
  {
    name: "Orcetra ✓",
    shortName: "Orcetra",
    color: "#00ff88",
    scoreAt: (rf) => {
      // Starts moderate, surges past others in second half
      if (rf < 0) return 0;
      if (rf < 60)  return interpolate(rf, [0, 60],   [0, 0.38], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 200) return interpolate(rf, [60, 200],  [0.38, 0.70], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 360) return interpolate(rf, [200, 360], [0.70, 0.82], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      if (rf < 500) return interpolate(rf, [360, 500], [0.82, 0.91], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
      return interpolate(rf, [500, 560], [0.91, 0.966], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    },
  },
];

// Sort by current score descending for rank label
function ranked(rf: number) {
  return [...COMPETITORS]
    .map(c => ({ c, score: c.scoreAt(rf) }))
    .sort((a, b) => b.score - a.score);
}

const SCORE_LABELS: Record<string, string> = {
  "AutoGluon": "0.8430",
  "FLAML":     "0.8241",
  "Orcetra ✓": "0.8619",
};

const RaceChart: React.FC<{ frame: number }> = ({ frame }) => {
  const { fps } = useVideoConfig();
  const CW = 864; const CH = 1080;
  // race starts after title ~frame 90
  const rf = Math.max(0, frame - 90);

  const containerOpacity = interpolate(frame, [90, 120], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  // Orcetra winner glow
  const orcScore = COMPETITORS[2].scoreAt(rf);
  const winnerGlow = interpolate(rf, [490, 560], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const pulse = 1 + Math.sin(frame * 0.22) * 0.04 * winnerGlow;

  // Budget bar progress 0→1 over rf 0→560
  const budgetP = interpolate(rf, [0, 560], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const budgetSec = Math.floor(budgetP * 60);

  // Rank labels
  const rankOrder = ranked(rf);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative", opacity: containerOpacity }}>
      {/* section title */}
      <div style={{
        position: "absolute", top: 32, left: 24, right: 24,
        fontSize: 13, color: "#1e3a5f", fontFamily: "monospace",
        letterSpacing: "0.08em",
      }}>LIVE BENCHMARK · 60s BUDGET</div>

      {/* Bars container */}
      <div style={{
        position: "absolute", top: 120, left: 20, right: 24,
        display: "flex", flexDirection: "column", gap: 52,
      }}>
        {rankOrder.map(({ c, score }, rank) => {
          const isOrcetra = c.name === "Orcetra ✓";
          const barW = score * 0.88; // max 88% of container

          const barOpacity = interpolate(rf, [0, 20], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

          const realScore = parseFloat(SCORE_LABELS[c.name]);
          const displayScore = interpolate(score, [0, 1], [0, realScore]).toFixed(4);

          const rankBounce = spring({ frame: rf - 400, fps, config: { damping: 18, stiffness: 140 } });
          const glowSize = isOrcetra && winnerGlow > 0.1
            ? `0 0 ${20 * winnerGlow}px ${c.color}80, 0 0 ${8 * winnerGlow}px ${c.color}`
            : "none";

          return (
            <div key={c.name} style={{ opacity: barOpacity }}>
              {/* Name + rank */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <div style={{
                  width: 22, height: 22, borderRadius: 4,
                  background: rank === 0 && rf > 480
                    ? `linear-gradient(135deg, ${c.color}cc, ${c.color}55)`
                    : "#1e293b",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: "bold", color: rank === 0 && rf > 480 ? "#000" : "#475569",
                  fontFamily: "monospace",
                  transform: isOrcetra && rf > 490 ? `scale(${rankBounce})` : "scale(1)",
                }}>
                  {rank + 1}
                </div>
                <div style={{
                  fontSize: 17, fontWeight: "bold", color: c.color,
                  fontFamily: "monospace",
                  textShadow: isOrcetra && winnerGlow > 0.2 ? `0 0 12px ${c.color}80` : "none",
                  transform: `scale(${isOrcetra ? pulse : 1})`,
                }}>
                  {c.name}
                </div>
                {isOrcetra && winnerGlow > 0.5 && (
                  <div style={{
                    fontSize: 13, color: "#00ff88", fontFamily: "monospace",
                    border: "1px solid #00ff8840", padding: "1px 8px", borderRadius: 3,
                    opacity: winnerGlow,
                  }}>
                    WINNER
                  </div>
                )}
              </div>

              {/* Bar track */}
              <div style={{
                height: 36, backgroundColor: "#0d1520", borderRadius: 6,
                overflow: "hidden", position: "relative",
                border: isOrcetra && winnerGlow > 0.2 ? `1px solid ${c.color}40` : "1px solid #1e293b",
              }}>
                {/* Filled portion */}
                <div style={{
                  position: "absolute", left: 0, top: 0, bottom: 0,
                  width: `${barW * 100}%`,
                  background: isOrcetra
                    ? `linear-gradient(90deg, ${c.color}99, ${c.color}dd)`
                    : `linear-gradient(90deg, ${c.color}55, ${c.color}88)`,
                  borderRadius: 6,
                  boxShadow: glowSize,
                  transition: "none",
                }} />
                {/* Score label inside bar */}
                <div style={{
                  position: "absolute", right: 10, top: 0, bottom: 0,
                  display: "flex", alignItems: "center",
                  fontSize: 15, fontWeight: "bold", color: c.color,
                  fontFamily: "monospace", opacity: 0.9,
                }}>
                  {displayScore}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Budget bar at bottom */}
      <div style={{
        position: "absolute", bottom: 120, left: 20, right: 24,
        opacity: interpolate(rf, [0, 25], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
          <span style={{ fontSize: 13, color: "#475569", fontFamily: "monospace" }}>⏱ 60s budget</span>
          <span style={{ fontSize: 13, color: "#00d4ff", fontFamily: "monospace" }}>
            {budgetSec}s / 60s
          </span>
        </div>
        <div style={{ height: 7, backgroundColor: "#0d1520", borderRadius: 4, overflow: "hidden", border: "1px solid #1e293b" }}>
          <div style={{
            height: "100%", width: `${budgetP * 100}%`,
            background: "linear-gradient(90deg, #00ff88, #00d4ff)",
            borderRadius: 4, boxShadow: "0 0 8px #00ff8840",
          }} />
        </div>
      </div>

      {/* Mini stat callouts at the bottom */}
      <div style={{
        position: "absolute", bottom: 48, left: 20, right: 24,
        display: "flex", gap: 16,
        opacity: interpolate(rf, [500, 560], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
      }}>
        {[
          { v: "83%",  l: "win rate vs FLAML", c: "#00ff88" },
          { v: "406",  l: "datasets tested",   c: "#00d4ff" },
          { v: "$0",   l: "LLM cost",           c: "#E8734A" },
        ].map(({ v, l, c }) => (
          <div key={l} style={{ flex: 1, textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 900, color: c, fontFamily: "monospace", textShadow: `0 0 16px ${c}60` }}>
              {v}
            </div>
            <div style={{ fontSize: 12, color: "#ffffff50", fontFamily: "monospace", marginTop: 2 }}>{l}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Subtitle ──────────────────────────────────────────────────────────
const Sub: React.FC<{ frame: number; showAt: number; hideAt: number; text: string }> = ({ frame, showAt, hideAt, text }) => {
  const opacity = interpolate(frame, [showAt, showAt + 8, hideAt - 6, hideAt], [0, 1, 1, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  if (frame < showAt || frame > hideAt) return null;
  return (
    <div style={{
      position: "absolute", bottom: 44, left: "50%", transform: "translateX(-50%)",
      opacity, background: "rgba(0,0,0,0.78)", borderLeft: "4px solid #00d4ff",
      padding: "10px 32px", fontSize: 24, color: "#fff", fontFamily: "monospace",
      letterSpacing: "0.04em", whiteSpace: "nowrap",
    }}>{text}</div>
  );
};

// ── CTA ───────────────────────────────────────────────────────────────
const CTAOverlay: React.FC<{ frame: number; showAt: number }> = ({ frame, showAt }) => {
  const { fps } = useVideoConfig();
  const opacity = interpolate(frame, [showAt, showAt + 20], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const scale = spring({ frame: frame - showAt, fps, config: { damping: 20, stiffness: 120 } });
  if (frame < showAt) return null;
  return (
    <div style={{
      position: "absolute", inset: 0, display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", opacity,
      background: "radial-gradient(ellipse 80% 60% at center, rgba(0,0,0,0.92) 50%, transparent 100%)",
      pointerEvents: "none",
    }}>
      <div style={{ transform: `scale(${scale})`, display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{ fontSize: 78, fontWeight: 900, color: "#00d4ff", letterSpacing: "0.2em", fontFamily: "monospace", textShadow: "0 0 40px #00d4ff80" }}>
          ORCETRA
        </div>
        <div style={{ fontSize: 24, color: "#ffffff70", marginTop: 8, fontFamily: "monospace" }}>
          Beat FLAML 83% of the time. In 60 seconds.
        </div>
        <div style={{ marginTop: 28, fontSize: 22, color: "#00ff88", fontFamily: "monospace", border: "1px solid #00ff8840", padding: "10px 28px", borderRadius: 4 }}>
          $ pip install orcetra
        </div>
        <div style={{ marginTop: 14, fontSize: 20, color: "#ffffff50", fontFamily: "monospace" }}>orcetra.ai</div>
      </div>
    </div>
  );
};

// ── Main ──────────────────────────────────────────────────────────────
export const Video2AutoML: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const visibleLines = TERMINAL_LINES.filter(l => frame >= l.t);
  const displayLines = visibleLines.slice(-22);

  const volume = interpolate(frame, [0, 10, durationInFrames - 60, durationInFrames], [0, 0.68, 0.68, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  const titleOpacity = interpolate(frame, [0, 15, 60, 88], [1, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const splitOpacity = interpolate(frame, [80, 112], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <div style={{ width: "100%", height: "100%", backgroundColor: "#0a0f1a", fontFamily: "monospace", position: "relative", overflow: "hidden" }}>
      <Audio src={staticFile("music2-upbeat.mp3")} volume={volume} />

      {/* ── SPLIT LAYOUT: Left terminal 55% | Right race chart 45% ── */}
      <div style={{ position: "absolute", inset: 0, display: "flex", opacity: splitOpacity, backgroundColor: "#0a0f1a" }}>

        {/* Left: terminal 55% */}
        <div style={{ flex: "0 0 55%", height: "100%", padding: "28px 28px 28px 44px", display: "flex", flexDirection: "column", backgroundColor: "#0a0f1a" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#ff5f57" }} />
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#febc2e" }} />
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#28c840" }} />
            <div style={{ marginLeft: 14, fontSize: 13, color: "#334155", letterSpacing: "0.05em" }}>
              orcetra — adversarial evolve · 60s budget
            </div>
          </div>
          <div style={{ flex: 1, overflow: "hidden" }}>
            {displayLines.map((line, i) => {
              const age = frame - line.t;
              const lo = interpolate(age, [0, 6], [0, 1], { extrapolateRight: "clamp" });
              return (
                <div key={`${line.t}-${i}`} style={{
                  fontSize: 15.5, lineHeight: "25px", color: line.color || "transparent",
                  opacity: lo, whiteSpace: "pre",
                }}>{line.text}</div>
              );
            })}
            {frame < 340 && (
              <span style={{ display: "inline-block", width: 9, height: 18, backgroundColor: frame % 30 < 15 ? "#00ff88" : "transparent" }} />
            )}
          </div>
        </div>

        {/* Right: race chart 45% */}
        <div style={{ flex: "1 1 auto", height: "100%", position: "relative", backgroundColor: "#0a0f1a" }}>

          <RaceChart frame={frame} />
        </div>
      </div>

      {/* ── Title card ── */}
      <div style={{
        position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
        opacity: titleOpacity, backgroundColor: frame < 85 ? "#0a0f1a" : "transparent", pointerEvents: "none",
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 20, color: "#334155", fontFamily: "monospace", marginBottom: 16, letterSpacing: "0.15em" }}>
            AUTOML BENCHMARK
          </div>
          <div style={{ fontSize: 52, fontWeight: 700, color: "#ffffff", opacity: interpolate(frame, [0, 18], [0, 1], { extrapolateRight: "clamp" }) }}>
            What if AutoML
          </div>
          <div style={{ fontSize: 52, fontWeight: 700, color: "#00d4ff", textShadow: "0 0 30px #00d4ff80", opacity: interpolate(frame, [10, 28], [0, 1], { extrapolateRight: "clamp" }) }}>
            took 60 seconds?
          </div>
        </div>
      </div>

      {/* ── CTA ── */}
      <CTAOverlay frame={frame} showAt={752} />

      {/* ── Subtitles ── */}
      <Sub frame={frame} showAt={108} hideAt={240} text="Same dataset · same 60-second time budget" />
      <Sub frame={frame} showAt={340} hideAt={445} text="Orcetra: adversarial evolution · 8 parallel workers" />
      <Sub frame={frame} showAt={540} hideAt={640} text="406 OpenML datasets benchmarked head-to-head" />
    </div>
  );
};
