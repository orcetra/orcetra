import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Audio,
  staticFile,
  spring,
} from "remotion";

// ── Seeded deterministic random ───────────────────────────────────────
const sr = (seed: number) => {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

// ── Terminal lines ────────────────────────────────────────────────────
const TERMINAL_LINES = [
  { t: 0,   text: "$ orcetra predict --market polymarket --all", color: "#00ff88" },
  { t: 20,  text: "", color: "" },
  { t: 22,  text: "🎯  Orcetra v0.2.0  |  Polymarket Integration", color: "#00d4ff" },
  { t: 30,  text: "    Loading markets...", color: "#64748b" },
  { t: 50,  text: "    Found 3,145 resolved prediction markets", color: "#94a3b8" },
  { t: 60,  text: "", color: "" },
  { t: 62,  text: "── Market Scan ──────────────────────────────────", color: "#1e3a5f" },
  { t: 70,  text: "  [→] Golf: US Open 2024 winner?", color: "#64748b" },
  { t: 82,  text: "       Orcetra: Scheffler (p=0.74)  ✓  CORRECT", color: "#00ff88" },
  { t: 96,  text: "  [→] Politics: Senate seat flip AZ?", color: "#64748b" },
  { t: 109, text: "       Orcetra: NO (p=0.61)         ✓  CORRECT", color: "#00ff88" },
  { t: 123, text: "  [→] Economy: Fed cut Sept 2024?", color: "#64748b" },
  { t: 136, text: "       Orcetra: YES (p=0.68)        ✓  CORRECT", color: "#00ff88" },
  { t: 149, text: "  [→] Sports: Chiefs win AFC?", color: "#64748b" },
  { t: 162, text: "       Orcetra: YES (p=0.71)        ✓  CORRECT", color: "#00ff88" },
  { t: 175, text: "  [→] Economy: BTC above 60k Oct?", color: "#64748b" },
  { t: 188, text: "       Orcetra: NO (p=0.52)         ✗  WRONG",  color: "#ff4444" },
  { t: 201, text: "  [→] Politics: Harris wins Iowa?", color: "#64748b" },
  { t: 214, text: "       Orcetra: NO (p=0.78)         ✓  CORRECT", color: "#00ff88" },
  { t: 227, text: "  [→] Golf: Ryder Cup Europe win?", color: "#64748b" },
  { t: 240, text: "       Orcetra: YES (p=0.55)        ✓  CORRECT", color: "#00ff88" },
  { t: 253, text: "  ...", color: "#475569" },
  { t: 270, text: "── Results ──────────────────────────────────────", color: "#1e3a5f" },
  { t: 280, text: "  Category    Correct  Total  Beat Rate", color: "#64748b" },
  { t: 290, text: "  ──────────────────────────────────────────", color: "#1e3a5f" },
  { t: 300, text: "  Golf        107/119  89.7%  ████████▉", color: "#00d4ff" },
  { t: 314, text: "  Politics    258/333  77.4%  ███████▋ ", color: "#00d4ff" },
  { t: 328, text: "  Economy     448/653  68.6%  ██████▊  ", color: "#00d4ff" },
  { t: 342, text: "  Sports      510/759  67.2%  ██████▋  ", color: "#00d4ff" },
  { t: 358, text: "  ──────────────────────────────────────────", color: "#1e3a5f" },
  { t: 368, text: "  OVERALL  2093/3145  67.0%   BEATS MKT ✓", color: "#00ff88" },
  { t: 387, text: "  Zero human input. Zero LLM cost.", color: "#94a3b8" },
  { t: 402, text: "  Runtime: 4m 12s  |  Cost: $0.00", color: "#64748b" },
  { t: 422, text: "$ █", color: "#00ff88" },
];

// ── Particle definitions ──────────────────────────────────────────────
interface Particle {
  id: number; x0: number; y0: number;
  quality: number; label: string; color: string;
  driftPhase: number; driftScale: number;
}

const PARTICLES: Particle[] = Array.from({ length: 50 }, (_, i) => {
  const q = 0.1 + sr(i * 5 + 3) * 0.88;
  const col = q >= 0.60 ? "#00ff88" : q >= 0.38 ? "#00d4ff" : "#E8734A";
  return {
    id: i,
    x0: sr(i * 5 + 1) * 0.84 + 0.06,
    y0: sr(i * 5 + 2) * 0.78 + 0.08,
    quality: q,
    label: (0.28 + sr(i * 5 + 4) * 0.62).toFixed(2),
    color: col,
    driftPhase: sr(i * 7 + 1) * 6.28,
    driftScale: 0.4 + sr(i * 7 + 2) * 0.9,
  };
});

// ── Particle field ────────────────────────────────────────────────────
const ParticleField: React.FC<{ frame: number }> = ({ frame }) => {
  const { fps } = useVideoConfig();
  const PW = 864; const PH = 1080;
  const cx = PW * 0.50; const cy = PH * 0.42;
  const rf = Math.max(0, frame - 90); // local time after terminal appears

  type PState = { x: number; y: number; r: number; opacity: number };

  const states: PState[] = PARTICLES.map(p => {
    const d = 14 * p.driftScale;
    const dx = Math.sin(rf * 0.016 + p.driftPhase) * d;
    const dy = Math.cos(rf * 0.011 + p.driftPhase * 1.37) * d * 0.7;
    const rawX = p.x0 * PW + dx;
    const rawY = p.y0 * PH + dy;

    const convDelay = 240 + p.quality * 80;
    const convP = p.quality >= 0.55
      ? interpolate(rf, [convDelay, convDelay + 280], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
      : 0;

    const x = rawX + (cx - rawX) * convP;
    const y = rawY + (cy - rawY) * convP;

    let opacity: number;
    if (p.quality < 0.38) {
      opacity = interpolate(rf, [10, 120], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    } else if (p.quality < 0.55) {
      opacity = interpolate(rf, [60, 240], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    } else {
      opacity = interpolate(rf, [540, 620], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    }

    const r = (2.5 + p.quality * 6) * (1 + convP * 1.8);
    return { x, y, r, opacity };
  });

  const highQ = PARTICLES.map((p, i) => ({ p, v: states[i] })).filter(({ p }) => p.quality >= 0.55);

  const lineAlpha = interpolate(rf, [0, 30, 280, 430], [0, 0.5, 0.5, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  // Blob merge
  const mergeP = interpolate(rf, [520, 615], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const mergePop = spring({ frame: rf - 520, fps, config: { damping: 14, stiffness: 200 } });
  const blobR = 11 * Math.min(1, mergePop);
  const glow1 = 85 * mergeP * (1 + Math.sin(frame * 0.12) * 0.18);
  const glow2 = 40 * mergeP;
  const labelO = interpolate(rf, [590, 650], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${PW} ${PH}`} style={{ position: "absolute", inset: 0 }}>
      {/* Subtle background vignette grid */}
      <defs>
        <radialGradient id="vg" cx="50%" cy="42%" r="60%">
          <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.04" />
          <stop offset="100%" stopColor="#0a0f1a" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width={PW} height={PH} fill="url(#vg)" />

      {/* Network lines */}
      {lineAlpha > 0.01 && highQ.flatMap(({ v: v1, p: p1 }, i) =>
        highQ.slice(i + 1).map(({ v: v2, p: p2 }) => {
          const dist = Math.hypot(v1.x - v2.x, v1.y - v2.y);
          if (dist > 200) return null;
          const lo = (1 - dist / 200) * lineAlpha * Math.min(v1.opacity, v2.opacity) * 0.4;
          if (lo < 0.015) return null;
          return (
            <line key={`ln-${p1.id}-${p2.id}`}
              x1={v1.x} y1={v1.y} x2={v2.x} y2={v2.y}
              stroke="#00d4ff" strokeWidth={0.7} opacity={lo} />
          );
        })
      )}

      {/* Particles */}
      {PARTICLES.map((p, i) => {
        const v = states[i];
        if (v.opacity < 0.02) return null;
        return (
          <g key={p.id}>
            <circle cx={v.x} cy={v.y} r={v.r * 3} fill={p.color} opacity={v.opacity * 0.07} />
            <circle cx={v.x} cy={v.y} r={v.r * 1.6} fill={p.color} opacity={v.opacity * 0.16} />
            <circle cx={v.x} cy={v.y} r={v.r} fill={p.color} opacity={v.opacity * 0.93} />
            {v.opacity > 0.3 && (
              <text x={v.x + v.r + 5} y={v.y + 4}
                fontSize={11} fill={p.color} opacity={v.opacity * 0.55} fontFamily="monospace">
                {p.label}
              </text>
            )}
          </g>
        );
      })}

      {/* Convergence blob */}
      {mergeP > 0.01 && (
        <g>
          <circle cx={cx} cy={cy} r={glow1} fill="#00ff88" opacity={0.04 * mergeP} />
          <circle cx={cx} cy={cy} r={glow2} fill="#00ff88" opacity={0.10 * mergeP} />
          <circle cx={cx} cy={cy} r={blobR * 2.4} fill="#00ff88" opacity={0.26 * mergeP} />
          <circle cx={cx} cy={cy} r={blobR} fill="#00ff88" opacity={0.97} />
          <circle cx={cx} cy={cy} r={blobR * 3.8 + Math.sin(frame * 0.15) * 4}
            fill="none" stroke="#00ff88" strokeWidth={1.3} opacity={0.28 * mergeP} />
          {labelO > 0.01 && (
            <>
              <text x={cx} y={cy - 56} textAnchor="middle"
                fontSize={23} fontWeight="bold" fill="#00ff88"
                opacity={labelO} fontFamily="monospace">
                67% Beat Rate ✓
              </text>
              <text x={cx} y={cy + 56} textAnchor="middle"
                fontSize={14} fill="#00d4ff" opacity={labelO * 0.8} fontFamily="monospace">
                2,093 / 3,145 verified
              </text>
            </>
          )}
        </g>
      )}
    </svg>
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
        <div style={{ fontSize: 26, color: "#ffffff70", marginTop: 8, fontFamily: "monospace" }}>
          AutoML · Zero Cost · Real Results
        </div>
        <div style={{ marginTop: 28, fontSize: 22, color: "#00ff88", fontFamily: "monospace", border: "1px solid #00ff8840", padding: "10px 28px", borderRadius: 4 }}>
          $ pip install orcetra
        </div>
        <div style={{ marginTop: 14, fontSize: 20, color: "#ffffff50", fontFamily: "monospace" }}>orcetra.ai</div>
      </div>
    </div>
  );
};

// ── Sub ───────────────────────────────────────────────────────────────
const Sub: React.FC<{ frame: number; showAt: number; hideAt: number; text: string }> = ({ frame, showAt, hideAt, text }) => {
  const opacity = interpolate(frame, [showAt, showAt + 8, hideAt - 6, hideAt], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
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

// ── Main ──────────────────────────────────────────────────────────────
export const Video1BeatMarket: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const visibleLines = TERMINAL_LINES.filter(l => frame >= l.t);
  const displayLines = visibleLines.slice(-24);

  const volume = interpolate(frame, [0, 10, durationInFrames - 60, durationInFrames], [0, 0.68, 0.68, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  const titleOpacity = interpolate(frame, [0, 15, 62, 85], [1, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const splitOpacity = interpolate(frame, [78, 112], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <div style={{ width: "100%", height: "100%", backgroundColor: "#0a0f1a", fontFamily: "monospace", position: "relative", overflow: "hidden" }}>
      <Audio src={staticFile("music1-chill.mp3")} volume={volume} />

      {/* ── SPLIT LAYOUT: Left terminal 55% | Right particles 45% ── */}
      <div style={{ position: "absolute", inset: 0, display: "flex", opacity: splitOpacity, backgroundColor: "#0a0f1a" }}>

        {/* Left: terminal 55% */}
        <div style={{ flex: "0 0 55%", height: "100%", padding: "28px 32px 28px 44px", display: "flex", flexDirection: "column", backgroundColor: "#0a0f1a" }}>
          {/* Terminal chrome */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#ff5f57" }} />
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#febc2e" }} />
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: "#28c840" }} />
            <div style={{ marginLeft: 14, fontSize: 13, color: "#334155", letterSpacing: "0.05em" }}>
              orcetra — polymarket prediction audit
            </div>
          </div>
          {/* Lines */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            {displayLines.map((line, i) => {
              const age = frame - line.t;
              const lo = interpolate(age, [0, 6], [0, 1], { extrapolateRight: "clamp" });
              return (
                <div key={`${line.t}-${i}`} style={{
                  fontSize: 16, lineHeight: "26px", color: line.color || "transparent",
                  opacity: lo, whiteSpace: "pre",
                }}>{line.text}</div>
              );
            })}
            {frame < 430 && (
              <span style={{ display: "inline-block", width: 9, height: 18, backgroundColor: frame % 30 < 15 ? "#00ff88" : "transparent" }} />
            )}
          </div>
        </div>

        {/* Right: particle viz 45% */}
        <div style={{ flex: "1 1 auto", height: "100%", position: "relative", backgroundColor: "#0a0f1a" }}>

          {/* Small section label */}
          <div style={{
            position: "absolute", top: 28, right: 44, zIndex: 3,
            fontSize: 12, color: "#1e3a5f", fontFamily: "monospace", letterSpacing: "0.08em",
            opacity: interpolate(frame, [110, 140], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
          }}>
            PREDICTION CONVERGENCE
          </div>
          <ParticleField frame={frame} />
        </div>
      </div>

      {/* ── Title card (fades out at ~85) ── */}
      <div style={{
        position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
        opacity: titleOpacity, backgroundColor: frame < 82 ? "#0a0f1a" : "transparent", pointerEvents: "none",
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 52, fontWeight: 700, color: "#ffffff", opacity: interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" }) }}>
            Can AI Beat
          </div>
          <div style={{ fontSize: 52, fontWeight: 700, color: "#00d4ff", textShadow: "0 0 30px #00d4ff80", opacity: interpolate(frame, [8, 28], [0, 1], { extrapolateRight: "clamp" }) }}>
            Prediction Markets?
          </div>
        </div>
      </div>

      {/* ── CTA ── */}
      <CTAOverlay frame={frame} showAt={750} />

      {/* ── Subtitles ── */}
      <Sub frame={frame} showAt={92}  hideAt={182} text="Running live against 3,145 Polymarket predictions..." />
      <Sub frame={frame} showAt={275} hideAt={368} text="Results: 2,093 correct out of 3,145" />
      <Sub frame={frame} showAt={425} hideAt={505} text="Zero human input · Zero LLM cost" />
    </div>
  );
};
