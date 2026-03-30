import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Audio,
  staticFile,
  spring,
} from "remotion";

// ── Background terminal (ambient, slow scroll) ─────────────────────
const BG_LINES = [
  { t: 0,   text: "$ orcetra benchmark --datasets openml --n 670", color: "#00d4ff18" },
  { t: 8,   text: "  Loading 670 OpenML datasets...", color: "#ffffff10" },
  { t: 16,  text: "  Splitting train/test folds...", color: "#ffffff10" },
  { t: 24,  text: "  [001/670] dataset=1464  acc=0.9312  ✓", color: "#00ff8820" },
  { t: 30,  text: "  [002/670] dataset=40984  acc=0.8741  ✓", color: "#00ff8820" },
  { t: 36,  text: "  [003/670] dataset=1497  acc=0.7863  ✓", color: "#00ff8820" },
  { t: 42,  text: "  [004/670] dataset=458   acc=0.9104  ✓", color: "#00ff8820" },
  { t: 48,  text: "  [005/670] dataset=1169  acc=0.8299  ✓", color: "#00ff8820" },
  { t: 54,  text: "  [006/670] dataset=23517  acc=0.7621  ✓", color: "#00ff8820" },
  { t: 60,  text: "  [007/670] dataset=31   acc=0.8834  ✓", color: "#00ff8820" },
  { t: 66,  text: "  ...", color: "#ffffff10" },
  { t: 72,  text: "  [100/670] benchmarked  avg=0.8491", color: "#00d4ff18" },
  { t: 80,  text: "  [200/670] benchmarked  avg=0.8312", color: "#00d4ff18" },
  { t: 88,  text: "  [300/670] benchmarked  avg=0.8271", color: "#00d4ff18" },
  { t: 96,  text: "  [400/670] benchmarked  avg=0.8248", color: "#00d4ff18" },
  { t: 104, text: "  [500/670] benchmarked  avg=0.8239", color: "#00d4ff18" },
  { t: 112, text: "  [600/670] benchmarked  avg=0.8234", color: "#00d4ff18" },
  { t: 120, text: "  [670/670] benchmarked  avg=0.8231  ✓", color: "#00ff8820" },
  { t: 128, text: "", color: "" },
  { t: 130, text: "  FLAML  comparison: 406 datasets overlap", color: "#ffffff10" },
  { t: 138, text: "  Orcetra wins: 336/406 = 82.8%", color: "#00ff8820" },
  { t: 146, text: "  AutoGluon wins: 163/406 = 40.1%", color: "#ffffff10" },
  { t: 154, text: "", color: "" },
  { t: 156, text: "  Polymarket audit: 3145 predictions", color: "#ffffff10" },
  { t: 164, text: "  Correct: 2093  Wrong: 1052  Rate: 67.0%", color: "#00ff8820" },
  { t: 172, text: "  LLM calls: 0   API cost: $0.00", color: "#00d4ff18" },
  { t: 180, text: "", color: "" },
  { t: 182, text: "  All results reproducible. Zero human labels.", color: "#ffffff10" },
  { t: 190, text: "  $ █", color: "#00ff8820" },
  // loop more lines
  { t: 220, text: "$ orcetra predict --market polymarket --all", color: "#00d4ff18" },
  { t: 228, text: "  [0001/3145] golf:us-open-2024  → YES  ✓", color: "#00ff8820" },
  { t: 236, text: "  [0002/3145] politics:senate-az → NO   ✓", color: "#00ff8820" },
  { t: 244, text: "  [0003/3145] economy:fed-sept  → YES  ✓", color: "#00ff8820" },
  { t: 252, text: "  [0004/3145] sports:chiefs-afc → YES  ✓", color: "#00ff8820" },
  { t: 260, text: "  ...", color: "#ffffff10" },
  { t: 300, text: "  [3145/3145] complete  beat_rate=67.0%  ✓", color: "#00ff8820" },
  { t: 308, text: "  Total cost: $0.00   Time: 4m12s", color: "#00d4ff18" },
];

// ── Number slide ───────────────────────────────────────────────────
interface NumberSlide {
  showAt: number;
  hideAt: number;
  target: number;
  prefix?: string;
  suffix?: string;
  label: string;
  sublabel: string;
  color: string;
}

const SLIDES: NumberSlide[] = [
  { showAt: 60,  hideAt: 240, target: 3145, suffix: "",  label: "predictions verified",      sublabel: "against resolved Polymarket markets",      color: "#00d4ff" },
  { showAt: 240, hideAt: 420, target: 67,   suffix: "%", label: "beat the market consensus", sublabel: "2,093 correct · zero human input",           color: "#00ff88" },
  { showAt: 420, hideAt: 540, target: 406,  suffix: "",  label: "datasets benchmarked",       sublabel: "head-to-head vs FLAML · OpenML",             color: "#00d4ff" },
  { showAt: 540, hideAt: 660, target: 83,   suffix: "%", label: "win rate vs FLAML",          sublabel: "336/406 datasets · 60s budget each",         color: "#00ff88" },
  { showAt: 660, hideAt: 780, target: 0,    prefix: "$", suffix: "", label: "LLM cost per prediction", sublabel: "pure evolutionary search · no GPT calls", color: "#E8734A" },
];

const NumberSlideComp: React.FC<{ slide: NumberSlide; frame: number }> = ({ slide, frame }) => {
  const { fps } = useVideoConfig();
  const { showAt, hideAt, target, prefix = "", suffix, label, sublabel, color } = slide;

  const opacity = interpolate(
    frame,
    [showAt, showAt + 15, hideAt - 15, hideAt],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const progress = spring({
    frame: frame - (showAt + 10),
    fps,
    config: { damping: 22, stiffness: 80, mass: 0.6 },
  });

  const currentNumber = Math.floor(interpolate(progress, [0, 1], [0, target], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }));

  const scale = spring({
    frame: frame - showAt,
    fps,
    config: { damping: 20, stiffness: 100 },
  });

  if (frame < showAt || frame > hideAt) return null;

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        pointerEvents: "none",
      }}
    >
      {/* radial dark bg behind number */}
      <div
        style={{
          background: "radial-gradient(ellipse 65% 55% at center, rgba(0,0,0,0.87) 35%, transparent 100%)",
          padding: "36px 100px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          transform: `scale(${scale})`,
        }}
      >
        <div
          style={{
            fontSize: 170,
            fontWeight: 900,
            fontFamily: "'JetBrains Mono', monospace",
            color,
            textShadow: `0 0 80px ${color}70, 0 0 160px ${color}30`,
            lineHeight: 1,
            letterSpacing: "-0.04em",
          }}
        >
          {prefix}{currentNumber.toLocaleString()}{suffix}
        </div>
        <div
          style={{
            fontSize: 34,
            color: "#ffffffcc",
            marginTop: 18,
            fontFamily: "monospace",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
          }}
        >
          {label}
        </div>
        <div
          style={{
            fontSize: 22,
            color: "#ffffff55",
            marginTop: 12,
            fontFamily: "monospace",
            letterSpacing: "0.06em",
            opacity: interpolate(frame, [showAt + 25, showAt + 45], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          {sublabel}
        </div>
      </div>
    </div>
  );
};

// ── Final grid (all numbers together) ─────────────────────────────
const FinalGrid: React.FC<{ frame: number; showAt: number }> = ({ frame, showAt }) => {
  const { fps } = useVideoConfig();
  const opacity = interpolate(frame, [showAt, showAt + 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  if (frame < showAt) return null;

  const items = [
    { n: "3,145", label: "predictions verified",  color: "#00d4ff" },
    { n: "67%",   label: "beat the market",        color: "#00ff88" },
    { n: "406",   label: "datasets benchmarked",   color: "#00d4ff" },
    { n: "83%",   label: "win rate vs FLAML",       color: "#00ff88" },
    { n: "$0",    label: "LLM cost",                color: "#E8734A" },
  ];

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        background: "radial-gradient(ellipse 90% 80% at center, rgba(0,0,0,0.9) 50%, transparent 100%)",
        pointerEvents: "none",
      }}
    >
      {/* Number grid */}
      <div
        style={{
          display: "flex",
          gap: 48,
          marginBottom: 52,
        }}
      >
        {items.map((item, i) => {
          const itemOpacity = interpolate(
            frame,
            [showAt + i * 8, showAt + i * 8 + 18],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const itemScale = spring({
            frame: frame - (showAt + i * 8),
            fps,
            config: { damping: 22, stiffness: 120 },
          });
          return (
            <div
              key={item.n}
              style={{
                textAlign: "center",
                opacity: itemOpacity,
                transform: `scale(${itemScale})`,
              }}
            >
              <div
                style={{
                  fontSize: 64,
                  fontWeight: 900,
                  fontFamily: "monospace",
                  color: item.color,
                  textShadow: `0 0 30px ${item.color}60`,
                  lineHeight: 1,
                }}
              >
                {item.n}
              </div>
              <div
                style={{
                  fontSize: 15,
                  color: "#ffffff60",
                  fontFamily: "monospace",
                  marginTop: 8,
                  letterSpacing: "0.05em",
                }}
              >
                {item.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* ORCETRA logo */}
      <div
        style={{
          opacity: interpolate(frame, [showAt + 40, showAt + 60], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div
          style={{
            fontSize: 72,
            fontWeight: 900,
            color: "#00d4ff",
            fontFamily: "monospace",
            letterSpacing: "0.25em",
            textShadow: "0 0 40px #00d4ff80",
            textAlign: "center",
          }}
        >
          ORCETRA
        </div>
        <div
          style={{
            fontSize: 22,
            color: "#ffffff40",
            fontFamily: "monospace",
            textAlign: "center",
            marginTop: 6,
            letterSpacing: "0.1em",
          }}
        >
          orcetra.ai  ·  pip install orcetra
        </div>
      </div>
    </div>
  );
};

// ── Typewriter intro ───────────────────────────────────────────────
const TypewriterIntro: React.FC<{ frame: number }> = ({ frame }) => {
  const text = "No hype. Just numbers.";
  const charsVisible = Math.floor(interpolate(frame, [5, 50], [0, text.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }));
  const opacity = interpolate(frame, [0, 5, 45, 60], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        backgroundColor: "#0a0f1a",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          fontSize: 56,
          fontWeight: 700,
          color: "#ffffff",
          fontFamily: "monospace",
          letterSpacing: "0.04em",
        }}
      >
        {text.slice(0, charsVisible)}
        {charsVisible < text.length && (
          <span
            style={{
              display: "inline-block",
              width: 3,
              height: 56,
              backgroundColor: frame % 16 < 8 ? "#00d4ff" : "transparent",
              marginLeft: 4,
              verticalAlign: "middle",
            }}
          />
        )}
      </div>
    </div>
  );
};

// ── Subtitle ───────────────────────────────────────────────────────
const Sub: React.FC<{ frame: number; showAt: number; hideAt: number; text: string }> = ({
  frame, showAt, hideAt, text,
}) => {
  const opacity = interpolate(frame, [showAt, showAt + 8, hideAt - 6, hideAt], [0, 1, 1, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  if (frame < showAt || frame > hideAt) return null;
  return (
    <div
      style={{
        position: "absolute",
        bottom: 44,
        left: "50%",
        transform: "translateX(-50%)",
        opacity,
        background: "rgba(0,0,0,0.75)",
        borderLeft: "4px solid #E8734A",
        padding: "10px 32px",
        fontSize: 24,
        color: "#ffffff",
        fontFamily: "monospace",
        letterSpacing: "0.04em",
        whiteSpace: "nowrap",
      }}
    >
      {text}
    </div>
  );
};

// ── Main ───────────────────────────────────────────────────────────
export const Video3Numbers: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const volume = interpolate(
    frame,
    [0, 10, durationInFrames - 60, durationInFrames],
    [0, 0.75, 0.75, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // BG terminal visible during number slides
  const bgOpacity = interpolate(frame, [55, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const bgLines = BG_LINES.filter((l) => frame >= l.t).slice(-32);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#0a0f1a",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <Audio src={staticFile("music3-energetic.mp3")} volume={volume} />

      {/* ── Background ambient terminal ── */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          padding: "32px 48px",
          opacity: bgOpacity,
        }}
      >
        <div
          style={{
            height: 30,
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 12,
          }}
        >
          <div style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: "#ff5f57" }} />
          <div style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: "#febc2e" }} />
          <div style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: "#28c840" }} />
          <div style={{ marginLeft: 12, fontSize: 12, color: "#1e3a5f", letterSpacing: "0.05em" }}>
            orcetra — full benchmark run
          </div>
        </div>
        {bgLines.map((line, i) => (
          <div
            key={`${line.t}-${i}`}
            style={{
              fontFamily: "monospace",
              fontSize: 15,
              lineHeight: "22px",
              color: line.color || "transparent",
              whiteSpace: "pre",
            }}
          >
            {line.text}
          </div>
        ))}
      </div>

      {/* ── Typewriter intro ── */}
      <TypewriterIntro frame={frame} />

      {/* ── Number slides ── */}
      {SLIDES.map((slide, i) => (
        <NumberSlideComp key={i} slide={slide} frame={frame} />
      ))}

      {/* ── Final grid ── */}
      <FinalGrid frame={frame} showAt={780} />

      {/* ── Subtitles ── */}
      <Sub frame={frame} showAt={65}  hideAt={235}  text="3,145 resolved Polymarket predictions" />
      <Sub frame={frame} showAt={245} hideAt={415}  text="Beating market consensus — no tuning, no prompts" />
      <Sub frame={frame} showAt={425} hideAt={535}  text="OpenML · reproducible benchmark" />
      <Sub frame={frame} showAt={545} hideAt={655}  text="82.8% win rate · head-to-head · 60s budget" />
      <Sub frame={frame} showAt={665} hideAt={775}  text="Pure evolution — zero LLM calls at inference time" />
    </div>
  );
};
