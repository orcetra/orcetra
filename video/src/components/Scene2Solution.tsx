import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";

export const Scene2Solution: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 标题动画
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Orcetra修正线出现
  const correctionLineOpacity = interpolate(frame, [30, 60], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 三个步骤依次出现
  const step1Opacity = interpolate(frame, [60, 90], [0, 1], {
    extrapolateRight: "clamp",
  });
  const step1X = interpolate(frame, [60, 90], [-200, 0], {
    extrapolateRight: "clamp",
  });

  const step2Opacity = interpolate(frame, [100, 130], [0, 1], {
    extrapolateRight: "clamp",
  });
  const step2X = interpolate(frame, [100, 130], [-200, 0], {
    extrapolateRight: "clamp",
  });

  const step3Opacity = interpolate(frame, [140, 170], [0, 1], {
    extrapolateRight: "clamp",
  });
  const step3X = interpolate(frame, [140, 170], [-200, 0], {
    extrapolateRight: "clamp",
  });

  // 修正箭头动画
  const arrowOpacity = interpolate(frame, [180, 210], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 数字变化动画 50% → 46.4%
  const numberTransition = interpolate(frame, [200, 240], [50, 46.4], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #0a0f1a 0%, #1a2540 100%)",
      }}
    >
      {/* 标题 */}
      <div
        style={{
          position: "absolute",
          top: 100,
          fontSize: 64,
          fontWeight: "bold",
          color: "#00ff88",
          textAlign: "center",
          opacity: titleOpacity,
          textShadow: "0 0 30px #00ff8860, 0 4px 20px rgba(0,0,0,0.8)",
        }}
      >
        Orcetra Corrects the Bias
      </div>

      {/* 校准曲线背景 */}
      <div
        style={{
          position: "absolute",
          top: 200,
          width: 800,
          height: 200,
        }}
      >
        <svg width="800" height="200" viewBox="0 0 800 200">
          <defs>
            <linearGradient id="biasGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: "#ff4757", stopOpacity: 0.3 }} />
              <stop offset="100%" style={{ stopColor: "#ff4757", stopOpacity: 0.8 }} />
            </linearGradient>
            <linearGradient id="correctionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: "#00ff88", stopOpacity: 0.8 }} />
              <stop offset="100%" style={{ stopColor: "#00d4ff", stopOpacity: 0.8 }} />
            </linearGradient>
          </defs>
          
          {/* 理想线 */}
          <line
            x1="50"
            y1="150"
            x2="750"
            y2="50"
            stroke="#ffffff30"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
          
          {/* 原始偏差曲线 */}
          <path
            d="M 50 150 Q 250 110 400 100 Q 550 95 750 90"
            stroke="url(#biasGradient)"
            strokeWidth="3"
            fill="none"
          />
          
          {/* Orcetra修正线 */}
          <path
            d="M 50 150 Q 250 130 400 120 Q 550 110 750 100"
            stroke="url(#correctionGradient)"
            strokeWidth="4"
            fill="none"
            opacity={correctionLineOpacity}
            filter="drop-shadow(0 0 15px #00ff8840)"
          />
        </svg>
      </div>

      {/* 三个步骤 */}
      <div
        style={{
          position: "absolute",
          top: 450,
          display: "flex",
          gap: 120,
          alignItems: "center",
        }}
      >
        {/* 步骤1: Scan */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: step1Opacity,
            transform: `translateX(${step1X}px)`,
          }}
        >
          <div
            style={{
              fontSize: 80,
              marginBottom: 20,
              filter: "drop-shadow(0 0 20px #00d4ff60)",
            }}
          >
            📡
          </div>
          <div
            style={{
              fontSize: 32,
              fontWeight: "bold",
              color: "#00d4ff",
              marginBottom: 10,
            }}
          >
            Scan
          </div>
          <div
            style={{
              fontSize: 20,
              color: "#ffffff80",
              textAlign: "center",
              maxWidth: 180,
            }}
          >
            Monitor 17,000+ markets
          </div>
        </div>

        {/* 步骤2: Calibrate */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: step2Opacity,
            transform: `translateX(${step2X}px)`,
          }}
        >
          <div
            style={{
              fontSize: 80,
              marginBottom: 20,
              filter: "drop-shadow(0 0 20px #00ff8860)",
            }}
          >
            ⚖️
          </div>
          <div
            style={{
              fontSize: 32,
              fontWeight: "bold",
              color: "#00ff88",
              marginBottom: 10,
            }}
          >
            Calibrate
          </div>
          <div
            style={{
              fontSize: 20,
              color: "#ffffff80",
              textAlign: "center",
              maxWidth: 180,
            }}
          >
            Correct long-shot bias
          </div>
        </div>

        {/* 步骤3: Verify */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: step3Opacity,
            transform: `translateX(${step3X}px)`,
          }}
        >
          <div
            style={{
              fontSize: 80,
              marginBottom: 20,
              filter: "drop-shadow(0 0 20px #E8734A60)",
            }}
          >
            ✅
          </div>
          <div
            style={{
              fontSize: 32,
              fontWeight: "bold",
              color: "#E8734A",
              marginBottom: 10,
            }}
          >
            Verify
          </div>
          <div
            style={{
              fontSize: 20,
              color: "#ffffff80",
              textAlign: "center",
              maxWidth: 180,
            }}
          >
            Track every prediction
          </div>
        </div>
      </div>

      {/* 修正箭头和数字变化 */}
      <div
        style={{
          position: "absolute",
          bottom: 150,
          display: "flex",
          alignItems: "center",
          gap: 60,
          opacity: arrowOpacity,
        }}
      >
        <div
          style={{
            fontSize: 48,
            color: "#ffffff80",
            fontWeight: "bold",
          }}
        >
          Market Price: 50%
        </div>

        <div
          style={{
            fontSize: 60,
            color: "#00ff88",
          }}
        >
          →
        </div>

        <div
          style={{
            fontSize: 48,
            color: "#00ff88",
            fontWeight: "bold",
            textShadow: "0 0 20px #00ff8860",
          }}
        >
          Orcetra: {numberTransition.toFixed(1)}%
        </div>
      </div>
    </div>
  );
};