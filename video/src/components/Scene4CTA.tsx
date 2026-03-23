import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";

export const Scene4CTA: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 元素淡出动画
  const fadeOutOpacity = interpolate(frame, [0, 30], [1, 0], {
    extrapolateRight: "clamp",
  });

  // ORCETRA logo 缩放出现
  const logoScale = spring({
    frame: frame - 30,
    fps,
    config: {
      damping: 50,
      stiffness: 100,
      mass: 0.8,
    },
  });

  const logoOpacity = interpolate(frame, [30, 60], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 副标题出现
  const subtitleOpacity = interpolate(frame, [90, 120], [0, 1], {
    extrapolateRight: "clamp",
  });

  const subtitleY = interpolate(frame, [90, 120], [50, 0], {
    extrapolateRight: "clamp",
  });

  // 网址和GitHub出现
  const linksOpacity = interpolate(frame, [150, 180], [0, 1], {
    extrapolateRight: "clamp",
  });

  const linksY = interpolate(frame, [150, 180], [30, 0], {
    extrapolateRight: "clamp",
  });

  // 最终静止状态的光效
  const glowIntensity = interpolate(frame, [180, 240], [0, 1], {
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
        position: "relative",
      }}
    >
      {/* 背景光效 */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          background: `radial-gradient(circle at center, #00d4ff${Math.floor(glowIntensity * 15).toString(16).padStart(2, '0')} 0%, transparent 50%)`,
          pointerEvents: "none",
        }}
      />

      {/* 淡出遮罩 - 用于过渡效果 */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          backgroundColor: "#0a0f1a",
          opacity: 1 - fadeOutOpacity,
          pointerEvents: "none",
        }}
      />

      {/* ORCETRA 主logo */}
      <div
        style={{
          fontSize: 128,
          fontWeight: "900",
          color: "#ffffff",
          textAlign: "center",
          opacity: logoOpacity,
          transform: `scale(${logoScale})`,
          textShadow: `
            0 0 30px #00d4ff${Math.floor(glowIntensity * 80).toString(16).padStart(2, '0')},
            0 0 60px #00ff88${Math.floor(glowIntensity * 40).toString(16).padStart(2, '0')},
            0 8px 30px rgba(0,0,0,0.8)
          `,
          letterSpacing: "0.1em",
          fontFamily: "Inter, Arial, sans-serif",
        }}
      >
        ORCETRA
      </div>

      {/* 副标题 */}
      <div
        style={{
          fontSize: 36,
          color: "#ffffff80",
          textAlign: "center",
          opacity: subtitleOpacity,
          transform: `translateY(${subtitleY}px)`,
          marginTop: 40,
          letterSpacing: "0.05em",
          fontWeight: "300",
        }}
      >
        AI Prediction Intelligence Engine
      </div>

      {/* 网址和GitHub链接 */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 25,
          marginTop: 80,
          opacity: linksOpacity,
          transform: `translateY(${linksY}px)`,
        }}
      >
        {/* 网站链接 */}
        <div
          style={{
            fontSize: 32,
            color: "#00d4ff",
            fontWeight: "500",
            textShadow: "0 0 15px #00d4ff60",
            letterSpacing: "0.02em",
          }}
        >
          orcetra.ai
        </div>

        {/* GitHub链接 */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 15,
            fontSize: 28,
            color: "#ffffff80",
          }}
        >
          {/* GitHub图标 */}
          <svg
            width="32"
            height="32"
            viewBox="0 0 24 24"
            fill="currentColor"
            style={{
              filter: `drop-shadow(0 0 10px #ffffff40)`,
            }}
          >
            <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
          </svg>
          <span>GitHub</span>
        </div>
      </div>

      {/* 装饰性粒子效果 */}
      {[...Array(8)].map((_, i) => {
        const angle = (i * 45) * (Math.PI / 180);
        const radius = 300 + Math.sin(frame * 0.02 + i) * 50;
        const x = Math.cos(angle + frame * 0.01) * radius;
        const y = Math.sin(angle + frame * 0.01) * radius;
        
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: "50%",
              top: "50%",
              width: 4,
              height: 4,
              backgroundColor: i % 2 === 0 ? "#00d4ff" : "#00ff88",
              borderRadius: "50%",
              transform: `translate(${x}px, ${y}px)`,
              opacity: glowIntensity * 0.6,
              boxShadow: `0 0 10px ${i % 2 === 0 ? "#00d4ff" : "#00ff88"}80`,
            }}
          />
        );
      })}
    </div>
  );
};