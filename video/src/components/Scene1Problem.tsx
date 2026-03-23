import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";

export const Scene1Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 标题淡入动画
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 仪表盘出现动画
  const gaugeScale = spring({
    frame: frame - 30,
    fps,
    config: {
      damping: 50,
      stiffness: 100,
      mass: 0.5,
    },
  });

  // 指针旋转动画 (从0到50%)
  const pointerRotation = interpolate(frame, [60, 120], [0, 0.5], {
    extrapolateRight: "clamp",
  });

  // 副标题出现
  const subtitleOpacity = interpolate(frame, [90, 120], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 偏差警告出现
  const biasWarningOpacity = interpolate(frame, [120, 150], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 校准曲线出现
  const curveOpacity = interpolate(frame, [150, 180], [0, 1], {
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
          top: 150,
          fontSize: 64,
          fontWeight: "bold",
          color: "#ffffff",
          textAlign: "center",
          opacity: titleOpacity,
          textShadow: "0 4px 20px rgba(0,0,0,0.8)",
        }}
      >
        The Crowd Gets It Wrong
      </div>

      {/* 仪表盘容器 */}
      <div
        style={{
          position: "relative",
          width: 400,
          height: 400,
          transform: `scale(${gaugeScale})`,
          opacity: gaugeScale,
        }}
      >
        {/* 仪表盘背景圆圈 */}
        <div
          style={{
            position: "absolute",
            width: 400,
            height: 400,
            borderRadius: "50%",
            border: "8px solid #ffffff20",
            background: "radial-gradient(circle, #1a2540 0%, #0a0f1a 100%)",
          }}
        />

        {/* 刻度线 */}
        {[...Array(11)].map((_, i) => {
          const angle = -90 + (i * 18); // 从-90度开始，每18度一个刻度
          return (
            <div
              key={i}
              style={{
                position: "absolute",
                width: 4,
                height: 30,
                backgroundColor: "#ffffff60",
                top: 20,
                left: 198,
                transformOrigin: "2px 180px",
                transform: `rotate(${angle}deg)`,
              }}
            />
          );
        })}

        {/* 百分比标签 */}
        {[0, 25, 50, 75, 100].map((value, i) => {
          const angle = -90 + (i * 45);
          const radian = (angle * Math.PI) / 180;
          const x = 200 + Math.cos(radian) * 150;
          const y = 200 + Math.sin(radian) * 150;
          
          return (
            <div
              key={value}
              style={{
                position: "absolute",
                left: x - 15,
                top: y - 10,
                color: "#ffffff80",
                fontSize: 20,
                fontWeight: "bold",
                textAlign: "center",
              }}
            >
              {value}%
            </div>
          );
        })}

        {/* 指针 */}
        <div
          style={{
            position: "absolute",
            width: 6,
            height: 160,
            backgroundColor: "#00d4ff",
            top: 40,
            left: 197,
            transformOrigin: "3px 160px",
            transform: `rotate(${-90 + pointerRotation * 180}deg)`,
            borderRadius: "3px",
            boxShadow: "0 0 15px #00d4ff60",
          }}
        />

        {/* 中心点 */}
        <div
          style={{
            position: "absolute",
            width: 20,
            height: 20,
            borderRadius: "50%",
            backgroundColor: "#00d4ff",
            top: 190,
            left: 190,
            boxShadow: "0 0 20px #00d4ff80",
          }}
        />

        {/* 数值显示 */}
        <div
          style={{
            position: "absolute",
            bottom: 80,
            left: "50%",
            transform: "translateX(-50%)",
            color: "#ffffff",
            fontSize: 36,
            fontWeight: "bold",
            textShadow: "0 0 15px #00d4ff40",
          }}
        >
          Market Consensus: {Math.round(pointerRotation * 100)}%
        </div>
      </div>

      {/* 副标题 */}
      <div
        style={{
          position: "absolute",
          bottom: 300,
          fontSize: 28,
          color: "#ffffff80",
          textAlign: "center",
          opacity: subtitleOpacity,
        }}
      >
        Prediction markets aggregate crowd wisdom...
      </div>

      {/* 偏差警告 */}
      <div
        style={{
          position: "absolute",
          bottom: 220,
          fontSize: 32,
          color: "#ff4757",
          fontWeight: "bold",
          textAlign: "center",
          opacity: biasWarningOpacity,
          textShadow: "0 0 15px #ff475760",
        }}
      >
        But crowds have systematic biases
      </div>

      {/* 校准曲线 */}
      <div
        style={{
          position: "absolute",
          bottom: 50,
          width: 600,
          height: 150,
          opacity: curveOpacity,
        }}
      >
        <svg width="600" height="150" viewBox="0 0 600 150">
          <defs>
            <linearGradient id="curveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: "#00d4ff", stopOpacity: 0.8 }} />
              <stop offset="100%" style={{ stopColor: "#ff4757", stopOpacity: 0.8 }} />
            </linearGradient>
          </defs>
          
          {/* 理想线 (对角线) */}
          <line
            x1="50"
            y1="100"
            x2="550"
            y2="50"
            stroke="#ffffff40"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
          
          {/* 实际偏差曲线 (系统性高估) */}
          <path
            d="M 50 100 Q 200 70 300 60 Q 400 55 550 50"
            stroke="url(#curveGradient)"
            strokeWidth="4"
            fill="none"
            filter="drop-shadow(0 0 10px #ff475740)"
          />
          
          {/* 轴标签 */}
          <text x="300" y="140" textAnchor="middle" fill="#ffffff60" fontSize="16">
            Predicted vs Actual Performance
          </text>
        </svg>
      </div>
    </div>
  );
};