import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { NumberCounter } from "./NumberCounter";
import { STATS } from "../data";

export const Scene3Results: React.FC = () => {
  const frame = useCurrentFrame();

  // 标题淡入动画
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 条形图动画开始时间
  const chartStartFrame = 120;
  
  // 各分类的进度条动画
  const golfProgress = interpolate(frame, [chartStartFrame, chartStartFrame + 60], [0, 0.897], {
    extrapolateRight: "clamp",
  });
  
  const politicsProgress = interpolate(frame, [chartStartFrame + 15, chartStartFrame + 75], [0, 0.762], {
    extrapolateRight: "clamp",
  });
  
  const sportsProgress = interpolate(frame, [chartStartFrame + 30, chartStartFrame + 90], [0, 0.691], {
    extrapolateRight: "clamp",
  });
  
  const economyProgress = interpolate(frame, [chartStartFrame + 45, chartStartFrame + 105], [0, 0.634], {
    extrapolateRight: "clamp",
  });

  // 图表整体出现
  const chartOpacity = interpolate(frame, [chartStartFrame - 30, chartStartFrame], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 底部说明文字
  const footerOpacity = interpolate(frame, [200, 230], [0, 1], {
    extrapolateRight: "clamp",
  });

  const commoditiesProgress = interpolate(frame, [chartStartFrame + 60, chartStartFrame + 120], [0, 0.55], {
    extrapolateRight: "clamp",
  });

  const categories = [
    { name: "Golf", progress: golfProgress, color: "#00ff88" },
    { name: "Politics", progress: politicsProgress, color: "#00d4ff" },
    { name: "Sports", progress: sportsProgress, color: "#E8734A" },
    { name: "Economy", progress: economyProgress, color: "#ffaa00" },
    { name: "Commodities", progress: commoditiesProgress, color: "#bb88ff" },
  ];

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
          top: 80,
          fontSize: 64,
          fontWeight: "bold",
          color: "#ffffff",
          textAlign: "center",
          opacity: titleOpacity,
          textShadow: "0 4px 20px rgba(0,0,0,0.8)",
        }}
      >
        Verified Performance
      </div>

      {/* 主要指标 */}
      <div
        style={{
          position: "absolute",
          top: 200,
          display: "flex",
          gap: 120,
          alignItems: "center",
        }}
      >
        {/* Beat Rate */}
        <NumberCounter
          targetNumber={STATS.beatRate}
          unit="%+"
          label="Beat Rate"
          color="#00ff88"
          delay={30}
        />

        {/* Verified Predictions */}
        <NumberCounter
          targetNumber={STATS.verifiedPredictions}
          unit="+"
          label="Verified Predictions"
          color="#00d4ff"
          delay={60}
        />
      </div>

      {/* 分类条形图 */}
      <div
        style={{
          position: "absolute",
          top: 450,
          width: 800,
          opacity: chartOpacity,
        }}
      >
        <div
          style={{
            fontSize: 28,
            color: "#ffffff",
            fontWeight: "bold",
            marginBottom: 30,
            textAlign: "center",
          }}
        >
          Performance by Category
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 20,
          }}
        >
          {categories.map((category, index) => (
            <div
              key={category.name}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 20,
              }}
            >
              {/* 分类名称 */}
              <div
                style={{
                  width: 120,
                  fontSize: 24,
                  color: "#ffffff",
                  fontWeight: "500",
                  textAlign: "right",
                }}
              >
                {category.name}:
              </div>

              {/* 进度条容器 */}
              <div
                style={{
                  width: 400,
                  height: 32,
                  backgroundColor: "#ffffff20",
                  borderRadius: 16,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                {/* 进度条填充 */}
                <div
                  style={{
                    width: `${category.progress * 100}%`,
                    height: "100%",
                    backgroundColor: category.color,
                    borderRadius: 16,
                    transition: "width 0.3s ease-out",
                    boxShadow: `0 0 20px ${category.color}60`,
                  }}
                />
              </div>

              {/* 进度条末端光点 */}
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  backgroundColor: category.color,
                  boxShadow: `0 0 10px ${category.color}80`,
                  opacity: category.progress > 0.01 ? 1 : 0,
                }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* 底部说明 */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          fontSize: 24,
          color: "#ffffff80",
          textAlign: "center",
          opacity: footerOpacity,
        }}
      >
        Every prediction verifiable on our live dashboard
      </div>
    </div>
  );
};