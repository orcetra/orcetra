import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

interface NumberCounterProps {
  targetNumber: number;
  unit: string;
  label: string;
  color?: string;
  description?: string;
  delay?: number;
}

export const NumberCounter: React.FC<NumberCounterProps> = ({
  targetNumber,
  unit,
  label,
  color = "#00d4ff",
  description,
  delay = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 数字弹性动画，带延迟
  const progress = spring({
    frame: frame - delay,
    fps,
    config: {
      damping: 50,
      stiffness: 100,
      mass: 0.5,
    },
  });

  const currentNumber = Math.floor(interpolate(progress, [0, 1], [0, targetNumber]));

  // 透明度淡入
  const opacity = interpolate(frame, [delay, delay + 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  // 缩放动画
  const scale = interpolate(progress, [0, 1], [0.8, 1]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        transform: `scale(${scale})`,
      }}
    >
      {/* 主数字 */}
      <div
        style={{
          fontSize: 120,
          fontWeight: "bold",
          fontFamily: "Inter, Arial, sans-serif",
          color: color,
          textShadow: `0 0 30px ${color}40, 0 0 60px ${color}20`,
          letterSpacing: "-0.02em",
          display: "flex",
          alignItems: "baseline",
        }}
      >
        {currentNumber.toLocaleString()}
        <span style={{ fontSize: 60, marginLeft: 10 }}>{unit}</span>
      </div>

      {/* 标签 */}
      <div
        style={{
          fontSize: 24,
          color: "#ffffff80",
          marginTop: 15,
          fontFamily: "Inter, Arial, sans-serif",
          letterSpacing: "0.1em",
          textAlign: "center",
        }}
      >
        {label}
      </div>

      {/* 描述文字 */}
      {description && (
        <div
          style={{
            fontSize: 18,
            color: "#ffffffa0",
            marginTop: 20,
            letterSpacing: "0.05em",
            opacity: interpolate(frame, [delay + fps * 1, delay + fps * 2], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
            textShadow: "0 2px 10px rgba(0,0,0,0.8)",
            textAlign: "center",
          }}
        >
          {description}
        </div>
      )}
    </div>
  );
};