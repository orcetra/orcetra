import React from "react";
import { Sequence } from "remotion";
// 暂时移除 Google Fonts
import { Scene1Problem } from "./components/Scene1Problem";
import { Scene2Solution } from "./components/Scene2Solution";
import { Scene3Results } from "./components/Scene3Results";
import { Scene4CTA } from "./components/Scene4CTA";

const fontFamily = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

export const OrcetraVideo: React.FC = () => {
  return (
    <div
      style={{
        flex: 1,
        backgroundColor: "#0a0f1a", // 深蓝黑背景
        fontFamily,
      }}
    >
      {/* Scene 1: Problem (0-6s, frame 0-180) */}
      <Sequence from={0} durationInFrames={180}>
        <Scene1Problem />
      </Sequence>

      {/* Scene 2: Solution (6-14s, frame 180-420) */}
      <Sequence from={180} durationInFrames={240}>
        <Scene2Solution />
      </Sequence>

      {/* Scene 3: Results (14-22s, frame 420-660) */}
      <Sequence from={420} durationInFrames={240}>
        <Scene3Results />
      </Sequence>

      {/* Scene 4: CTA (22-30s, frame 660-900) */}
      <Sequence from={660} durationInFrames={240}>
        <Scene4CTA />
      </Sequence>
    </div>
  );
};