import React from "react";
import { Composition } from "remotion";
import { SplitScreenDemo } from "./SplitScreenDemo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SplitScreenDemo"
        component={SplitScreenDemo}
        durationInFrames={900} // 30 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
