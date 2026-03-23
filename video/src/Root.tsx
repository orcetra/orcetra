import React from "react";
import { Composition } from "remotion";
import { OrcetraVideo } from "./OrcetraVideo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="OrcetraVideo"
        component={OrcetraVideo}
        durationInFrames={900} // 30 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};