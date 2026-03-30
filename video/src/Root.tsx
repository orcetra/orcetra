import React from "react";
import { Composition } from "remotion";
import { SplitScreenDemo } from "./SplitScreenDemo";
import { Video1BeatMarket } from "./Video1BeatMarket";
import { Video2AutoML } from "./Video2AutoML";
import { Video3Numbers } from "./Video3Numbers";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SplitScreenDemo"
        component={SplitScreenDemo}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="BeatTheMarket"
        component={Video1BeatMarket}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SixtySecondAutoML"
        component={Video2AutoML}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="NumbersSpeak"
        component={Video3Numbers}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
