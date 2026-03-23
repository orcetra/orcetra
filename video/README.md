# Orcetra 产品概念动画

30 秒 Remotion 产品宣传视频，展示 Orcetra 如何修正预测市场中的系统性偏差。

## 🎬 视频结构 (30秒 @ 30fps = 900帧)

### Scene 1: 问题 (0-6s, frames 0-180)
**"The Crowd Gets It Wrong"**
- 深蓝渐变背景
- 仪表盘显示"Market Consensus: 50%"
- 指针动画 + 校准曲线显示偏差
- "But crowds have systematic biases"

### Scene 2: 解决方案 (6-14s, frames 180-420)  
**"Orcetra Corrects the Bias"**
- Orcetra 修正线出现在校准曲线上
- 三步流程动画：📡 Scan → 🧠 Calibrate → ✅ Verify
- 数字修正: 50% → 46.4%

### Scene 3: 结果 (14-22s, frames 420-660)
**"Verified Performance"**
- 大数字动画：70.2% Beat Rate, 1,974 预测
- 4个分类条形图：
  - Golf: 89.7%
  - Politics: 76.2%  
  - Sports: 69.1%
  - Economy: 63.4%

### Scene 4: CTA (22-30s, frames 660-900)
**"Orcetra"**
- Logo 缩放出现
- 副标题："AI Prediction Intelligence Engine"
- 联系信息：orcetra.ai + GitHub

## 🎨 设计风格

- **背景色**: #0a0f1a (深蓝黑)
- **强调色**: 
  - #00d4ff (青蓝)
  - #00ff88 (绿) 
  - #E8734A (珊瑚橙)
- **字体**: Inter (Google Fonts)
- **尺寸**: 1920x1080 @ 30fps

## 🚀 使用方法

### 预览
```bash
npm run dev
# 或
npx remotion preview
```

### 渲染测试版 (3秒)
```bash
npx remotion render OrcetraVideo out/test.mp4 --frames=0-89
```

### 渲染完整版 (30秒)
```bash
./render-full.sh
# 或
npx remotion render OrcetraVideo out/orcetra-demo.mp4
```

## 📦 依赖

- `remotion` - 核心渲染引擎
- `@remotion/google-fonts` - Inter 字体
- `react` & `react-dom` - UI 框架

## 🔧 组件结构

```
src/
├── OrcetraVideo.tsx         # 主视频组件
├── Root.tsx                 # Remotion 根组件
├── index.tsx               # 入口点
└── components/
    ├── Scene1Problem.tsx    # 场景1：问题描述
    ├── Scene2Solution.tsx   # 场景2：解决方案
    ├── Scene3Results.tsx    # 场景3：验证结果
    ├── Scene4CTA.tsx       # 场景4：行动号召
    └── NumberCounter.tsx   # 数字动画组件 (参考岩瞳项目)
```

## 📊 真实数据

所有展示的数字都是 Orcetra 的真实表现数据：
- Beat Rate: 70.2%
- Verified Predictions: 1,974
- Golf: 89.7%, Politics: 76.2%, Sports: 69.1%, Economy: 63.4%

## 🎯 参考

基于 `/home/guilinzhang/allProjects/luqiao/demo/video/` 项目的 NumberCounter 组件，复用动画模式和组件思路。