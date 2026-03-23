#!/bin/bash

# Orcetra 产品概念动画完整渲染脚本

echo "🎬 开始渲染 Orcetra 完整版本 (30秒, 900帧)..."

# 创建输出目录
mkdir -p out

# 渲染完整视频 (30秒 = 900帧)
npx remotion render OrcetraVideo out/orcetra-demo.mp4 \
  --frames=0-899 \
  --concurrency=4 \
  --quiet

echo "✅ 渲染完成！"
echo "📁 输出文件: out/orcetra-demo.mp4"
echo "🎥 30秒 @ 1920x1080 @ 30fps"
echo ""
echo "📊 文件信息:"
ls -lh out/orcetra-demo.mp4

echo ""
echo "🚀 预览命令: npx remotion preview"
echo "📱 本地预览: http://localhost:3000"