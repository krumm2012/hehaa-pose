# 🚀 网球分析系统 - 快速启动指南

## 📋 启动命令详解

### 标准启动命令
```bash
source venv/bin/activate && python3 main.py \
  --config configs/comprehensive_tennis_config.yaml \
  --input "视频路径" \
  --output_dir "输出目录" \
  --original_name "文件名"
```

### 命令参数说明

| 参数 | 说明 | 示例 | 必需 |
|------|------|------|------|
| `--config` | 配置文件路径 | `configs/comprehensive_tennis_config.yaml` | ✅ |
| `--input` | 输入视频路径 | `data/tennis_match.mp4` | ✅ |
| `--output_dir` | 输出目录 | `output/match_001/` | ❌ |
| `--original_name` | 原始文件名 | `match_001.mp4` | ❌ |

---

## 🎯 使用场景和配置选择

### 1. 首次使用 - 综合分析
```bash
source venv/bin/activate && python3 main.py \
  --config configs/comprehensive_tennis_config.yaml \
  --input "data/input_video.mp4"
```

**特点**:
- ✅ 完整功能: 姿态检测、球追踪、球拍检测、挥拍分析
- ✅ 精彩瞬间捕捉
- ✅ ROI区域优化
- ⚠️ 处理速度: 约3.99 FPS

---

### 2. 性能优先 - 快速处理
```bash
source venv/bin/activate && python3 main.py \
  --config configs/performance_optimized_config.yaml \
  --input "data/input_video.mp4"
```

**特点**:
- ✅ 处理速度: 目标5-8 FPS
- ✅ 跳帧处理
- ✅ 简化UI
- ✅ 无界面运行
- ⚠️ 功能简化

**配置差异**:
```yaml
performance_optimization:
  headless: true          # 无界面运行
  minimal_ui: true        # 简化UI
  skip_frames: 2          # 每2帧处理1帧

video_processing:
  resize_factor: 0.8      # 视频缩放0.8倍
```

---

### 3. 球检测优化 - 解决误识别
```bash
source venv/bin/activate && python3 main.py \
  --config configs/optimized_ball_config.yaml \
  --input "data/input_video.mp4"
```

**特点**:
- ✅ 优化静止球过滤
- ✅ 增强运动球检测
- ✅ 减少背景干扰

**关键参数**:
```yaml
static_ball_movement_threshold_px: 6
static_ball_frames_threshold: 10
noise_filter_quality_threshold: 0.45
```

---

### 4. 噪点过滤 - 清除小尺寸干扰
```bash
source venv/bin/activate && python3 main.py \
  --config configs/noise_filtered_ball_config.yaml \
  --input "data/input_video.mp4"
```

**特点**:
- ✅ 严格尺寸过滤
- ✅ 圆度检查
- ✅ 边缘距离过滤

---

### 5. ROI区域检测 - 精确分析
```bash
source venv/bin/activate && python3 main.py \
  --config configs/roi_enabled_config.yaml \
  --input "data/input_video.mp4"
```

**特点**:
- ✅ 4点描线兴趣区域
- ✅ 计算量减少50%+
- ✅ 精度提升
- ✅ 自动坐标转换

**首次使用需要交互式选择ROI**:
```yaml
roi_settings:
  enabled: true
  interactive_selection: true  # 首次设为true
  auto_load_config: false      # 首次设为false
```

---

## 🔧 配置文件对比

### 配置文件列表

| 配置文件 | 用途 | 速度 | 功能 | 推荐场景 |
|---------|------|------|------|---------|
| `comprehensive_tennis_config.yaml` | 综合分析 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 完整分析 |
| `performance_optimized_config.yaml` | 性能优先 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 快速处理 |
| `optimized_ball_config.yaml` | 球检测优化 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 球检测问题 |
| `noise_filtered_ball_config.yaml` | 噪点过滤 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 小噪点干扰 |
| `roi_enabled_config.yaml` | ROI区域 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 区域分析 |
| `default_config.yaml` | 默认配置 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 基础使用 |

---

## 📁 输出文件说明

### 1. 主输出视频
```
data/output_video.mp4
```
- 包含所有检测结果的可视化
- 编码格式: avc1
- FPS: 与输入视频相同

### 2. 精彩瞬间文件夹
```
data/highlights/
├── highlight_000123_hit.mp4           # 短视频 (前5帧+中心帧+后15帧)
├── highlight_000123_hit.jpg           # 中心帧图片
├── highlight_000123_hit.analysis.json # 分析数据
├── highlight_000123_hit_prep.jpg      # 准备阶段截图
├── highlight_000123_hit_turn.jpg      # 转身阶段截图
├── highlight_000123_hit_drop.jpg      # 降拍阶段截图
├── highlight_000123_hit_swing.jpg     # 挥拍阶段截图
└── highlight_000123_hit_foot.jpg      # 步伐阶段截图
```

### 3. 调试帧 (如果启用)
```
debug_frames/
├── frame_000025_debug.jpg
├── frame_000050_debug.jpg
└── ...
```

---

## 🎬 完整使用流程

### 步骤1: 环境激活
```bash
cd /Users/krum5539/Documents/tennis_analyzer
source venv/bin/activate
```

### 步骤2: 准备输入视频
```bash
# 将视频放到data目录
cp ~/Downloads/tennis_match.mp4 data/input_video.mp4
```

### 步骤3: 选择配置并运行
```bash
# 综合分析模式
python3 main.py \
  --config configs/comprehensive_tennis_config.yaml \
  --input "data/input_video.mp4" \
  --output_dir "output/match_001/" \
  --original_name "tennis_match.mp4"
```

### 步骤4: 查看输出
```bash
# 查看主输出视频
open data/output_video.mp4

# 查看精彩瞬间
open data/highlights/

# 查看分析JSON
cat data/highlights/highlight_000123_hit.analysis.json | jq
```

---

## 🔍 调试和测试

### 1. 球检测测试
```bash
python test_enhanced_ball_detection.py
```

### 2. 噪点过滤测试
```bash
python test_noise_filtering.py
```

### 3. ROI功能演示
```bash
python demo_roi_detection.py
```

### 4. 球颜色标识演示
```bash
python demo_ball_color_identification.py
```

### 5. 性能测试
```bash
python test_performance_optimization.py
```

---

## ⚙️ 常用参数调整

### 提高球检测准确性
```yaml
# 调整HSV颜色范围
hsv_lower_hue: 20
hsv_upper_hue: 70
hsv_lower_sat: 50
hsv_upper_sat: 255

# 调整Hough圆检测
hough_param2: 6  # 降低阈值增加召回
```

### 减少静止球误识别
```yaml
static_ball_movement_threshold_px: 8  # 提高阈值
static_ball_frames_threshold: 12      # 增加判定帧数
```

### 调整精彩瞬间检测
```yaml
highlights:
  hit_distance_factor: 1.5      # 击球距离因子
  distance_scale: 5             # 距离缩放
  cooldown_frames: 12           # 冷却帧数
  min_inside_frames: 1          # 最少在阈值内帧数
```

### 提高处理速度
```yaml
performance_optimization:
  headless: true                # 无界面运行
  minimal_ui: true              # 简化UI

video_processing:
  skip_frames: 2                # 跳帧处理
  resize_factor: 0.8            # 视频缩放
```

---

## 🛠️ 故障排除

### 问题1: 无法打开视频
**症状**: `错误: 无法打开视频`

**解决方案**:
```bash
# 检查视频文件是否存在
ls -lh data/input_video.mp4

# 检查视频编码格式
ffprobe data/input_video.mp4

# 转换视频格式
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

---

### 问题2: 模型文件缺失
**症状**: `FileNotFoundError: models/yolov8n-pose.pt`

**解决方案**:
```bash
# 下载YOLOv8模型
mkdir -p models
cd models

# 下载姿态检测模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

# 下载目标检测模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

cd ..
```

---

### 问题3: 内存不足
**症状**: `MemoryError` 或系统卡顿

**解决方案**:
```yaml
# 使用性能优化配置
performance_optimization:
  memory_optimization: true
  batch_processing: false

video_processing:
  resize_factor: 0.6  # 降低分辨率
  max_frames: 500     # 限制处理帧数
```

---

### 问题4: ROI选择失败
**症状**: ROI交互窗口无响应

**解决方案**:
```bash
# 使用演示脚本重新选择
python demo_roi_detection.py

# 或手动编辑ROI配置
vim configs/roi_config.yaml
```

---

### 问题5: 精彩瞬间检测过多/过少
**过多解决方案**:
```yaml
highlights:
  cooldown_frames: 20           # 增加冷却时间
  min_inside_frames: 3          # 提高最少停留帧数
  min_speed_px_per_frame: 10    # 提高最小球速
```

**过少解决方案**:
```yaml
highlights:
  distance_scale: 8             # 放大距离阈值
  min_inside_frames: 1          # 降低最少停留帧数
  min_exit_increase_px: 0       # 取消离开增加限制
```

---

## 📊 性能监控

### 查看处理进度
```bash
# 运行时会显示:
处理帧 250/1000 (25.0%) - 批处理时间: 0.25秒, 估计剩余时间: 18.8秒
```

### 查看最终统计
```bash
# 处理完成后会显示:
处理完成! 共处理 1000 帧
总处理时间: 250.45 秒
平均处理速度: 3.99 FPS
```

---

## 🎯 最佳实践

### 1. 首次使用流程
```bash
# 1. 使用演示脚本熟悉系统
python demo_ball_color_identification.py

# 2. 使用ROI演示选择兴趣区域
python demo_roi_detection.py

# 3. 使用综合配置进行完整分析
python3 main.py --config configs/comprehensive_tennis_config.yaml \
  --input "data/input_video.mp4"

# 4. 查看输出结果
open data/output_video.mp4
open data/highlights/
```

### 2. 批量处理流程
```bash
# 使用批量处理脚本
python batch_process_videos.py \
  --input_dir "data/videos/" \
  --output_dir "output/" \
  --config "configs/performance_optimized_config.yaml"
```

### 3. 参数调优流程
```bash
# 1. 使用默认配置处理
python3 main.py --config configs/default_config.yaml --input "test.mp4"

# 2. 查看调试帧分析问题
ls debug_frames/

# 3. 调整参数
vim configs/custom_config.yaml

# 4. 重新处理验证
python3 main.py --config configs/custom_config.yaml --input "test.mp4"
```

---

## 📝 配置模板

### 自定义配置模板
```yaml
# custom_config.yaml
# 基于comprehensive_tennis_config.yaml修改

# 视频路径
video_input_path: "data/my_video.mp4"
video_output_path: "output/my_output.mp4"

# 模型路径
yolo_pose_model_path: "models/yolov8n-pose.pt"
racket_yolo_model_path: "models/yolov8n.pt"

# 球检测参数 (根据视频调整)
ball_confidence_threshold: 0.75
min_ball_radius: 18
max_ball_radius: 45
hsv_lower_hue: 20
hsv_upper_hue: 70

# ROI设置
roi_settings:
  enabled: true
  interactive_selection: false
  auto_load_config: true
  roi_config_path: "configs/my_roi_config.yaml"

# 精彩瞬间设置
highlights:
  enabled: true
  output_dir: "output/highlights/"
  hit_distance_factor: 1.5
  distance_scale: 5
  max_highlights: 50

# 性能设置
performance_optimization:
  headless: true
  minimal_ui: true

# 显示设置
display_options:
  show_swing_type: true
  show_ball_position: true
  show_ball_trajectory: true
  show_racket_state: true
  show_fps: true
```

---

## 🚀 快速命令参考

### 常用命令
```bash
# 激活环境
source venv/bin/activate

# 综合分析
python3 main.py --config configs/comprehensive_tennis_config.yaml --input "video.mp4"

# 性能优先
python3 main.py --config configs/performance_optimized_config.yaml --input "video.mp4"

# ROI演示
python demo_roi_detection.py

# 球检测测试
python test_enhanced_ball_detection.py

# 查看帮助
python3 main.py --help
```

### 快捷脚本
创建 `quick_run.sh`:
```bash
#!/bin/bash
source venv/bin/activate
python3 main.py \
  --config configs/comprehensive_tennis_config.yaml \
  --input "$1" \
  --output_dir "output/$(date +%Y%m%d_%H%M%S)/" \
  --original_name "$(basename $1)"
```

使用:
```bash
chmod +x quick_run.sh
./quick_run.sh data/tennis_match.mp4
```

---

**更新时间**: 2026-01-29  
**版本**: 1.0  
**维护者**: Tennis Analyzer Team
