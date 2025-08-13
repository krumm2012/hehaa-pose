# 🎾 球检测调试图像生成脚本使用指南

## 📄 脚本介绍

`generate_ball_debug_frames.py` 是一个专门用于生成指定帧编号球检测调试图像的脚本，基于 `default_config.yaml` 配置自动生成带球识别框的图像。

## 🚀 快速使用

### 基本用法
```bash
# 生成指定帧的调试图像
python generate_ball_debug_frames.py 5 15 25 50

# 指定不同的配置文件
python generate_ball_debug_frames.py 10 20 30 --config configs/balanced_config.yaml

# 指定输出目录
python generate_ball_debug_frames.py 5 10 --output my_debug_images
```

### 参数说明

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `frames` | 整数列表 | 要处理的帧编号（必需） | `5 10 25 50` |
| `--config`, `-c` | 文件路径 | 配置文件路径（可选） | `configs/default_config.yaml` |
| `--output`, `-o` | 目录路径 | 输出目录（可选） | `debug_frames` |

## 📊 输出内容

### 生成的文件
- **调试图像**: `debug_frames/ball_frame_XXXX_debug.jpg`
- **格式**: 4位数帧编号，如 `ball_frame_0005_debug.jpg`

### 图像内容
- ✅ **绿色圆圈**: 检测到的球位置
- ✅ **球编号**: Ball 1, Ball 2 等标识
- ✅ **坐标显示**: 每个球的精确坐标 (x,y)
- ✅ **边界框**: 黄色矩形显示检测边界（如果启用）
- ✅ **帧信息**: 帧编号、检测球数、分辨率等

## 📝 使用示例

### 示例1: 基础使用
```bash
# 生成帧 5, 15, 25, 50 的调试图像
python generate_ball_debug_frames.py 5 15 25 50
```

**输出**:
```
🎯 球检测调试图像生成器
📄 配置文件: configs/default_config.yaml
🎬 目标帧编号: [5, 15, 25, 50]
📁 输出目录: debug_frames
======================================================================
✅ 帧 5: 检测到 0 个球
📸 已保存: debug_frames/ball_frame_0005_debug.jpg
...
🎉 成功生成 4 个调试图像！
```

### 示例2: 连续帧分析
```bash
# 分析连续帧（30-35）查看球运动轨迹
python generate_ball_debug_frames.py 30 31 32 33 34 35
```

### 示例3: 关键时刻分析
```bash
# 分析击球时刻前后的帧
python generate_ball_debug_frames.py 45 46 47 48 49 50
```

### 示例4: 使用优化配置
```bash
# 使用调试优化配置生成图像
python generate_ball_debug_frames.py 5 10 15 --config configs/debug_optimized_config.yaml
```

## 🔍 诊断功能

### 自动诊断信息
脚本会显示详细的检测过程：
- 🎯 **HSV检测**: 颜色匹配像素数、圆形检测结果
- 📏 **尺寸过滤**: 球半径验证、过滤原因
- 🎯 **边界检查**: 位置验证、边界内外状态
- 📊 **最终结果**: 通过所有过滤的球数量

### 配置对比
```bash
# 对比不同配置的检测效果
python generate_ball_debug_frames.py 10 --config configs/default_config.yaml
python generate_ball_debug_frames.py 10 --config configs/debug_optimized_config.yaml
```

## 🛠️ 高级用法

### 批量帧分析
```bash
# 每隔10帧生成一次，覆盖整个视频
python generate_ball_debug_frames.py 10 20 30 40 50 60 70 80 90 100
```

### 特定场景分析
```bash
# 发球场景分析
python generate_ball_debug_frames.py 1 2 3 4 5

# 击球场景分析  
python generate_ball_debug_frames.py 40 45 50 55 60

# 网前截击分析
python generate_ball_debug_frames.py 70 75 80 85 90
```

## 📈 优化建议

### 根据结果调整配置
1. **如果完全检测不到球**:
   - 检查HSV颜色范围是否合适
   - 验证球尺寸限制是否过严
   - 确认边界设置是否合理

2. **如果误检太多**:
   - 提高置信度阈值
   - 缩小球尺寸范围
   - 缩小检测边界

3. **如果遗漏真球**:
   - 放宽尺寸限制
   - 扩大检测边界
   - 调整HSV颜色范围

## 🔧 故障排除

### 常见问题

**Q: 脚本运行但不生成图像**
A: 检查帧编号是否在视频范围内 (1-100)

**Q: 检测结果全是0个球**
A: 使用 `configs/debug_optimized_config.yaml` 配置

**Q: 生成的图像看不到球**
A: 球可能被边界过滤，检查日志输出的边界检查信息

**Q: 帧编号超出范围**
A: 使用 `--help` 查看视频总帧数，或先运行小范围测试

### 获取帮助
```bash
python generate_ball_debug_frames.py --help
```

## 💡 最佳实践

1. **先测试单帧**: `python generate_ball_debug_frames.py 10`
2. **查看日志输出**: 了解检测过程和过滤原因
3. **对比不同配置**: 找到最适合的参数设置
4. **批量生成前先验证**: 避免生成大量无用图像

---

*脚本版本: v1.0 | 更新日期: 2024-05-23* 