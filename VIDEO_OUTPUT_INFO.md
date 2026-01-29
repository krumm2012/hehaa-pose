# 视频输出路径说明

## 📹 视频输出配置

### 默认输出路径

**配置文件**: `configs/yolo26_tennis_config.yaml`

```yaml
video_output_path: "data/output_video.mp4"
```

**实际路径**: `/Users/krum5539/Documents/tennis_analyzer/data/output_video.mp4`

---

## 📁 当前输出文件

### 最新输出

```bash
$ ls -lh data/output_video.mp4

-rw-r--r--  1 krum5539  staff   7.0M Jan 29 21:11 data/output_video.mp4
```

**文件信息**:
- 大小: 7.0 MB
- 最后修改: 2026-01-29 21:11 (刚刚运行的结果)
- 路径: `data/output_video.mp4`

---

## 🔧 如何修改输出路径

### 方法1: 修改配置文件

```yaml
# configs/yolo26_tennis_config.yaml

video_output_path: "data/output_video.mp4"  # 修改这里
```

**示例**:
```yaml
# 输出到不同目录
video_output_path: "output/result.mp4"

# 输出到绝对路径
video_output_path: "/tmp/tennis_output.mp4"

# 按日期命名
video_output_path: "data/output_2026-01-29.mp4"
```

### 方法2: 命令行参数

```bash
# 使用 --output 参数
python3 main.py \
  --config configs/yolo26_tennis_config.yaml \
  --input data/16.10.mp4 \
  --output data/my_output.mp4
```

### 方法3: 使用输出目录

```bash
# 使用 --output-dir 参数
python3 main.py \
  --config configs/yolo26_tennis_config.yaml \
  --input data/16.10.mp4 \
  --output-dir results/
```

---

## 📊 视频 I/O 性能分析

### 当前配置

```
输出格式: MP4 (H.264/AVC)
编码器: avc1
分辨率: 2560x1440
FPS: 25
```

### 性能开销

```
写入时间: 8.64ms/帧
占比: 30.3% (总处理时间)
```

**说明**: 
- 视频编码和写入是第2大性能开销
- 高分辨率 (2560x1440) 导致编码时间较长

---

## 💡 优化建议

### 1. 降低输出分辨率 ⭐⭐⭐⭐

**修改代码**: `main.py`

```python
# 当前
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'avc1'),
                      fps,
                      (frame_width, frame_height))

# 优化 - 降低到 1280x720
output_width = frame_width // 2
output_height = frame_height // 2
display_frame_resized = cv2.resize(display_frame, (output_width, output_height))

out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'avc1'),
                      fps,
                      (output_width, output_height))

out.write(display_frame_resized)
```

**预期效果**:
- 写入时间: 8.64ms → ~3-4ms
- 节省: ~5ms/帧
- FPS 提升: ~20%
- 文件大小: 减少 75%

### 2. 使用更快的编码器 ⭐⭐⭐

```python
# 当前: avc1 (H.264)
cv2.VideoWriter_fourcc(*'avc1')

# 优化: MJPEG (更快但文件更大)
cv2.VideoWriter_fourcc(*'MJPG')

# 或: mp4v (兼容性好)
cv2.VideoWriter_fourcc(*'mp4v')
```

**预期效果**:
- 写入时间: 可能减少 20-30%
- 文件大小: 可能增加

### 3. 禁用视频输出 (仅分析) ⭐⭐⭐⭐⭐

**添加配置选项**:

```yaml
# configs/yolo26_tennis_config.yaml

video_output:
  enabled: true  # 改为 false 禁用视频输出
  path: "data/output_video.mp4"
```

**预期效果**:
- 节省: 8.64ms/帧
- FPS 提升: ~30%
- 仅保存分析数据

---

## 📝 其他输出文件

### 高亮视频 (已禁用)

```
路径: data/highlights/
文件: highlight_XXXXXX_hit.mp4
状态: 已禁用 ✅
```

### 分析数据

```
CSV: 根据配置
JSON: data/highlights/*.json (已禁用)
```

---

## 🎯 推荐配置

### 场景1: 最快处理速度

```yaml
video_output:
  enabled: false  # 禁用视频输出

highlights:
  enabled: false  # 禁用高亮保存
```

**FPS**: ~18-23

### 场景2: 平衡模式

```yaml
video_output:
  enabled: true
  resolution_scale: 0.5  # 降低到 1280x720

highlights:
  enabled: false
```

**FPS**: ~14-18

### 场景3: 完整输出

```yaml
video_output:
  enabled: true
  resolution_scale: 1.0  # 保持原分辨率

highlights:
  enabled: true
```

**FPS**: ~9-12

---

## 📍 总结

**当前输出路径**: `data/output_video.mp4`

**性能影响**:
- 写入时间: 8.64ms/帧 (30.3%)
- 第2大性能开销

**优化建议**:
1. 降低输出分辨率 → 节省 ~5ms
2. 使用更快编码器 → 节省 ~2ms
3. 禁用视频输出 → 节省 ~9ms

**最大 FPS 提升**: 通过禁用视频输出可提升 30%
