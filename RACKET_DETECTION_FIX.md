# 球拍检测优化报告

## 🔍 问题

**球拍检测框过大**

从用户提供的截图可以看到，绿色的球拍检测框几乎覆盖了整个上半身区域，而实际的网球拍只是右手持拍的部分。

---

## 🛠️ 优化方案

### 1. 提高置信度阈值 ✅

**修改前**:
```yaml
racket_confidence_threshold: 0.3
```

**修改后**:
```yaml
racket_confidence_threshold: 0.5  # 提高以过滤误检
```

**效果**: 过滤掉低置信度的误检

---

### 2. 添加边界框大小过滤 ✅

**新增逻辑** (`yolo26n_unified_detector.py`):

```python
# 计算边界框面积
box_width = box[2] - box[0]
box_height = box[3] - box[1]
box_area = box_width * box_height

# 过滤过大的边界框（可能是误检）
# 球拍通常不会超过图像的 1/4
max_area = (self.original_width * self.original_height) / 4

# 过滤过小的边界框
min_area = 1000  # 最小 1000 像素²

# 宽高比检查（球拍通常是长条形）
aspect_ratio = box_height / max(box_width, 1)
min_aspect_ratio = 0.5  # 最小宽高比
max_aspect_ratio = 5.0  # 最大宽高比

if (min_area <= box_area <= max_area and 
    min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
    # 保留检测
    racket_detections.append({...})
```

**过滤条件**:
1. **面积限制**: 1000 ≤ 面积 ≤ 图像面积/4
2. **宽高比限制**: 0.5 ≤ 宽高比 ≤ 5.0

---

## 📊 优化效果

### 测试结果

```
处理时间: 26.27秒
平均 FPS: 9.52
```

**性能**: 无明显影响 ✅

---

## 🎯 过滤规则详解

### 1. 面积过滤

**目的**: 过滤过大或过小的检测框

```python
# 最小面积: 1000 像素²
# 对于 2560x1440 的图像:
# - 最小边界框约: 32x32 像素
# - 最大边界框约: 1280x720 像素 (图像的 1/4)

min_area = 1000
max_area = (2560 * 1440) / 4 = 921,600 像素²
```

**示例**:
- ✅ 合理的球拍框: 100x400 = 40,000 像素²
- ❌ 过大的框: 1000x1000 = 1,000,000 像素² (超过 max_area)
- ❌ 过小的框: 20x20 = 400 像素² (小于 min_area)

### 2. 宽高比过滤

**目的**: 确保检测框符合球拍的形状特征

```python
# 球拍通常是长条形
# 宽高比 = 高度 / 宽度

aspect_ratio = box_height / box_width

# 允许范围: 0.5 ~ 5.0
# - 横向球拍: 约 0.5 (宽 > 高)
# - 竖向球拍: 约 3-5 (高 > 宽)
```

**示例**:
- ✅ 竖向球拍: 100x400 → 宽高比 = 4.0
- ✅ 横向球拍: 300x150 → 宽高比 = 0.5
- ❌ 正方形框: 500x500 → 宽高比 = 1.0 (可能通过，但需要结合面积)
- ❌ 极端矩形: 50x1000 → 宽高比 = 20 (超出范围)

---

## 🔧 调优参数

如果检测效果仍不理想，可以调整以下参数：

### 1. 置信度阈值

```yaml
# configs/yolo26_tennis_config.yaml

racket_confidence_threshold: 0.5  # 当前值

# 调整建议:
# - 如果漏检: 降低到 0.4
# - 如果误检: 提高到 0.6-0.7
```

### 2. 面积限制

```python
# yolo26n_unified_detector.py

min_area = 1000  # 当前值

# 调整建议:
# - 如果小球拍被过滤: 降低到 500
# - 如果仍有过大框: 降低 max_area 到 1/6 或 1/8
```

### 3. 宽高比限制

```python
min_aspect_ratio = 0.5  # 当前值
max_aspect_ratio = 5.0  # 当前值

# 调整建议:
# - 如果横向球拍被过滤: 降低 min_aspect_ratio 到 0.3
# - 如果竖向球拍被过滤: 提高 max_aspect_ratio 到 6.0-7.0
```

---

## 📝 验证步骤

### 1. 运行测试

```bash
python3 main.py --config configs/yolo26_tennis_config.yaml --input data/16.10.mp4
```

### 2. 检查输出视频

```bash
open data/output_video.mp4
```

### 3. 观察球拍检测框

**期望结果**:
- ✅ 检测框紧贴球拍
- ✅ 不包含过多背景
- ✅ 稳定跟踪

**如果仍有问题**:
- 检查检测框的 `area` 和 `aspect_ratio` 值
- 根据实际情况调整参数

---

## 🎯 总结

### 已实施的优化

1. ✅ **提高置信度阈值**: 0.3 → 0.5
2. ✅ **添加面积过滤**: 1000 ≤ 面积 ≤ 图像/4
3. ✅ **添加宽高比过滤**: 0.5 ≤ 宽高比 ≤ 5.0

### 预期效果

- ✅ 过滤过大的检测框
- ✅ 过滤误检
- ✅ 提高检测准确性
- ✅ 性能无明显影响

### 下一步

1. 查看新的输出视频
2. 验证球拍检测框大小是否合理
3. 如需要，根据实际情况微调参数

---

**优化日期**: 2026-01-29  
**状态**: 已实施，待验证
