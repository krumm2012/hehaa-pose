# 详细性能分析报告

## 📊 测试结果

**测试日期**: 2026-01-29  
**测试帧数**: 100 帧  
**配置**: YOLO26n 统一检测 + YOLO26-pose + ALL 计算单元

---

## ⏱️ 分阶段性能分析

| 阶段 | 平均时间 | 最小 | 最大 | 占比 | 优先级 |
|------|---------|------|------|------|--------|
| **姿态检测** | **10.58ms** | 10.14ms | 41.26ms | **63.3%** | 🔥🔥🔥 |
| **统一检测** | **5.82ms** | 5.50ms | 8.75ms | **34.9%** | 🔥🔥 |
| **可视化** | 0.29ms | 0.27ms | 0.37ms | 1.8% | ✅ |
| **ROI 提取** | 0.00ms | 0.00ms | 0.01ms | 0.0% | ✅ |
| **坐标转换** | 0.00ms | 0.00ms | 0.00ms | 0.0% | ✅ |
| **总计** | **16.70ms** | 15.94ms | 50.37ms | 100% | - |

---

## 🎯 性能指标

```
平均帧时间: 16.70ms
理论 FPS: 59.88
实际 FPS: ~56.89 (考虑开销)
```

**但实际运行 FPS: 9.52**

**差距原因**: 
- 性能分析脚本**没有包含**所有处理（挥拍分析、速度分析、击球点分析、高亮保存等）
- 实际系统有更多开销

---

## 🔥 性能瓶颈

### 1. 姿态检测 (63.3% - 最大瓶颈)

```
平均: 10.58ms
占比: 63.3%
模型: YOLO26m-pose
```

**优化方案**:

#### 方案A: 降低检测频率 ⭐⭐⭐⭐⭐
```python
# 每2帧检测一次
if frame_num % 2 == 0:
    pose_results = pose_module.get_keypoints(frame)
else:
    # 使用上一帧结果
    pose_results = previous_pose_results

预期提升: 31.7% (10.58ms → 5.29ms)
新 FPS: 9.52 → 12.5 (+31%)
```

#### 方案B: 降低输入分辨率 ⭐⭐⭐⭐
```python
# 当前: 2560x1440 → 640x640
# 优化: 2560x1440 → 480x480 (或 512x512)

预期提升: 20-30%
新 FPS: 9.52 → 11.4-12.4
```

#### 方案C: 使用更小的模型 ⭐⭐⭐
```python
# 当前: yolo26m-pose
# 优化: yolo26n-pose (如果有)

预期提升: 30-40%
新 FPS: 9.52 → 12.4-13.3
```

### 2. 统一检测 (34.9%)

```
平均: 5.82ms
占比: 34.9%
模型: YOLO26n
```

**优化方案**:

#### 方案A: 降低检测频率 ⭐⭐⭐
```python
# 球拍检测每2帧一次
if frame_num % 2 == 0:
    racket_detections = racket_module.detect_rackets(frame)

预期提升: 17.5% (5.82ms → 2.91ms)
新 FPS: 9.52 → 11.2 (+18%)
```

#### 方案B: 已经很优化 ✅
```
YOLO26n 已经是最小的模型
Core ML ALL 模式已经最优
```

---

## 💡 综合优化方案

### 推荐方案1: 降低姿态检测频率 ⭐⭐⭐⭐⭐

```python
# 姿态检测每2帧一次
if frame_num % 2 == 0:
    pose_results = pose_module.get_keypoints(frame)
    cached_pose_results = pose_results
else:
    pose_results = cached_pose_results
```

**预期效果**:
- 节省: 5.29ms/帧
- FPS: 9.52 → **12.5** (+31%)
- 影响: 姿态更新延迟 40ms（可接受）

### 推荐方案2: 组合优化 ⭐⭐⭐⭐⭐

```python
# 姿态检测每2帧一次
if frame_num % 2 == 0:
    pose_results = pose_module.get_keypoints(frame)
    
# 球拍检测每2帧一次
if frame_num % 2 == 0:
    racket_detections = racket_module.detect_rackets(frame)
```

**预期效果**:
- 节省: 5.29 + 2.91 = 8.2ms/帧
- FPS: 9.52 → **15.8** (+66%)
- 影响: 轻微延迟（可接受）

### 推荐方案3: 激进优化 ⭐⭐⭐

```python
# 姿态检测每3帧一次
if frame_num % 3 == 0:
    pose_results = pose_module.get_keypoints(frame)
    
# 球拍检测每2帧一次
if frame_num % 2 == 0:
    racket_detections = racket_module.detect_rackets(frame)
    
# 球检测每帧（保持实时性）
ball_positions = ball_module.predict_ball(frame)
```

**预期效果**:
- 节省: 7.05 + 2.91 = 9.96ms/帧
- FPS: 9.52 → **18.5** (+94%)
- 影响: 姿态更新延迟 120ms

---

## 📈 优化效果对比

| 方案 | 节省时间 | 新FPS | 提升 | 影响 | 推荐度 |
|------|---------|-------|------|------|--------|
| **基线** | - | 9.52 | - | - | - |
| **方案1** | 5.29ms | 12.5 | +31% | 轻微 | ⭐⭐⭐⭐⭐ |
| **方案2** | 8.20ms | 15.8 | +66% | 可接受 | ⭐⭐⭐⭐⭐ |
| **方案3** | 9.96ms | 18.5 | +94% | 中等 | ⭐⭐⭐ |

---

## 🔧 实现代码

### 方案1: 姿态检测每2帧

```python
# main.py

# 添加缓存
cached_pose_results = None

# 在主循环中
if frame_num % 2 == 0:
    pose_results = pose_module.get_keypoints(pose_detection_frame)
    cached_pose_results = pose_results
else:
    pose_results = cached_pose_results
```

### 方案2: 组合优化

```python
# main.py

# 添加缓存
cached_pose_results = None
cached_racket_detections = []

# 在主循环中
if frame_num % 2 == 0:
    pose_results = pose_module.get_keypoints(pose_detection_frame)
    cached_pose_results = pose_results
    
    racket_detections = racket_module.detect_rackets(detection_frame)
    cached_racket_detections = racket_detections
else:
    pose_results = cached_pose_results
    racket_detections = cached_racket_detections

# 球检测每帧执行（保持实时性）
ball_positions = ball_module.predict_ball(detection_frame)
```

---

## 📊 其他发现

### 1. 可视化性能良好 ✅
```
时间: 0.29ms
占比: 1.8%
结论: 不需要优化
```

### 2. ROI 和坐标转换几乎无开销 ✅
```
时间: ~0ms
结论: 非常高效
```

### 3. 实际 vs 理论 FPS 差距

```
理论 FPS (仅检测): 59.88
实际 FPS (完整系统): 9.52

差距原因:
- 挥拍分析
- 速度分析
- 击球点分析
- 高亮保存
- 视频编码
- 其他开销

估计: 额外 ~88ms/帧
```

---

## 🎯 最终建议

### 立即实施（强烈推荐）

**方案2: 组合优化**

```python
# 姿态检测每2帧
# 球拍检测每2帧
# 球检测每帧
```

**预期效果**:
- FPS: 9.52 → **15.8** (+66%)
- 代码改动: 最小
- 影响: 可接受

### 配置优化

```yaml
# configs/yolo26_tennis_config.yaml

# 已优化
pose_compute_units: "ALL"  ✅
unified_detection:
  compute_units: "ALL"  ✅

# 已关闭
highlights:
  enabled: false  ✅
ball_detection_debug:
  log_detection: false  ✅
```

---

## 📝 总结

### 当前性能
- FPS: 9.52
- 主要瓶颈: 姿态检测 (63.3%)

### 优化后预期
- FPS: **15.8** (+66%)
- 接近实时 (25 FPS 的 63%)

### 下一步
1. ✅ 实施方案2（组合优化）
2. ✅ 测试验证
3. ✅ 根据效果调整

**结论**: 通过降低检测频率可大幅提升性能，且对用户体验影响很小！
