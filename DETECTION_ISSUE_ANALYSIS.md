# 球拍和网球检测问题分析

## 🔍 问题描述

用户反馈:
1. 球拍识别有问题
2. 网球标识框大小不对

## 📊 当前实现分析

### 球的绘制 (main.py:531)

```python
cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
          5, (0, 255, 0), -1)  # 固定半径 5 像素
```

**问题**: 半径固定为 5 像素，太小！

### 球的检测数据 (yolo26n_unified_detector.py:147-152)

```python
ball_detections.append({
    'position': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],  # 中心点
    'box': box,  # [x1, y1, x2, y2]
    'confidence': float(conf),
    'radius': int((box[2] - box[0]) / 2)  # 计算的半径
})
```

**发现**: 检测数据包含 `radius` 和 `box`，但绘制时没有使用！

### 球拍的绘制 (main.py:761-762)

```python
cv2.rectangle(display_frame, (int(box[0]), int(box[1])), 
             (int(box[2]), int(box[3])), (255, 0, 0), 2)
```

**看起来正常**，需要查看实际检测数据。

## 🔧 修复方案

### 方案1: 使用检测到的球半径

```python
# 当前
cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
          5, (0, 255, 0), -1)

# 修复后
if len(ball_position) >= 3:
    radius = int(ball_position[2])  # 使用检测到的半径
else:
    radius = 10  # 默认半径
cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
          radius, (0, 255, 0), -1)
```

### 方案2: 同时绘制球的边界框

```python
# 绘制球的圆形
cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
          radius, (0, 255, 0), -1)

# 绘制球的边界框（调试用）
if len(ball_position) >= 4:
    # ball_position 格式: [x, y, radius, confidence, ...]
    # 或从 ball detection 获取 box
    pass
```

### 方案3: 检查球拍检测数据

需要查看实际的 `racket_detections` 数据格式和值。

## 🎯 建议

1. **立即修复**: 使用检测到的球半径
2. **调试**: 添加日志输出检测数据
3. **验证**: 重新运行并检查结果

## 📝 修复代码

见下一个文件...
