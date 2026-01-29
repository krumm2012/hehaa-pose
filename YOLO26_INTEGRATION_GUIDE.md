# YOLO26 Core ML 集成指南

## 📋 概述

您已经成功将 YOLO26n Core ML 模型集成到网球分析系统中，用于球和球拍检测。

## ✅ 已完成的工作

### 1. 模型信息
- **模型文件**: `yolo26n.mlpackage`
- **作者**: Ultralytics
- **版本**: 8.4.0
- **训练数据集**: COCO

### 2. 模型规格
- **输入**:
  - 名称: `image`
  - 类型: Image (RGB)
  - 尺寸: 640×640
  
- **输出**:
  - 名称: `var_1441`
  - 类型: MultiArray
  - 形状: `[1, 300, 6]`
  - 格式: `[x_center, y_center, width, height, confidence, class]`

### 3. 类别映射
- **Class 0**: Ball (球)
- **Class 1**: Racket (球拍)

## 📁 创建的文件

### 1. `yolo26_detector.py`
核心检测器类，提供统一的球和球拍检测接口。

**主要功能**:
- 加载 Core ML 模型
- 图像预处理 (BGR → RGB, 调整大小)
- 模型推理
- 后处理 (坐标缩放, NMS)
- 分离球和球拍检测结果

**使用示例**:
```python
from yolo26_detector import YOLO26Detector

# 创建检测器
config = {
    'ball_confidence_threshold': 0.5,
    'racket_confidence_threshold': 0.4
}
detector = YOLO26Detector('yolo26n.mlpackage', config)

# 检测
ball_detections, racket_detections = detector.detect(frame)

# 或分别检测
balls = detector.detect_balls(frame)
rackets = detector.detect_rackets(frame)
```

### 2. `inspect_yolo26_model.py`
模型检查工具，用于查看模型的输入输出规格。

**使用**:
```bash
python3 inspect_yolo26_model.py yolo26n.mlpackage
```

### 3. `test_yolo26_detector_simple.py`
简单的测试脚本，验证检测器功能。

**使用**:
```bash
python3 test_yolo26_detector_simple.py
```

## 🔧 集成到现有系统

### 方案A: 替换球检测模块

修改 `ball_tracker.py` 使用 YOLO26 检测器：

```python
# 在 BallTracker.__init__ 中
from yolo26_detector import YOLO26Detector

self.yolo26_detector = YOLO26Detector(
    'yolo26n.mlpackage',
    {
        'ball_confidence_threshold': config.get('ball_confidence_threshold', 0.5)
    }
)

# 在 predict_ball 方法中
def predict_ball(self, frame):
    # 使用 YOLO26 检测
    ball_detections = self.yolo26_detector.detect_balls(frame)
    
    # 转换格式
    detected_balls = []
    for det in ball_detections:
        x1, y1, x2, y2 = det['box']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        r = (x2 - x1) / 2  # 假设球是圆形
        
        detected_balls.append((cx, cy, r, det['confidence']))
    
    return detected_balls
```

### 方案B: 替换球拍检测模块

修改 `racket_detector.py` 使用 YOLO26 检测器：

```python
# 在 RacketDetector.__init__ 中
from yolo26_detector import YOLO26Detector

self.yolo26_detector = YOLO26Detector(
    'yolo26n.mlpackage',
    {
        'racket_confidence_threshold': config.get('racket_confidence_threshold', 0.4)
    }
)

# 在 detect_rackets 方法中
def detect_rackets(self, frame):
    # 使用 YOLO26 检测
    racket_detections = self.yolo26_detector.detect_rackets(frame)
    
    # 格式已经兼容，直接返回
    return racket_detections
```

### 方案C: 统一检测器（推荐）

创建新的统一检测模块 `unified_detector.py`：

```python
from yolo26_detector import YOLO26Detector
from pose_estimator_yolo26 import PoseEstimatorYOLO26

class UnifiedDetector:
    \"\"\"统一的检测器，使用 YOLO26 进行所有检测\"\"\"
    
    def __init__(self, config):
        # 姿态检测
        self.pose_detector = PoseEstimatorYOLO26(
            config['yolo_pose_model_path'],
            config
        )
        
        # 球和球拍检测
        self.object_detector = YOLO26Detector(
            config['yolo26_model_path'],
            config
        )
    
    def detect_all(self, frame):
        \"\"\"一次性检测所有目标\"\"\"
        # 姿态
        keypoints = self.pose_detector.get_keypoints(frame)
        
        # 球和球拍
        balls, rackets = self.object_detector.detect(frame)
        
        return {
            'keypoints': keypoints,
            'balls': balls,
            'rackets': rackets
        }
```

## ⚙️ 配置更新

更新 `configs/yolo26_tennis_config.yaml`：

```yaml
# 🆕 YOLO26 模型路径
yolo26_model_path: "yolo26n.mlpackage"
yolo_pose_model_path: "models/yolo26m-pose.mlpackage"

# 🆕 YOLO26 检测参数
yolo26_detection:
  ball_confidence_threshold: 0.5
  racket_confidence_threshold: 0.4
  nms_threshold: 0.5
  max_detections: 100
```

## 🚀 下一步

1. **测试实际视频**: 使用真实网球视频测试检测效果
2. **调整阈值**: 根据测试结果优化置信度阈值
3. **集成到主程序**: 选择合适的集成方案
4. **性能优化**: 根据实际FPS进行优化

## 🎯 总结

✅ YOLO26 Core ML 模型已成功集成  
✅ 检测器类已创建并测试  
✅ 支持球和球拍同时检测  
✅ 提供多种集成方案  

**推荐**: 使用方案C（统一检测器），可以最大化利用 YOLO26 的性能优势。

---

**创建日期**: 2026-01-29  
**模型**: yolo26n.mlpackage  
**状态**: 已集成，待实际测试
