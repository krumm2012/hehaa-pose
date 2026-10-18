# YOLO26m-Pose 网球拍 7 关键点模型使用文档

**模型版本**：v2.0 Gold Master\
**支持格式**：Apple Core ML (`.mlpackage`)、PyTorch (`.pt`)、ONNX (`.onnx`)\
**适用场景**：网球击球动作分析、球拍 3D 姿态重建约束、球员握拍贴合、甜区（Sweet Spot）击球核验。

---

## 1. 模型概览与技术规格

### 1.1 模型规格一览

| 规格项 | Core ML FP16 (推荐) | Core ML INT8 | Core ML 640 极速版 | PyTorch (MPS) |
|---|---|---|---|---|
| **文件名** | `racket_pose_yolo26m_fp16.mlpackage` | `racket_pose_yolo26m_int8.mlpackage` | `racket_pose_yolo26m_fp16_640.mlpackage` | `best.pt` |
| **输入尺寸** | `960 × 960` (RGB) | `960 × 960` (RGB) | `640 × 640` (RGB) | `960 × 960` (RGB) |
| **输出张量** | `var_2295: (1, 26, 18900)` | `var_2295: (1, 26, 18900)` | `var_2295: (1, 26, 8400)` | `(1, 26, 18900)` |
| **单帧时延** | **`24.8 ms` (~40.2 FPS)** | **`24.8 ms` (~40.2 FPS)** | **`9.0 ms` (~111.0 FPS)** | `84.4 ms` (~11.8 FPS) |
| **存储体积** | 40.0 MB | **20.0 MB (体积减半)** | 40.0 MB | 40.3 MB |
| **硬件调度** | Apple Silicon ANE + GPU | Apple Silicon ANE + GPU | Apple Silicon ANE + GPU | Apple Silicon MPS |

---

## 2. 标准 7 关键点拓扑定义 (7-Keypoint Topology)

模型检测类别为单类别 `tennis_racket`（类别索引 0）。标准球拍骨架由 **7 个解剖学基准点** 定义：

```
                 4: head_top (拍头 12 点钟外沿)
                   /          \
     5: frame_left (9点)     6: frame_right (3点)
             \      3: sweet_spot (甜区/网线几何中心) /
               \          /
                 2: throat (拍喉三角连接中心)
                     |
               1: grip_center (握持中心/手腕贴合锚点)
                     |
                0: butt_cap (拍柄底盖中心)
```

### 关键点与连通骨架说明：
- **0 - butt_cap**：拍柄底盖中心（握把末端基准）。
- **1 - grip_center**：握拍中心点（右手/左手腕贴合基准点）。
- **2 - throat**：拍喉三角连接处中心。
- **3 - sweet_spot**：拍面几何中心（击球点/甜区核心真值）。
- **4 - head_top**：拍头正上方 12 点钟位置外沿。
- **5 - frame_left**：拍面左侧 9 点钟位置边框。
- **6 - frame_right**：拍面右侧 3 点钟位置边框。
- **8 条连接骨骼 (SKELETON_EDGES)**：
  `[(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (2, 6), (5, 4), (6, 4)]`
- **水平镜像对称翻转序列 (Flip Indices)**：
  `[0, 1, 2, 3, 4, 6, 5]`（当图像做水平翻转时，5 与 6 互换）。

---

## 3. 输出张量格式与 26 维通道详解

模型输出张量形状为：
$$\text{Output Shape} = [1, 26, 18900]$$

> [!IMPORTANT]
> **26 是特征通道数，不是 26 个关键点！**
> 18,900 是 960×960 分辨率下 3 个检测特征层的全部候选锚点数（$120\times120 + 60\times60 + 30\times30 = 18900$）。
> 必须先将张量转置为 `(18900, 26)`，每一行代表一个候选锚框。

### 26 个数值的完整字段布局映射：

| 通道索引 | 字段名称 | 物理含义 | 取值范围 |
|---|---|---|---|
| **0** | `box_center_x` | 球拍包围框中心 X 坐标（模型输入像素系） | $0 \sim 960$ |
| **1** | `box_center_y` | 球拍包围框中心 Y 坐标（模型输入像素系） | $0 \sim 960$ |
| **2** | `box_width` | 球拍包围框宽度 | $> 0$ |
| **3** | `box_height` | 球拍包围框高度 | $> 0$ |
| **4** | `confidence` | **球拍检测置信度得分**（过滤阈值建议 $\ge 0.25$） | $0.0 \sim 1.0$ |
| **5, 6, 7** | `butt_cap` | **关键点 0 (底盖)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |
| **8, 9, 10** | `grip_center` | **关键点 1 (握柄)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |
| **11, 12, 13** | `throat` | **关键点 2 (拍喉)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |
| **14, 15, 16** | `sweet_spot` | **关键点 3 (甜区)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |
| **17, 18, 19** | `head_top` | **关键点 4 (拍头)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |
| **20, 21, 22** | `frame_left` | **关键点 5 (左框)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |
| **23, 24, 25** | `frame_right` | **关键点 6 (右框)**：`[x, y, visibility]` | $x, y \in [0, 960], v \in [0, 1]$ |

**总计**：$4 \text{ (框)} + 1 \text{ (置信度)} + 7 \times 3 \text{ (7 关键点)} = 26$ 个数值。

---

## 4. 快速开始与推理代码示例

### 4.1 方案 A：使用 Core ML 原生推理（macOS Apple Silicon 推荐，25ms）

无需依赖 PyTorch，纯 Python + CoreML + NumPy 极速运行：

```python
import coremltools as ct
from PIL import Image
import numpy as np
import cv2

# 1. 加载 Core ML 模型 (推荐 ComputeUnit.ALL 混合加速)
model_path = "models/racket_pose/racket_pose_yolo26m_fp16.mlpackage"
model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)

# 2. 预处理 (读取图像并缩放至 960x960)
img_bgr = cv2.imread("frame.jpg")
orig_h, orig_w = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb).resize((960, 960), Image.BILINEAR)

# 3. Core ML 推理 (耗时仅 ~25ms)
preds = model.predict({"image": img_pil})
output_key = list(preds.keys())[0]  # 通常是 'var_2295'
raw_output = preds[output_key][0]   # 形状: (26, 18900)

# 4. 解码与转置
candidates = raw_output.T           # 形状: (18900, 26)

# 5. 置信度过滤 (阈值取 0.25)
conf_thresh = 0.25
scores = candidates[:, 4]
valid_mask = scores >= conf_thresh
selected = candidates[valid_mask]

if len(selected) > 0:
    # 转换中心坐标 (xc, yc, w, h) -> (x1, y1, x2, y2)
    boxes = selected[:, :4].copy()
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2

    # NMS 非极大值抑制
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), selected[:, 4].tolist(),
        score_threshold=conf_thresh, nms_threshold=0.45
    )

    sx = orig_w / 960.0
    sy = orig_h / 960.0

    keypoint_names = [
        "butt_cap", "grip_center", "throat", "sweet_spot",
        "head_top", "frame_left", "frame_right"
    ]

    for idx in np.asarray(indices).reshape(-1):
        row = selected[int(idx)]
        racket_score = float(row[4])

        # 提取 7 关键点并缩放回原图尺寸
        kpts_raw = row[5:].reshape(7, 3)
        kpts_dict = {}
        for k_idx, name in enumerate(keypoint_names):
            kpts_dict[name] = {
                "x": float(kpts_raw[k_idx, 0] * sx),
                "y": float(kpts_raw[k_idx, 1] * sy),
                "conf": float(kpts_raw[k_idx, 2]),
            }

        print(f"🎯 检出球拍! 置信度: {racket_score:.3f}")
        print(f"   甜区坐标: ({kpts_dict['sweet_spot']['x']:.1f}, {kpts_dict['sweet_spot']['y']:.1f})")
        print(f"   握柄坐标: ({kpts_dict['grip_center']['x']:.1f}, {kpts_dict['grip_center']['y']:.1f})")
```

---

### 4.2 方案 B：使用 PyTorch / Ultralytics 运行（离线/调试模式）

如果您在开发环境使用 PyTorch（支持 `mps`、`cuda` 或 `cpu`）：

```python
from ultralytics import YOLO
import cv2

# 加载黄金模型权重
model = YOLO("runs/pose/experiments/exp-007-racket-pose-yolo26m-v2.0-gold/weights/best.pt")

# 推理
results = model.predict("frame.jpg", imgsz=960, conf=0.25, device="mps")[0]

if results.keypoints is not None and len(results.keypoints.data) > 0:
    for box, kpts in zip(results.boxes, results.keypoints.data):
        score = float(box.conf[0])
        kpts_np = kpts.cpu().numpy()  # shape: (7, 3)
        print(f"球拍置信度: {score:.3f}")
        print("7关键点矩阵 [x, y, conf]:\n", kpts_np)
```

---

## 5. 关键业务场景调用指南

### 5.1 场景一：手腕与拍柄的动态贴合 (Wrist-to-Grip Association)
如何判断这把球拍是球员右手持拍还是左手持拍？
```python
def associate_holding_hand(racket_kpts, right_wrist_2d, left_wrist_2d, max_dist_px=140.0):
    # 握柄中点 = (butt_cap + grip_center) / 2
    handle_mid = (racket_kpts[0][:2] + racket_kpts[1][:2]) / 2.0

    d_right = np.linalg.norm(handle_mid - right_wrist_2d) if right_wrist_2d is not None else 9999
    d_left = np.linalg.norm(handle_mid - left_wrist_2d) if left_wrist_2d is not None else 9999

    if min(d_right, d_left) > max_dist_px:
        return "unattached", min(d_right, d_left)
    return ("right", d_right) if d_right < d_left else ("left", d_left)
```

### 5.2 场景二：触球瞬间与甜区验证 (Sweet Spot Contact Evidence)
判断网球是否击中有效拍面甜区：
```python
def check_sweet_spot_contact(ball_x, ball_y, racket_kpts, contact_radius_px=35.0):
    # 关键点 3 是甜区几何中心
    sweet_spot = racket_kpts[3][:2]
    distance = np.hypot(ball_x - sweet_spot[0], ball_y - sweet_spot[1])
    is_sweet_spot_hit = (distance <= contact_radius_px)
    return is_sweet_spot_hit, distance
```

---

## 6. 常见注意事项与排错清单

1. **切勿使用人体姿态估计器读取球拍模型**：
   人体姿态模型期望输出 $17 \times 3 + 6 = 57$ 个数值。本模型是球拍模型，输出包含 $7 \times 3 + 5 = 26$ 个数值。如果使用人体模型解析器读取，会因关键点数目不等于 17 而丢弃检测。
2. **高速挥拍模糊帧阈值设置**：
   在球拍极速划过画面时（如正手击球瞬间），球拍会出现明显拖影。建议生产环境将 `confidence_threshold` 设定在 `0.20 ~ 0.25` 之间，以保证 99% 以上的高速召回率。
3. **输入分辨率建议**：
   推荐固定使用 `imgsz=960`，以充分捕获拍框细线与手腕紧密接触面的纹理特征；仅在 60~120 FPS 极速高帧率摄像机场景下选用 `640` 版本。
