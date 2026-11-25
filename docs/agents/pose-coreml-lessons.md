# Core ML 与双视角姿态管线工程心得（Apple M5 / macOS 26）

本文件沉淀自 2026-11-22 在本机 **Apple M5 (32GB) / macOS 26.7.1 / coremltools 9.0** 上的真机实测与 2250 帧 4K 实片（`10-30-xue`）回归验证结论，供后续迭代直接复用、避免重复踩坑。

---

## 1. Core ML 计算单元铁律（严禁切 GPU / CPU 跑 YOLO26）

- **禁切 `CPU_AND_GPU`**：
  - `specVersion <= 9`（`iOS15` ~ `iOS18` / `macOS15`）的 YOLO26 `.mlpackage` 在 `CPU_AND_GPU` 下会直接触发 `MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR pass manager failed'`（C++ `SIGABRT`，Python 无法捕获）。
  - 即便升级 `minimum_deployment_target=ct.target.iOS26`（`specVersion=10`）避开崩溃，实测 `CPU_AND_GPU` 单次仍高达 **`41.97 ms`**（因算子回退 CPU + 跨设备同步），比 ANE（`9.84 ms`）慢 **4.26×**。
- **禁切 `CPU_ONLY`**：实测 `yolo26m-pose @ 640` 需 **`48.89 ms`**（慢 5.1×），会成为 `barrier` 管线木桶短板。
- **最佳实践**：
  - 全线固定 **`Det(ANE) + Pose(ALL)`**。实测 3 模型并发（`yolo26n-960` 4.98ms + 双路 `yolo26m-pose-640` 9.59ms×2）在 ANE 上顺序排队合计仅 **`24.18 ms`（41.4 FPS，p95 24.48 ms）**，CPU 专职负责 4K 切片、翻转、letterbox、生物力学与编码。
  - 后续从 `.pt` 重新导出 Core ML 模型时，建议顺手指定 `minimum_deployment_target=ct.target.iOS26` 以保持最新 `specVersion=10` 兼容性。

---

## 2. 4K 双视角姿态预处理规范（先修缩放漏洞，再谈升分辨率）

- **单次源像素直裁（`pose_resize_policy: direct_source`）**：跳过 `4K → 540×720 → 640×640` 二次有损缩放，保持规范视图坐标系 `(540, 720)` 不变。
- **等比灰边填充（`pose_preprocess_mode: letterbox`）**：与统一检测器保持一致（114 灰边），消除正面 3:4 与背面镜面的非等比拉伸畸变。
- **亚像素浮点坐标**：`PoseEstimatorYOLO26.get_keypoints(..., include_confidence=True)` 测量主链路必须保留 `float` 坐标，禁止 `int()` 截断，消除 `540×720 → 4K` 逆映射时的 2–3 px 台阶抖动。
- **实测收益（2250 帧对照）**：触球肘伸展 (`arm_extension`) IQR 收敛 **37.4%**（`69.46° → 43.46°`），正手多检出 2 拍，`Unknown` 减半，整帧 `stage_b_ms` p95 反降 `2.34 ms`（`39.77 → 37.43 ms`）。

---

## 3. 背面镜面姿态双守卫（防 X 型锁死 + 前景遮挡过滤）

- **教训**：仅靠 `_filter_backview_temporal_swap` 与上一帧缓存 `_last_valid_back_pose` 对比存在致命风险——一旦单帧误触发局部翻腿，脏状态会污染缓存，导致后续上百帧（实测最长连续 222 帧）即使模型预测正确也被滤波器强行掰回 X 型打结。
- **强制护栏**（位于 `dual_pose_estimator.py` `_predict_single_view` 背面分支，写入缓存前必须执行）：
  1. `_enforce_backview_anatomical_consistency`：以双肩横轴为主锚，当双髋、双膝或双踝明显张开却与躯干横轴反号时当帧解开，实测将背面 X 型交叉从 `24.32%` 降至 **`0.00%`**。
  2. `_suppress_backview_foreground_occluded_ankles`：利用下半部前景闯入者眼部打码框 `_last_back_eyes` 向下外推头肩遮挡区，剔除被误吸附到前景人肩膀的镜中脚踝，实测背面可见脚踝平均置信度提升至 **`0.959`**。

---

## 4. 姿态模型选型与分辨率演进备忘

- **不替换 YOLO26-Pose 为 MediaPipe Pose**：MediaPipe 为单人 CPU(XNNPACK) 跟踪架构，无法适配镜面多人过滤与隐私打码，高速挥拍易滞后/脑补；若未来需要 26 点（含脚尖/脚跟），优先评估 Top-Down `RTMPose-m (Body26)`。
- **2 合 1 拼图单次推理已上线（`c582584`）**：`configs/yolo26_tennis_config.yaml` 默认开启 `pose_stitched_single_pass: true`，将正/背双视角拼为单帧并复用零拷贝缓冲完成单次姿态推理（600 帧实测 **`16.54 ms` / 60.5 FPS**），可随时设 `false` 回退双实例并发推理；坐标映射与亚像素回归由 `test_stitched_dual_pose.py` 守护。
- **分辨率现状**：7 点球拍与统一检测器均基于 `960×960` 训练部署；若后续升级人体姿态至 `960×960`，延续上述 **2 合 1 拼图单次推理**（预计整帧 ~30 ms / 33 FPS），避免双实例独立 960（~48 ms）。
