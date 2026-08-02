# 🎾 Tennis Analyzer — 项目总结报告

> 生成时间：2026-08-01
> 分析对象：`/Users/krum5539/Documents/tennis_analyzer`（152 个 Python 文件，55 个测试文件）

---

## 1. 项目定位

**Tennis Analyzer** 是一个面向网球训练视频的**本地 AI 视频分析系统**，核心目标是从**单机位视频**中自动提取：

- 人体姿态（关键点）
- 网球（位置、轨迹）
- 球拍（位置、运动）
- 挥拍事件（start / contact / peak / end）
- 面向 AI 网球教练的结构化数据（动作阶段、关键角度、问题标签、纠错建议）

项目强调**本地运行**（macOS + Core ML / Apple Neural Engine 加速）、**可审计**（逐帧 JSON + 事件 JSON + 证据包）和**闭环校准**（模型输出 ↔ 人工标注对比评估）。

当前主线分支为 `new`，已同步远端 `origin-paused/new`。

---

## 2. 技术栈

| 层 | 技术 |
|---|---|
| 语言 | Python 3.9–3.13（多套 venv 并存） |
| 深度学习 | PyTorch + Ultralytics（YOLOv8 → **YOLO26 系列**） |
| 视觉处理 | OpenCV（HSV / Hough 圆检测，仅作历史/兜底） |
| 模型部署 | **Core ML（.mlpackage）+ ANE 神经引擎**加速 |
| 姿态估计 | yolo26m-pose（含时序 EMA 平滑） |
| 统一检测 | yolo26n（人/球/拍，替代原 HSV + YOLOv8n 双模型） |
| 球追踪（历史） | TrackNet v2（曾尝试，后转为 YOLO26n 统一检测） |
| 实时架构 | 多进程 + `shared_memory` + 线程池 + 队列反压 |
| LLM 增强 | DeepSeek Chat Completion（异步旁路教练，非思考模式） |
| 本地 Web | Python `http.server` 控制台（127.0.0.1:8765） |
| 配置 | YAML（`configs/yolo26_tennis_config.yaml` 为主） |
| 测试 | `unittest`（41+ 个相关测试通过，共 55 个测试文件） |

---

## 3. 系统架构（六层）

```
输入层 ──▶ 预处理层 ──▶ 检测层 ──▶ 处理层 ──▶ 分析层 ──▶ 可视化/输出层
视频/RTSP   ROI裁剪      姿态/球/拍   静态球过滤   挥拍分析     标注视频/JSON/HTML
          帧缓存        YOLO26      轨迹追踪     精彩瞬间     报告/教练数据集
```

关键设计点：

- **ROI（感兴趣区域）**：`roi_stream_config.py` + `calibrate_roi.py` 支持三路摄像头（Court 01–03）按脱敏 RTSP 地址绑定 ROI；球/拍检测只处理 ROI 外接矩形（`crop_margin=12`），姿态仍全帧，检测结果恢复为 2560×1440 原图坐标。
- **单上下文仓库**：`CONTEXT.md` 定义领域术语（主球、静止锚点、轨迹支持、重捕获窗口 8 帧），`docs/agents/` 提供工程约束。

---

## 4. 核心模块职责

### 4.1 主流水线

| 模块 | 职责 |
|---|---|
| `main_pipe.py`（1572 行） | 多进程检测流水线，满负载同步版；支持本地视频与 RTSP 直播（`--live-mode`、`--drop-stale-frames`）、HDMI 输出、逐帧 JSONL、事件 JSON/HTML 原子更新 |
| `yolo26n_unified_detector.py`（725 行） | 统一检测球+拍：静止球时序抑制与 hard mask、镜中球过滤、轨迹连续性/速度预测重排序、球拍候选重排序、8 帧重捕获窗口 |
| `pose_estimator_yolo26.py`（562 行） | YOLO26 pose（Core ML/ANE），EMA 平滑 + 跳变限幅 + 短时保持，减少骨骼节点跳动 |
| `realtime_swing_pipeline.py` | 实时/离线共用的增量挥拍事件分析（`analyze_frame_records()`），避免两套算法 |

### 4.2 Swing 分析闭环（离线）

```
main_pipe ─▶ *_closed_loop.json ─┬─▶ swing_event_analyzer ─▶ *_swing_events.json/csv
(逐帧检测)                       ├─▶ swing_coach_data_collector ─▶ *_coach_dataset.json
                                 ├─▶ swing_event_video_renderer ─▶ *_swing_annotated.mp4
                                 ├─▶ swing_report_builder ─▶ *_swing_report.html（人工标注控件）
                                 └─▶ swing_evaluation（模型 vs 人工标注，准确率评估）
```

### 4.3 实时 AI 教练

- **`local_realtime_coach.py`**：确定性规则教练，挥拍确认后输出 1–3 条 ≤15 字中文纠错，带独立 `confidence`。
- **`deepseek_realtime_coach.py`**：DeepSeek 异步旁路增强，`pending → ready/failed/unavailable`，超时/缺 Key 不影响本地分析；只上传结构化证据不上传视频画面。
- **`coach_evidence_policy.py`**：按证据域授权建议——姿态数据可靠时即使球/拍漏检仍给技术建议；缺证据的领域（拍面、旋转、球速、落点）只允许拍摄改善或复核建议。
- **`swing_evidence_builder.py`**：按 `event_id` 合并逐帧 JSON + 事件 JSON + Coach 数据 → `SwingEvidencePacket`，供 LLM 消费。
- **`swing_biomechanics.py`**：肩髋分离、肩部转动、手臂伸展、重心转移、平衡漂移等指标（按肩宽/髋宽归一化，2D 图像平面估计）。

### 4.4 配套工具

| 模块 | 职责 |
|---|---|
| `local_control_panel.py` | 本地 Web 控制台：场地选择、ROI 预览、参数调整、管道启停（凭据不落浏览器/命令行） |
| `calibrate_roi.py` | 鼠标交互式 ROI 四角点标定，自动归一化 P1–P4 |
| `convert_yolo26_to_coreml.py` 等 | YOLO26/TrackNet → Core ML 转换工具链 |
| `analysis_data_contracts.py` | 会话元数据、数据契约、逐帧记录盖章 |

---

## 5. 项目演进历史（git log）

| 时间 | 里程碑 |
|---|---|
| 早期 | YOLOv8-pose + HSV/Hough 球检测 + TrackNet 追踪尝试（`main.py` 853 行时代） |
| 2026-01 | **YOLO26 系列集成**：yolo26n 统一检测、yolo26m-pose、Core ML 转换、性能优化 |
| 2026-02 | main_pipe 多进程流水线、挥拍指标（tag: clawdon） |
| 2026-05 | **可审计挥拍事件识别流水线**（25108a3）→ 闭环评估 → 清理调试产物 |
| 2026-07 | **实时挥拍教练**：交互式事件时间轴 → 实时流水线优化 → 证据感知教练（7f5644d）→ 多场地实时教练（7912040） |
| 2026-07-29 | 实时分析流水线版本化隔离（c645ec5，HEAD） |
| 2026-08-01 | 自定义码流 UI（design-qa.md 通过，无 P0/P1/P2） |

---

## 6. 验证与质量现状

- 测试：41+ 个 swing 闭环相关测试通过（README 记录），共 55 个测试文件。
- 回归样例：`data/players-video/results_20260517/`（03.15 / 18.12）。
- 03.15 人工标注评估：模型事件数 2 vs 人工 2（计数一致）；**挥拍类型准确率 50%**（优先优化方向）；contact frame 准确率 100%。
- 07.20 Gemini 对比：多版本报告对比（3.5 Flash / 3.6 Flash / 新提示词 / 视频帧优化），校验方向校准、FPS 处理、触球帧一致性。

---

## 7. 已知限制

1. **缺少大规模人工真值标注集**，准确率依赖持续人工标注闭环校准。
2. **单摄像头 2D 局限**：无法可靠估计 3D 旋转、真实拍面角度、球速、旋转、落点深度。
3. **镜像机位左右手规则**带当前场景假设，跨场地泛化需 `camera_profile`。
4. 高速挥拍、遮挡、强反光、球/拍漏检仍影响 contact frame 与动作质量判断。
5. 仓库存在大量历史调试文件/多套 venv（venv、venv_311、venv_tracknet、venv_yolo26），Git 噪声较大（`origin-paused/main` 有清理提交未合并）。

---

## 8. 下一步建议

1. 给 `18.12.mp4` 补人工标注，生成评估 JSON，扩大校准样本。
2. 基于多条人工标注优化 `swing_event_classifier.py`（当前 50% 类型准确率是最大短板）。
3. 引入 `camera_profile`：显式记录镜像/正视/侧视与左右手映射，提升跨场地泛化。
4. 将 HTML 报告页升级为**教练工作台**：支持导入 Gemini 反馈、训练建议归档。
5. 清理 `.gitignore` 与历史缓存产物（含多套 venv 策略），降低维护与 Git 噪声。
6. 评估是否将多套 venv 收敛为单一受控环境，减少环境漂移风险。

---

## 9. 一句话总结

> 一个从「YOLOv8 + 传统视觉」演进到「YOLO26 + Core ML/ANE + 实时证据感知 AI 教练」的本地网球视频分析系统：已打通「逐帧检测 → 挥拍事件 → 教练数据集 → 人工标注闭环评估」的完整链路，并支持三场地 RTSP 实时流水线与 DeepSeek 旁路增强；当前瓶颈是挥拍类型分类准确率与标注数据规模。
