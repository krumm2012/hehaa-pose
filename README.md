# Tennis Analyzer

`tennis_analyzer` 是一个面向网球训练视频的本地分析项目，重点能力是从单机位视频中提取人体姿态、网球、球拍、挥拍事件和面向 AI 网球教练的数据。当前主线分支是 `new`，已同步到远端 `origin-paused/new`。

## 当前能力

- `main_pipe.py`：多进程视频检测流水线，生成标注视频、逐帧 JSON 和诊断 JSON。
- `yolo26n_unified_detector.py`：统一检测人、球、球拍，包含静止球抑制、镜中球过滤、轨迹连续性和球拍候选重排序。
- `pose_estimator_yolo26.py`：YOLO26 pose 检测，支持 Core ML / ANE，并加入 pose 时序平滑以减少骨骼节点跳动。
- `swing_event_analyzer.py`：把逐帧检测结果聚合成事件级挥拍，输出 `start/contact/peak/end/stroke_type/confidence/quality_flags`。
- `swing_coach_data_collector.py`：生成面向 AI 网球教练的 `*_coach_dataset.json`，包含动作阶段、关键角度、球/拍质量、问题标签和解释证据。
- `swing_event_video_renderer.py`：生成统一 OSD 的 swing annotated 视频。
- `swing_report_builder.py`：生成本地 HTML 报告，展示视频、事件卡片、质量警告、JSON 摘要和人工标注控件。
- `swing_evaluation.py`：对比模型事件和人工标注，输出挥拍类型准确率、contact frame 准确率、误检/漏检/复核事件。

## 推荐工作流

### 1. 跑主检测流水线

```bash
python3 main_pipe.py \
  --config configs/yolo26_tennis_config.yaml \
  --input data/players-video/03.15.mp4 \
  --output data/players-video/results_YYYYMMDD/03.15_closed_loop.mp4
```

主流程会生成：

- `*_closed_loop.mp4`
- `*_closed_loop.json`
- `*_closed_loop_diagnostics.json`

### 2. 生成 swing 事件

```bash
python3 swing_event_analyzer.py data/players-video/results_YYYYMMDD/03.15_closed_loop.json
```

输出：

- `*_swing_events.json`
- `*_swing_events.csv`
- `*_swing_frames.csv`

### 3. 生成 AI 教练数据

```bash
python3 swing_coach_data_collector.py data/players-video/results_YYYYMMDD/03.15_closed_loop.json
```

输出：

- `*_coach_dataset.json`

### 4. 生成 swing 标注视频

```bash
python3 swing_event_video_renderer.py data/players-video/results_YYYYMMDD/03.15_closed_loop.json
```

输出：

- `*_swing_annotated.mp4`
- `*_swing_overlay_records.csv`

### 5. 生成可视化报告

```bash
python3 swing_report_builder.py data/players-video/results_YYYYMMDD/03.15_closed_loop.json
```

输出：

- `*_swing_report.html`

报告页可以直接在浏览器打开，支持人工修正事件类型、是否有效击球、是否需要复核，并下载 `swing_manual_annotations.json`。

### 6. 人工标注后做准确率评估

```bash
python3 swing_evaluation.py \
  --events data/players-video/results_YYYYMMDD/03.15_closed_loop_swing_events.json \
  --annotations /path/to/swing_manual_annotations.json
```

输出：

- `*_swing_evaluation.json`

评估 JSON 会记录：

- `stroke_type_accuracy`
- `contact_accuracy`
- `contact_mean_abs_error_frames`
- `manual_review_event_ids`
- `model_review_event_ids`
- `false_positive_event_ids`
- `unmatched_model_event_ids`
- `unmatched_annotation_event_ids`

如果同目录存在 `*_swing_evaluation.json`，`swing_report_builder.py` 会在报告页显示 `Evaluation Summary`。

## 已验证样例

当前回归样例位于：

```text
data/players-video/results_20260517/
```

重点产物：

- `03.15_closed_loop_swing_report.html`
- `03.15_closed_loop_swing_evaluation.json`
- `18.12_closed_loop_swing_report.html`

已知 `03.15` 的人工标注评估结果：

- 模型事件数：2
- 人工有效事件数：2
- 计数差异：0
- 挥拍类型准确率：50%
- contact frame 准确率：100%
- 模型建议复核事件：1、2

这说明当前优先优化方向是 `stroke_type` 分类，而不是事件计数。

## Gemini / AI 教练评估

给 Gemini 或其他大模型做专业教练评价时，建议同时提供：

1. `*_swing_annotated.mp4`
2. `*_swing_report.html`
3. `*_swing_events.json`
4. `*_coach_dataset.json`
5. 可选：`swing_manual_annotations.json`
6. 可选：`*_swing_evaluation.json`

使用指南见：

- [docs/GEMINI_COACH_REVIEW_GUIDE.md](docs/GEMINI_COACH_REVIEW_GUIDE.md)
- [docs/SWING_ACCURACY_CLOSED_LOOP.md](docs/SWING_ACCURACY_CLOSED_LOOP.md)
- [docs/SWING_GPT55_REVIEW.md](docs/SWING_GPT55_REVIEW.md)

## 配置重点

主要配置文件：

```text
configs/yolo26_tennis_config.yaml
```

当前关键策略：

- 检测模型：`yolo26n` 统一检测。
- pose 模型：`yolo26m-pose`。
- 计算单元：优先 `ANE`。
- pose 阈值：`pose_confidence_threshold: 0.4`。
- 静止球：启用静止球时序抑制和 hard mask。
- 轨迹：启用球连续性、速度预测和镜中球弱惩罚。

## 测试

推荐先跑 swing 闭环相关测试：

```bash
python3 -m unittest -v \
  test_swing_motion_pipeline.py \
  test_swing_presence_detector.py \
  test_ball_candidate_selector.py \
  test_static_ball_filter.py \
  test_pose_temporal_smoothing.py \
  test_swing_evaluation.py
```

当前最近一次验证：41 个相关测试通过。

## Git 状态说明

- 当前工作分支：`new`
- 当前远端：`origin-paused`
- `new` 与 `origin-paused/new` 已同步。
- `origin-paused/main` 有一个独立清理提交；它会删除/移动大量调试和历史文件，暂未直接合并到 `new`，避免破坏当前 swing 分析闭环。

## 已知限制

- 缺少大规模人工真值标注集，准确率仍需要持续用人工标注闭环校准。
- 单摄像头无法可靠估计 3D 旋转、真实拍面角度、球速、旋转和落点深度。
- 镜像机位的左右手规则仍带有当前场景假设，跨场地泛化需要 camera profile。
- 高速挥拍、遮挡、强反光、球拍/球漏检仍会影响 contact frame 和动作质量判断。

## 下一步建议

1. 给 `18.12.mp4` 也补人工标注，生成 `18.12_closed_loop_swing_evaluation.json`。
2. 基于多条人工标注结果优化 `swing_event_classifier.py`。
3. 增加 `camera_profile`，显式记录镜像/正视/侧视和左右手映射。
4. 把报告页升级为教练工作台，支持导入 Gemini 反馈和训练建议归档。
5. 单独清理 `.gitignore` 和历史缓存产物，减少后续 Git 噪声。

最后更新：2026-05-24
