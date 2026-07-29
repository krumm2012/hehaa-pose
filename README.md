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

### 实时码流：单一主入口

实时挥拍闭环统一由 `main_pipe.py` 驱动。每个完整挥拍在安静窗口确认后，
立即原子更新事件 JSON 和 HTML，并在后台编码独立挥拍片段；片段编码不会进入
检测/姿态推理关键路径。

`--realtime-swing-events` 同时支持直播地址与按时间线播放的本地视频。HTTP/RTSP
等直播源建议配合 `--live-mode` 使用，以关闭逐帧落盘并优先处理最新帧。

DeepSeek 旁路需要先在当前终端配置密钥，不要把密钥写进 YAML 或命令行：

```bash
export DEEPSEEK_API_KEY='你的DeepSeek API Key'
```

```bash
venv_yolo26/bin/python main_pipe.py \
  --config configs/yolo26_tennis_config.yaml \
  --input http://127.0.0.1:1234/ \
  --output data/analysis_results/live_session.mp4 \
  --live-mode \
  --drop-stale-frames \
  --no-save-video \
  --realtime-swing-events \
  --realtime-swing-json data/analysis_results/live_session_swing_events.json \
  --realtime-swing-html data/analysis_results/live_session_swing_report.html \
  --realtime-swing-clips-dir data/analysis_results/live_session_swing_clips \
  --realtime-frame-output \
  --realtime-frame-jsonl data/analysis_results/live_session_frames.jsonl \
  --realtime-frame-snapshot-json data/analysis_results/live_session_frames_latest.json \
  --realtime-frame-snapshot-size 200 \
  --realtime-frame-flush-interval 5 \
  --realtime-coach \
  --realtime-coach-max-chars 15 \
  --realtime-coach-max-suggestions 3 \
  --realtime-coach-min-confidence 0.45 \
  --deepseek-coach \
  --deepseek-model deepseek-v4-flash \
  --deepseek-api-key-env DEEPSEEK_API_KEY \
  --deepseek-timeout-seconds 3 \
  --deepseek-workers 2 \
  --deepseek-coach-max-chars 15 \
  --realtime-analysis-interval 5 \
  --realtime-settle-frames 15 \
  --realtime-window-frames 200 \
  --realtime-clip-workers 1 \
  --realtime-open-report \
  --output-fps 25 \
  --inference-workers 2
```

上例用 `--no-save-video` 省去整场录像编码，但仍会生成每次挥拍片段；如需同时
保存完整标注视频，移除该参数即可。

实时输出：

- `live_session_swing_events.json`：已确认的事件快照。
- `live_session_swing_report.html`：自动刷新事件页面，顶部显示带 ROI 标识的实时码流截图，播放视频时暂停刷新。
- `live_session_swing_report_roi_preview.jpg`：每秒更新的脱敏码流截图，标注 ROI 边界与 P1–P4。
- `live_session_swing_clips/`：每个挥拍的独立 OSD MP4。
- `live_session_frames.jsonl`：每个完成推理帧一行，适合实时追加和故障恢复。
- `live_session_frames_latest.json`：最近 200 个处理帧的原子 JSON 快照。

逐帧日志默认保证不静默丢弃已完成推理的记录；正常写盘完全在后台进行。若磁盘
持续严重阻塞并耗尽内部队列，流水线会短暂反压以优先保证 JSONL 完整性。

`--realtime-coach` 使用本地确定性规则，在挥拍确认后立即把 1–3 条中文动作纠错写入
事件 JSON、终端和 HTML；不调用网络模型，每条指导硬限制为最多 15 个字符，并带
独立的 `confidence`。`coach_advices` 保存完整建议列表，原有 `coach_advice` 继续
保存第一条，保持已有消费者兼容。

实时事件会聚合肩髋分离、肩部转动、触球时手臂伸展、击球点相对身体横向距离、
准备到触球的重心转移，以及触球后平衡漂移。空间距离按事件内可见肩宽/髋宽归一化，
HTML 同时展示指标值与指标置信度。它们属于单摄像头图像平面 2D 估计，不等同于
多机位或传感器得到的 3D 关节动力学。可用
`--realtime-coach-max-suggestions 1|2|3` 控制建议数上限，
`--realtime-coach-min-confidence 0.45` 过滤不可靠指标；具体阈值位于
`configs/yolo26_tennis_config.yaml` 的 `realtime_swing.coach_biomechanics`。

数据质量不足时优先提示机位、入镜或遮挡问题，质量合格后才给动作建议。
高速球允许间歇漏检：全事件球检测覆盖率达到 20%，且触球帧前后 4 帧内至少
检测到 2 帧球时，不触发“确保来球完整入镜”。只有全事件或触球关键窗口的
球证据严重不足时，才把 `ball_track_gaps` 作为可行动警告。

`--deepseek-coach` 使用 `deepseek-v4-flash` 做异步旁路增强。本地建议始终先返回；
DeepSeek 状态以 `pending → ready/failed/unavailable` 更新到事件 JSON 和 HTML，
超时或缺少密钥不会影响本地分析。模型使用非思考模式和 JSON 输出以降低延迟。
每次请求会同时发送事件摘要，以及 `start_frame` 到 `end_frame` 区间内按帧号排序的
紧凑逐帧证据；只上传结构化分析结果，不上传视频画面。完整原始帧 JSON 保留在
`SwingEvidencePacket` 中，DeepSeek 视图会去掉候选列表等冗余诊断，但不会跳过事件帧。
低质量事件通过本地证据门控只允许拍摄改善或复核建议，并禁止旋转、落点、拍面和
绝对速度等缺少可靠证据的结论。提示词要求返回建议类别、证据帧和置信度，并把中文
建议硬限制为最多15字。

Coach 门控按证据域授权，不再把局部识别警告升级为整次挥拍不可评价。姿态数据可靠时，
即使球拍存在间歇漏检或背景静态球被拒绝，仍可基于身体、准备、平衡、节奏和随挥数据
给出技术建议；对应警告只会封锁拍面、精确拍路、旋转、落点和精确球路等相关主题。
聚合数据缺少逐帧 phase trace 时，会回退使用事件 JSON 的 `phase_counts`，避免把准备
和随挥时长误算为零。模型输入包含按证据生成的 `advice_candidates`，有可靠技术候选时
不接受拍摄或泛化复核建议代替 Coach 评价。
接口格式参考 [DeepSeek Chat Completion 官方文档](https://api-docs.deepseek.com/api/create-chat-completion)。

## 本地 ROI / Pipeline 控制台

本地控制台可选择 Court 01–03、预览当前 ROI、调整实时分析参数，并安全启动或停止
`main_pipe.py`。服务只监听本机回环地址，RTSP 密码不会返回前端，也不会出现在
Pipeline 命令行中。

推荐先通过环境变量提供摄像头凭据：

```bash
export TENNIS_RTSP_USERNAME='admin'
export TENNIS_RTSP_PASSWORD='你的摄像头密码'

venv_yolo26/bin/python local_control_panel.py --open
```

然后访问 `http://127.0.0.1:8765/`。也可以不设置环境变量，直接在页面的用户名和
密码框中临时输入；页面不会把凭据写入浏览器本地预设。

页面提供：

- Court 码流选择、ROI 点位及实时截图预览；
- ROI 裁剪边距、边界/填充/P1–P4 显示开关；
- FPS、推理线程池、低延迟直播、完整视频和 HDMI 输出参数；
- Swing 事件间隔、结束等待、本地 Coach、建议条数与最低置信度；
- DeepSeek 旁路、逐帧 JSONL、启动/停止、状态、日志及 Swing 报告入口。

`run_swing_report.py` 仅保留为已完成录制的离线兼容适配器；实时模式不调用它。
离线与实时事件识别都复用 `swing_event_analyzer.analyze_frame_records()`，避免维护两套挥拍算法。

### RTSP 与 ROI 绑定

主配置通过 `roi_settings` 开启 ROI。球和球拍模型只处理 ROI 外接矩形（含
`crop_margin`），姿态模型仍处理全帧；检测结果在候选筛选前恢复为原图坐标，因此
静态球屏蔽区、轨迹连续性、逐帧 JSON 和 Swing 事件继续使用 2560×1440 坐标系。

```yaml
roi_settings:
  enabled: true
  interactive_selection: false
  auto_load_config: true
  roi_config_path: "configs/roi_config.yaml"
  crop_margin: 12
  preview_interval_frames: 25
```

`configs/roi_config.yaml` 用脱敏地址绑定三路摄像机，不保存 RTSP 用户名或密码。
`main_pipe.py` 会根据当前输入地址自动选择对应场地：

```yaml
streams:
  - stream_id: "court01-main"
    stream_source: "rtsp://192.168.1.191:554/h264/ch1/main/av_stream"
    roi_enabled: true
    frame_size: [2560, 1440]
    roi_points: [[850, 130], [1700, 140], [1950, 1320], [580, 1320]]
  - stream_id: "court02-main"
    stream_source: "rtsp://192.168.1.192:554/h264/ch1/main/av_stream"
    roi_enabled: true
    frame_size: [2560, 1440]
    roi_points: [[1112, 116], [1875, 119], [2326, 1360], [850, 1351]]
  - stream_id: "court03-main"
    stream_source: "rtsp://192.168.1.193:554/h264/ch1/main/av_stream"
    roi_enabled: true
    frame_size: [2560, 1440]
    roi_points: [[847, 67], [1943, 69], [2181, 1273], [677, 1249]]
```

运行时输入可以包含认证信息，匹配、日志、实时 JSON 和 HTML 只使用脱敏后的地址。
前端 ROI 截图随 `--realtime-swing-events` 自动生成，不需要新增命令行参数。

鼠标重新校准 Court 01 时，先停止正在运行的 `main_pipe.py`，然后执行：

```bash
export TENNIS_RTSP_URL='rtsp://用户名:密码@192.168.1.191:554/h264/ch1/main/av_stream'

venv_yolo26/bin/python calibrate_roi.py \
  --config configs/yolo26_tennis_config.yaml \
  --input "$TENNIS_RTSP_URL" \
  --roi-config configs/roi_config.yaml \
  --stream-id court01-main \
  --stream-label "Court 01 Main Camera"
```

用鼠标点击 ROI 的四个角点，点击顺序不限；脚本会自动归一化为
`P1 左上 → P2 右上 → P3 右下 → P4 左下`。选点窗口按一次 `c` 即保存，
`r` 重选，`q/ESC` 取消。如需保存前再弹出第二个确认窗口，可增加
`--confirm-preview`，此时预览窗口按 `s/c/Enter` 保存、`r` 返回重选。
脚本自动把缩放窗口坐标还原为源视频坐标，写入后重新读取复核，保存前备份原 YAML，并生成
`configs/roi_config_calibrated_preview.jpg`。保存后需重启 `main_pipe.py`。

### HDMI 全屏输出

`--hdmi-output` 将 `main_pipe.py` 标注后的单路画面全屏显示，并自动关闭 dual-view。
在 macOS 镜像显示模式下无需额外参数；按 `ESC` 或 `q` 可安全停止。扩展桌面模式可用
`--display-origin X Y` 把窗口移动到外接显示器左上角后再进入全屏。

### 聚合单次 Swing 证据 JSON

`swing_evidence_builder.py` 按 `event_id` 合并逐帧 JSON、Swing 事件 JSON 和 Coach
数据集，生成一个 `SwingEvidencePacket`。它会截取事件帧范围、去掉 Coach 数据中
重复的 `frame_trace`，并报告缺帧、重复帧和三条时间序列是否对齐。

```bash
venv_yolo26/bin/python swing_evidence_builder.py \
  data/analysis_results/16.10_active_ball.json \
  --event-json data/analysis_results/16.10_active_ball_swing_events.json \
  --coach-json data/analysis_results/16.10_active_ball_coach_dataset.json \
  --event-id 1 \
  --output-json data/analysis_results/16.10_active_ball_swing_1_evidence.json
```

省略三个可选路径时，默认使用同目录的 `<frame_stem>_swing_events.json`、
`<frame_stem>_coach_dataset.json` 和 `<frame_stem>_swing_<event_id>_evidence.json`。
还可用 `--player-context-json` 合并用户水平、持拍手和训练目标等本地配置。

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
