# Swing Event Recognition Change Log

## 2026-10-05 — 动力链时间敏感性与足部接地点复核

- v8动力链策略新增短源间隔敏感性检查，保留原始PTS与峰值候选；检查不稳定时暂停正式峰值，不用替代峰恢复结论。
- 独立视角追加身体段观测有效数、拒绝计数和源帧明细；报告解释背面边界峰、宽峰、短线及遮挡，OSD显示短间隔敏感状态。
- 对实际50.03的250帧重放；ffprobe与日志PTS吻合，第2拍峰值对102帧短间隔敏感；历史报告和原观测不回写。
- 生成66连续帧足底接地点辅助复核页，提供262个脚踝提示、腾空/不可辨认、全部复核已填写项、续标和导出；不自动生成接地点或技术分。
- 6项新增、694项全量通过；浏览器自动化导航超时，未宣称全交互现场验收。契约及后续见 [本轮说明](docs/KINEMATIC_CADENCE_GROUND_ITERATION_20261005.md)。

## 2026-10-05 — 球道 / 机位共享地面标定 JSON 导入

- 新增“应用 JSON 为机位标定”入口和受 token 保护的导入 API；注册独立不可变共享版本，不覆盖原 JSON、单视频修订或 ROI。
- 球道 2 实时源及明确绑定球道 2 的同尺寸视频使用共享标定；其他机位、未绑定或尺寸不符的输入不获得测量资格。
- 保留原标定与实际输入的不同来源身份，FrameRecord、session 与事件记录共享 profile 应用声明，不把机位复用解释成同一视频。
- 已应用用户的 `analyzer_ground_calibration (2).json`；尺寸 3.3×4.8 米，原图 2560×1440，镜中对应未确认标记保留。
- 10 项新增回归、32 项地面聚焦、688 项全量通过，实际 HTTP 注册及跨输入/机位隔离核验通过；未新增推理或评分规则。

## 2026-10-05 — 修复地面标定 JSON 导出

- 修复导出按钮的局部 `document` 变量覆盖浏览器 DOM 对象，导致 `document.createElement is not a function`；标定数据变量改名为 `calibration`。
- 新增执行实际导出回调的回归，校验下载文件名、JSON 格式、正背四角、来源绑定、尺寸、确认状态、URL 释放及无效草稿拒绝。
- 先保存用户当前选点和确认状态再刷新；实际浏览器下载的四角、尺寸、来源绑定与确认状态均与保存值一致。22 项地面回归、678 项全量通过。
- 验收见 [导出修复记录](validation/ground_export_fix_20261005.json)。

## 2026-10-05 — 独立地面标定与双视角投影参考

- 新增独立四角编辑器、当前输入来源绑定和不可变标定版本；不写入 ROI 配置。
- 球道 2 的 3.3×4.8 米按用户已实测保存；当前画面几何和镜中角点对应仍需核对。
- 分视角映射新鲜原图脚踝观测，保留源帧、模型分数及缺失原因；接地状态未知，不推导真实步幅、速度或技术评分。
- 双视角显示加入缓存地面网格，逐帧 OSD 与事件汇总贯穿控制台及两种报告。
- 21 项新增、64 项聚焦、677 项全量回归通过；原片预览、参考草稿保存和 250 帧原指标保持验收通过。
- 公式与限制见 [地面标定契约](docs/GROUND_REFERENCE.md)；验收见 [本轮记录](validation/ground_reference_acceptance_20261005.json)。

## 2026-10-05 — 源时间、观测资格与复核报告迭代

- 汇总双视角观测来源、动力链断段与峰宽、未标定球拍速度、指标公式和评分限制。
- 运动、身体统计、分类与触球候选改用合格源时间，人工锚点保持指定源帧。
- 评估、报告及编辑器导入先校验身份；合法零帧保留，显式空值不借旧值。
- 模型辅助复核与参考匹配不批准独立准确性；自动技术规则集合保持为空。
- 最新工程轮 656 项全量回归通过；独立标签、跨会话来源绑定、最新编辑器实际交互及 RTSP/TTS 现场验收仍有缺口。
- 完整提交、验收和剩余内容见 [最近修改汇总](docs/RECENT_CHANGES_20261005.md)；逐轮记录见 [优化验收历史](docs/OPTIMIZATION_ACCEPTANCE_20261004.md)。

## 2026-08-02 — Realtime overlap suppression and overlay re-recognition

- Aligned realtime peak duplicate suppression with the segmenter's 1.6-second minimum peak distance.
- Limited rolling analysis to frames after the latest immutable published event and defensively rejected intersecting event ranges.
- Added regression coverage for the observed `33–98` / `62–131` overlap while preserving a later non-overlapping Swing.
- Added opt-in recovery of analyzer-owned hollow ball markers and current/legacy racket rectangles for videos that are analyzed again.
- Kept overlay provenance in frame diagnostics and passed recovered candidates through existing ball/racket temporal selection.
- Added raw per-class model confidence and threshold diagnostics for both ball and racket calibration.
- Rejected small filled yellow balls, long ROI lines, and open pose strokes from overlay recovery.
- Verified 250 RTSP frames at approximately 25 FPS with four non-overlapping Swing ranges.

## 2026-08-02 — Session quality and drift dashboard

- Added the pure `build_session_quality_dashboard(events)` interface shared by offline analysis,
  realtime JSON, and both HTML reports.
- Separated visible-technique trend from capture/evidence quality so detection degradation is not
  presented as player regression.
- Added per-Swing trend series, recurring warning/advice counts, contact support, Coach latency,
  DeepSeek availability, and first-window versus recent-window indicators.
- Requires at least six Swing events before making a drift conclusion.
- Treats a 15% camera-scale change as a confounder and overlapping event ranges as an integrity
  blocker for technique drift.
- Verified the dashboard against 07.20, 16.10, and the local RTSP input at
  `rtsp://127.0.0.1:8554/input-video`.

## 2026-08-01 — Single-view biomechanics and Coach calibration

- Added one shared `single_view_visible_coach_v1` calibration policy for offline analysis,
  realtime local Coach, reports, overlays, and the DeepSeek sidecar.
- Added directly reviewable 2D metrics for shoulder-turn change and preparation knee flexion.
- Marked projected hip–shoulder separation, screen-space body translation, and translation-only
  balance as non-coaching proxies with explicit exclusion reasons.
- Stopped treating absent contact/racket evidence as a zero technique score.
- Split the 0–9 visible-technique score from event classification and capture quality, and added
  score uncertainty plus metric-level provenance.
- Gated preparation/follow-through advice on their own boundary/contact evidence confidence.
- Removed excluded numeric biomechanics and projected shoulder/hip angles from DeepSeek model
  evidence; the sidecar now receives only exclusion reasons and permitted advice candidates.
- Verified 07.20 as three Forehands with visible scores `6.69 / 3.80 / 5.72` (mean `5.40/9`),
  and preserved the three-Forehand result on 16.10.

Date: 2026-05-01

## Summary

This iteration redesigned swing counting from frame-label counting into an auditable event-level pipeline. The goal was to make swing recognition more precise, avoid counting follow-through/recovery as additional swings, and generate unified annotated videos plus frame-level records for review.

Final verified result on the current fixture `data/output_video.json` and `data/output_pipe.json`:

- Total swing events: 3
- Event types: 3 Forehand
- Important handedness rule: in this camera view, the image-left side corresponds to the player's real right-hand forehand side.

## Problem Background

The earlier recognition path was unstable for precise counting because it leaned too heavily on per-frame static labels such as `Forehand`, `Backhand`, and `Two-Handed Backhand`.

Observed issues:

- Follow-through and recovery frames were sometimes treated as new swing evidence.
- Static frame labels changed during one continuous action, causing one stroke to contain mixed labels.
- Two-hand proximity during recovery was incorrectly used as strong two-handed backhand evidence.
- The camera view flips the intuitive interpretation: image-left corresponds to the player's real right-hand forehand side.
- As a result, two events were previously misclassified as `Two-Handed Backhand`.

## Root Cause

The root cause was not only a threshold problem. The event classifier inherited noisy frame-level labels and then allowed post-impact/follow-through two-hand evidence to override the event type.

For the current video, event 1 and event 3 both had right-hand forehand evidence when interpreting image-left as the player's true right-hand side. However, later frames in each event showed hands moving close together and `Backhand`/`Two-Handed Backhand` frame labels, so the previous logic changed the event type to `Two-Handed Backhand`.

## New Architecture

The redesigned pipeline is split into four auditable layers.

1. Frame feature extraction

File: `swing_motion_features.py`

Extracts per-frame motion and geometry from the existing pipeline JSON. It does not run model inference.

Key fields:

- `wrist_speed`
- `wrist_accel`
- `racket_speed`
- `racket_accel`
- `ball_speed`
- `ball_racket_distance`
- `contact_score`
- `two_hand_distance`
- `two_hand_distance_body_width`
- `active_wrist_x_offset`
- `active_wrist_x_offset_body_width`
- `camera_facing_score`
- `arm_extension_deg`
- `shoulder_turn_deg`
- `hip_shoulder_sep_deg`

2. Event segmentation

File: `swing_event_segmenter.py`

Segments full swing events from the feature timeline.

Current behavior:

- Uses smoothed motion energy to find candidate swing peaks.
- Uses robust peak selection for long videos.
- Builds full event windows around peak frames.
- Refines `start_frame` from a stable quiet basin plus sustained motion/shoulder-turn onset instead of a fixed pre-peak offset.
- Emits `recovery_ready_transition` when recovery and preparation overlap without a trustworthy static ready frame.
- Makes phase labels contact-relative so `follow_through` cannot precede contact and `backswing` cannot follow it.
- Keeps fallback energy-island segmentation for short or synthetic inputs.
- Adds `peak_frame`, `phase_counts`, and per-frame `phase` traces.

3. Event classification

File: `swing_event_classifier.py`

Classifies an entire event instead of directly counting frame labels.

Current behavior:

- Treats frame labels as evidence, not ground truth.
- Uses a contact-centred window that excludes ready/recovery hand proximity.
- Records the configured player dominant hand instead of trying to infer identity from one swing.
- Infers whether the player faces the camera or faces away from anatomical shoulder projection.
- Maps the dominant-side wrist to forehand/backhand only after normalizing for that camera orientation.
- Normalizes wrist separation by shoulder width so two-hand evidence is stable across player distance and resolution.
- Requires both backhand-side and two-hand support before accepting `Two-Handed Backhand`.
- Emits auditable `evidence.classification_context` with player, camera, swing-side, and decision-rule evidence.

4. Offline analysis and video rendering

Files:

- `swing_event_analyzer.py`
- `swing_event_video_renderer.py`

Responsibilities:

- Read existing frame JSON output.
- Generate event-level JSON and CSV.
- Generate frame-level audit CSV.
- Render unified annotated videos from the original source video without rerunning inference.

## Changed Files

New files:

- `swing_motion_features.py`
- `swing_event_classifier.py`
- `swing_event_segmenter.py`
- `swing_event_analyzer.py`
- `swing_event_video_renderer.py`

Updated tests:

- `test_swing_motion_pipeline.py`

Generated outputs:

- `data/output_video_swing_events.json`
- `data/output_video_swing_events.csv`
- `data/output_video_swing_frames.csv`
- `data/output_video_swing_overlay_records.csv`
- `data/output_video_swing_annotated.mp4`
- `data/output_pipe_swing_events.json`
- `data/output_pipe_swing_events.csv`
- `data/output_pipe_swing_frames.csv`
- `data/output_pipe_swing_overlay_records.csv`
- `data/output_pipe_swing_annotated.mp4`

## Current Result

For both `output_video` and `output_pipe` frame JSON:

| Event | Frame Range | Peak Frame | Type | Notes |
| --- | ---: | ---: | --- | --- |
| 1 | 25-77 | 46 | Forehand | Corrected from previous false two-handed backhand |
| 2 | 97-149 | 118 | Forehand | Kept as forehand |
| 3 | 189-241 | 210 | Forehand | Corrected from previous false two-handed backhand |

Summary:

```text
swing_event_count = 3
swing_event_type_counts = {"Forehand": 3}
```

## Unified Video Overlay

The annotated videos now use one consistent event display format.

Overlay fields:

- `Event <id>/<total>: <type>`
- `Frame`
- `Phase`
- `Range`
- `Peak`
- `raw_label`
- `energy`
- `wrist`
- `racket`
- `2Hdist`
- `contact`
- `ball-racket`
- event progress bar
- peak frame border
- pose skeleton from saved JSON
- ball marker
- racket box

Output videos:

- `data/output_video_swing_annotated.mp4`
- `data/output_pipe_swing_annotated.mp4`

Both verified as:

```text
frames = 250
fps = 25.14
resolution = 2560x1440
```

## Commands

Run event analysis:

```bash
python3 swing_event_analyzer.py data/output_video.json
python3 swing_event_analyzer.py data/output_pipe.json
```

Render annotated videos:

```bash
python3 swing_event_video_renderer.py data/output_video.json
python3 swing_event_video_renderer.py data/output_pipe.json
```

Run regression tests:

```bash
python3 -m unittest -v test_swing_motion_pipeline.py test_swing_presence_detector.py test_detection_frame_context.py test_highlight_service.py
```

Syntax check:

```bash
python3 -m py_compile swing_event_segmenter.py swing_event_analyzer.py swing_event_video_renderer.py test_swing_motion_pipeline.py
```

## Verification

Latest verification:

```text
21 tests OK
```

Additional output validation:

```text
data/output_video_swing_annotated.mp4: 250 frames, 25.14 FPS, 2560x1440
data/output_pipe_swing_annotated.mp4: 250 frames, 25.14 FPS, 2560x1440
```

## Important Assumptions

This iteration encodes a camera-specific handedness rule:

```text
image-left = player's true right-hand forehand side
```

This rule is correct for the current analyzed video. For other camera angles, this should become a configurable setting instead of a hard-coded assumption.

## Known Limitations

- The current event type result is calibrated against the current fixture and camera orientation.
- The handedness rule should be moved into configuration before batch processing many videos with different camera setups.
- Ball-racket contact is still not always reliable because ball/racket detections can be missing around the peak frame.
- Current video rendering is an offline post-process. It does not yet integrate directly into `main_pipe.py` realtime output.

## Recommended Next Iterations

1. Add a config option for camera handedness mapping.

Example:

```yaml
swing_event_analysis:
  true_right_hand_screen_side: left
```

2. Integrate event analysis into `main_pipe.py` output mode after frame JSON generation.

3. Add more fixtures from different camera angles to avoid overfitting to this single video.

4. Improve contact confidence by combining ball trajectory, racket box continuity, and peak frame proximity.

5. Add a small event review HTML or CSV viewer to inspect `peak_frame`, `phase`, and feature evidence quickly.
