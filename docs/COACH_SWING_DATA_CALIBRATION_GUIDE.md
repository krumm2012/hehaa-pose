# Coach Swing Data Calibration Guide

本文档面向网球教练和视频复核人员，说明如何理解 `tennis_analyzer` 当前输出的挥拍动作数据，以及如何通过人工标注帮助系统持续校准。

目标不是让教练调代码，而是让教练知道：哪些结果可以直接参考，哪些结果需要人工确认，人工确认后技术团队应该优先调整哪些参数。

## 1. 系统能输出什么

一次视频分析后，通常会得到以下文件：

- `*_swing_annotated.mp4`：带挥拍事件、姿态骨架、球/拍位置和 OSD 信息的标注视频。
- `*_swing_report.html`：可视化报告页面，可看视频、事件卡片、质量提示，并做人工标注。
- `*_swing_events.json`：系统识别出的挥拍事件数据。
- `*_coach_dataset.json`：面向 AI 网球教练的数据集，包含动作阶段、身体角度、球/拍质量和问题标签。
- `swing_manual_annotations_v2.json`：教练或复核人员从报告页下载的人工时间轴真值。
- `*_swing_evaluation.json`：人工标注与系统识别的对比评估结果。

## 2. 教练看报告时先看什么

建议按这个顺序看：

1. 先看 `*_swing_annotated.mp4`，确认每次挥拍事件是否被正确圈出来。
2. 再看 `*_swing_report.html` 中的 Event 卡片。
3. 对每个 Event 判断：是否真实挥拍、挥拍类型是否正确、击球帧是否合理。
4. 如果系统给出质量警告，不要马上把它当成学生动作问题，要先判断是否是检测问题。
5. 完成整段时间轴复核后下载 `swing_manual_annotations_v2.json`。

## 3. 挥拍事件字段说明

### 3.1 事件边界

每个挥拍事件包含：

- `start_frame`：系统认为挥拍开始的帧。
- `contact_frame`：系统认为最接近击球的帧。
- `peak_frame`：动作能量最高的帧，通常接近快速挥拍或随挥开始。
- `end_frame`：系统认为挥拍事件结束的帧。

`evidence.start_boundary` 记录开始边界的可审计依据。`quiet_onset` 表示系统先找到
连续静稳区，再由持续动作能量或肩线转动确认新一拍开始；
`recovery_ready_transition` 表示连续击球时没有可信静态准备，边界只能标为
“恢复—准备过渡”，其 `confidence` 通常为 `low`，应优先人工复核。

教练复核重点：

- `start_frame` 是否过早包含准备站姿。
- `end_frame` 是否把恢复动作或下一次准备动作也算进来了。
- `contact_frame` 是否真的接近球拍触球瞬间。
- `peak_frame` 不一定等于击球帧，它更像动作能量峰值。
- 触球前不应出现 `follow_through`，触球后不应再次出现 `backswing`。

### 3.2 挥拍类型

当前主要类型：

- `Forehand`：正手。
- `Backhand`：单手反手或未明确双手的反手。
- `Two-Handed Backhand`：双手反手。
- `Unknown` / `Unclear`：证据不足。

重要说明：

事件中的 `evidence.classification_context` 会分别记录球员惯用手、机位朝向、挥拍侧和最终决策规则。机位不再固定假设“画面左边是真实右手”，而是通过左右肩的解剖关键点顺序判断球员面向或背向相机，再映射正反手。双手距离同时按肩宽归一化，减少远近和分辨率变化带来的误判。

- `player.dominant_hand`：配置指定的球员惯用手；单段视频不会擅自推断惯用手。
- `camera.view`：`facing_player`、`behind_player` 或 `side_or_uncertain`。
- `swing.side`：机位归一化后的 `forehand`、`backhand` 或不确定。
- `decision_rule`：最终采用的分类证据路径。

### 3.3 置信度

- `confidence` 表示系统对挥拍类型或事件判断的信心。
- 高置信度不代表一定正确，因为球/拍/pose 的检测质量也会影响结果。
- 如果 `quality_flags.review_recommended = true`，即使 `confidence` 较高，也建议人工复核。

## 4. 质量标记 quality_flags 怎么看

`quality_flags` 是最重要的复核依据。它告诉教练：这个事件是否应该相信。

常见字段：

- `pose_frame_ratio`：事件中有可用人体姿态的帧比例。
- `ball_frame_ratio`：事件中有可用网球位置的帧比例。
- `racket_frame_ratio`：事件中有可用球拍位置的帧比例。
- `warnings`：系统发现的风险标签。
- `review_recommended`：是否建议人工复核。

常见 warnings：

| Warning | 含义 | 教练应该怎么处理 |
| --- | --- | --- |
| `ball_track_gaps` | 网球追踪不连续 | 接触点、来球/出球方向要降低信任 |
| `racket_track_gaps` | 球拍追踪不连续 | 击球点、拍头速度、挥拍路径要人工确认 |
| `pose_gaps` | 姿态缺失或不稳定 | 身体角度、重心、站姿判断要降低信任 |
| `static_ball_mask_in_event` | 静止球过滤参与了判断 | 注意是否误过滤了运动球，或地上球干扰 |
| `mirror_ball_rejection_in_event` | 镜中球/干扰球被过滤 | 确认系统没有把镜子里的球当主球 |
| `camera_view_uncertain` | 肩部投影不足以确定机位朝向 | 分类退回帧标签证据，需人工确认挥拍类型 |
| `ball_continuity_disabled` | 球轨迹连续性被中断 | 网球轨迹可能跳变，需复核球位置 |
| `contact_frame_needs_review` | 击球帧不够可靠 | 必须人工确认 contact frame |

## 5. 人工标注怎么做

打开 `*_swing_report.html`，每个事件卡片都有人工标注控件。

建议最少标注这些字段：

- `人工开始帧 / 触球帧 / 结束帧`：拖动视频逐帧确认真实事件边界。
- `人工类型`：选择真实挥拍类型。
- `计数正确`：这次事件是否应该被计入一次挥拍。
- `有效击球`：是否是真实击球，而不是空挥、准备动作或误检。
- `需要复核`：教练认为还需要再次查看的事件。
- `问题标签`：选择最接近的问题，例如类型错、漏识别、多计、击球帧错、球轨错、姿态错。

如果视频中有系统没有生成事件卡片的挥拍，点击“新增漏检挥拍”并填写时间帧和类型。检查完整段视频、处理完所有“需要复核”标注后，再勾选“已完整检查整段视频”；否则 Precision、Recall 和 F1 会保持 provisional。

备注尽量简短，例如：

- `实际为正手，画面左侧是真实右手。`
- `随挥被多计一次。`
- `contact 应该晚 3 帧。`
- `地上静止球抢了追踪焦点。`

## 6. 评估结果怎么解释

生成 `*_swing_evaluation.json` 后，重点看这些指标：

| 指标 | 含义 | 判断建议 |
| --- | --- | --- |
| `predicted_event_count` | 系统识别的事件数 | 与人工有效事件数对比 |
| `manual_valid_event_count` | 人工确认的有效挥拍数 | 作为当前视频计数真值 |
| `event_count_delta` | 系统计数 - 人工计数 | 正数代表多计，负数代表漏计 |
| `precision` | 系统事件中匹配人工真值的比例 | 低说明多计较多 |
| `recall` | 人工真值中被系统匹配的比例 | 低说明漏检较多 |
| `f1` | Precision 与 Recall 的综合指标 | 适合比较不同参数版本 |
| `stroke_type_accuracy` | 挥拍类型准确率 | 低于 80% 应优先优化分类规则 |
| `contact_accuracy` | 击球帧在容忍范围内的比例 | 低于 80% 应优先优化球/拍追踪或 contact 规则 |
| `contact_mean_abs_error_frames` | 击球帧平均误差 | 25fps 下 3 帧约 0.12 秒 |
| `start/end_mean_abs_error_frames` | 事件边界平均误差 | 大说明挥拍开始或结束判断偏移 |
| `event_mean_iou` | 模型与人工事件时间范围重叠度 | 越接近 1 越好 |
| `manual_review_annotation_ids` | 人工要求复核的标注 | 优先查看这些事件 |
| `model_review_event_ids` | 系统认为低可信的事件 | 如果很多，说明检测质量需要优化 |
| `false_positive_event_ids` | 人工标为无效的系统事件 | 多说明系统把准备/随挥/恢复误算成挥拍 |
| `unmatched_model_event_ids` | 系统有、人工没有的事件 | 可能多计或标注漏掉 |
| `false_negative_annotation_ids` | 人工有、系统没有匹配到的事件 | 漏检真值 |

## 7. 参数校准指导

下面是教练反馈与技术参数之间的对应关系。教练不需要直接改参数，但可以用这些描述给技术团队反馈。

### 7.1 多计挥拍

表现：

- 一次挥拍被拆成两次。
- 随挥或恢复动作被算成新挥拍。
- `false_positive_event_ids` 增多。

优先检查参数：

- `min_event_gap`：两次挥拍之间最小间隔。增大可减少拆分。
- `min_event_frames`：一次事件最少持续帧数。增大可过滤短暂误检。
- `active_energy`：进入挥拍事件的动作能量门槛。增大可更保守。
- `min_peak_energy`：峰值能量门槛。增大可减少小动作误识别。

建议方向：

- 如果随挥被多计，优先增大 `min_event_gap`。
- 如果准备动作被计入，优先增大 `active_energy` 或 `min_peak_energy`。
- 如果很多短事件，优先增大 `min_event_frames`。

### 7.2 漏计挥拍

表现：

- 人工看到有挥拍，但系统没有事件。
- `false_negative_annotation_ids` 增多。

优先检查参数：

- `min_peak_energy`：过高会漏掉慢动作或轻挥。
- `active_energy`：过高会让事件起不来。
- `pose_confidence_threshold`：过高可能导致姿态缺失。
- `pose_smoothing_max_jump_px`：过低可能压制真实快速动作。

建议方向：

- 慢动作训练视频漏检时，适当降低 `min_peak_energy` 和 `active_energy`。
- 低清晰度视频漏检时，先查看 pose 是否稳定，再考虑降低 pose 阈值。

### 7.3 正反手类型错

表现：

- 正手被识别成双反。
- 双反被识别成正手。
- `stroke_type_accuracy` 偏低。

优先检查参数或规则：

- `two_hand_distance_px`：两手距离小于该值时，更容易判断为双手动作。
- `two_hand_min_ratio`：事件中满足双手距离的帧比例。
- `dominant_hand`：惯用手。
- `mirror_view` / 镜像左右手规则：画面左侧是否是真实右手。

建议方向：

- 如果正手经常被误判为双反：降低双手证据权重，或减小 `two_hand_distance_px` / 增大 `two_hand_min_ratio`。
- 如果双反经常被误判为正手：增大双手证据权重，或增大 `two_hand_distance_px` / 降低 `two_hand_min_ratio`。
- 如果换机位后正反手整体反了，先检查 `mirror_view` 和左右手映射，不要先调动作阈值。

### 7.4 击球帧不准

表现：

- `contact_frame` 明显早于或晚于真实击球。
- `contact_accuracy` 偏低。
- `contact_frame_needs_review` 经常出现。

优先检查参数：

- 球检测质量：`ball_frame_ratio`。
- 球拍检测质量：`racket_frame_ratio`。
- `ball_racket_proximity_weight`：球靠近球拍时的加权。
- `racket_ball_proximity_weight`：球拍靠近球时的加权。
- 静止球 hard mask 相关参数。

建议方向：

- 如果地上球或镜中球干扰 contact，优先优化静止球过滤和镜中球过滤。
- 如果球拍漏检，先提升球拍检测稳定性，不要直接改 contact 规则。
- 如果球和拍都稳定但 contact 偏早/偏晚，再调整 contact 选择逻辑。

### 7.5 地上静止球或镜中球抢焦点

表现：

- 标注视频中主球标到了地上静止球。
- 球被标到镜子里。
- 球在拍子附近凭空出现或跳变。

优先检查参数：

- `static_ball_suppression_enabled`
- `static_ball_hard_mask_enabled`
- `static_ball_hard_mask_min_seen_frames`
- `static_ball_hard_mask_radius_px`
- `static_ball_hard_mask_allow_near_racket`
- `ball_play_area_min_y_ratio_hard`
- `ball_max_motion_for_continuity_px`

建议方向：

- 地上静止球抢焦点：降低 `static_ball_hard_mask_min_seen_frames` 或增大 `static_ball_hard_mask_radius_px`。
- 球拍附近地上球被误放行：保持 `static_ball_hard_mask_allow_near_racket: false`。
- 镜中球被选中：提高有效击球区约束，检查 `ball_play_area_min_y_ratio_hard`。
- 主球轨迹跳到远处：降低 `ball_max_motion_for_continuity_px`。

### 7.6 骨骼节点跳动

表现：

- 手腕、肩、膝等关键点闪跳。
- 挥拍速度突然异常。
- 姿态线一帧跳到错误位置。

优先检查参数：

- `pose_confidence_threshold`
- `pose_keypoint_confidence`
- `pose_smoothing_alpha`
- `pose_smoothing_max_jump_px`
- `pose_smoothing_hold_missing_frames`
- `pose_smoothing_reset_frames`

建议方向：

- 节点抖动大：降低 `pose_smoothing_alpha`，或降低 `pose_smoothing_max_jump_px`。
- 快速挥拍被平滑过头：提高 `pose_smoothing_alpha`，或提高 `pose_smoothing_max_jump_px`。
- 短暂丢点：提高 `pose_smoothing_hold_missing_frames`。
- 长时间错跟：降低 `pose_smoothing_reset_frames`，让系统更快重置。

### 7.7 生物力学与 Coach 分数

当前策略版本为 `single_view_visible_coach_v1`。它只评价单机位画面中能够直接复核的
动作，不把检测完整度或挥拍分类置信度混入技术分数。

参与 `visible_technique_score_9` 的指标只有：

- `arm_extension`：可靠触球窗口内的持拍臂二维关节角；触球证据不足时退回动作峰值窗口，
  只评价“挥拍时舒展”，不声称是精确触球姿态；
- `shoulder_turn_change`：准备阶段相对早期基线的肩线投影变化；
- `preparation_knee_flexion`：准备阶段可见的二维屈膝幅度。

每项指标必须同时满足 `coach_eligible = true` 和最低置信度。至少两项可用时才生成
0 至 9 分的“可见动作分”，并同时输出 `uncertainty_9`；证据不足时分数保持 `null`，
不能用 0 分代替未知。

以下字段可以保留为分析代理，但不得驱动本地 Coach 或 DeepSeek 建议：

- 单机位肩线/髋线投影不能作为真实三维肩髋分离；
- 身体中心在屏幕中的位移不能直接称为重心转移；
- 触球后身体中心位移不能单独判断平衡稳定性；
- 没有可靠球拍轨迹时，球拍速度不得记为 0 分；
- 没有可靠球拍—球接近证据时，不评价击球点离身体远近。

`biomechanics.metrics.*.exclusion_reason` 记录排除原因。DeepSeek 旁路只接收这些原因，
不接收被排除指标的数值，避免模型重新生成过度确定的三维或动力链结论。

07.20 当前三拍可见动作分为 `6.69 / 3.80 / 5.72`，均值 `5.40/9`。这与该视频的
人工整体评价区间一致，但仍属于单视频初始校准，不代表跨球员、跨机位的常模。

### 7.8 会话质量与漂移看板

`summary.session_quality` 使用 `swing_session_quality_v1`，离线报告和实时报告共享同一份
聚合结果。看板包含：

- `quality.evidence_quality_score_100`：姿态、网球、球拍和触球证据的加权质量，不是球员技术分；
- `quality.visible_technique_mean_9`：当前会话可见动作均值；
- `series`：每次挥拍的技术分、证据质量、机位尺度、警告和 Coach 建议；
- `operations`：本地 Coach 延迟、DeepSeek 成功率和旁路延迟；
- `recurring`：反复出现的检测警告与动作建议；
- `drift.indicators`：会话前段与最近窗口的变化；
- `alerts`：区分 `capture`、`technique` 与 `segmentation` 的会话级问题。

少于 6 次挥拍时状态为“预热中”，即使首尾差值很大也不下漂移结论。人物画面尺度变化
超过 15% 时，技术变化标记为受机位影响；相邻事件时间范围重叠时，漂移状态直接暂停，
避免把事件重复切分误认为球员状态变化。

## 8. 推荐校准流程

每条新场地或新机位的视频，建议这样校准：

1. 选 2 到 5 条代表性视频。
2. 先跑完整流程，生成标注视频和报告。
3. 教练在报告页完成最小人工标注。
4. 生成 `*_swing_evaluation.json`。
5. 先看计数是否正确，再看类型是否正确，最后看 contact 是否准确。
6. 每次只调整一类参数，不要一次调很多。
7. 调整后重新跑同一批视频，对比 evaluation 指标。
8. 如果指标提高且肉眼确认合理，再固化参数。

推荐验收目标：

- 事件计数准确率：优先达到 90% 以上。
- 挥拍类型准确率：稳定场景下目标 85% 以上。
- contact frame 准确率：以 3 帧容忍为基准，目标 80% 以上。
- `review_recommended` 比例逐步下降，但不应强行降到 0。

## 9. 教练反馈模板

教练复核时可以按这个格式反馈：

```text
视频：
机位：正视 / 侧视 / 镜像 / 不确定
球员惯用手：右手 / 左手 / 不确定

事件总数：人工认为 X 次，系统识别 Y 次
类型问题：Event A 应为正手；Event B 应为双反
击球帧问题：Event C contact 早/晚约 N 帧
球/拍问题：Event D 球轨错 / 球拍漏检 / 镜中球干扰 / 静止球干扰
姿态问题：Event E 手腕/肩/膝节点跳动
是否可用于 AI 教练评价：可以 / 谨慎 / 不建议
备注：
```

## 10. 给 AI 教练使用时的原则

如果把视频和 JSON 交给 Gemini 或其他大模型做专业教练评价，请遵守：

- 视频是主要视觉证据。
- 人工标注优先于模型预测。
- `quality_flags` 是可信度提示，不是学生动作缺陷。
- 球/拍/pose 质量弱时，技术建议要降低确定性。
- 不要让 AI 编造 3D 旋转、真实拍面角度、球速、旋转或落点深度。
- 优先使用 `coach_calibration.assessments` 和 `decision_policy.advice_candidates`，
  不要从被排除的二维代理重新推断动力链。
- 如果 JSON 和视频不一致，应明确列出需要人工复核的事件。

## 11. 当前已知局限

- 当前仍缺少大规模人工真值标注集。
- 单摄像头无法可靠恢复真实 3D 动作。
- 镜像机位需要单独确认左右手映射。
- 球速、旋转、落点深度、拍面角度目前不能作为强结论。
- 高速挥拍、遮挡、反光、多人同框都会降低置信度。

## 12. 一句话总结

这套系统适合先做“可复核的挥拍事件分析”和“AI 教练数据准备”。教练的核心作用是校准事件真值：确认次数、类型、击球帧和质量问题。人工标注越充分，后续系统参数和 AI 教练反馈就越可靠。
