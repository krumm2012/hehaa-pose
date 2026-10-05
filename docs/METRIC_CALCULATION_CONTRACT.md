# 指标计算契约

版本：`tennis.metric-contract.v4`。实现定义源：`metric_contracts.py`。2026-10-05。

本文记录当前程序实际计算，不代表独立准确性验证。JSON 的每项 `contract` 与 OSD `metric_contracts`、报告数据行引用同一份定义。历史输出不自动改写；新分析携带版本。

2026-10-05补充：事件/Coach媒体时长消费者采用`event_phase_media_time_v1`，
原视频回放不再用帧号/FPS定位。自动建议另由代码批准规则集合约束，当前集合为空；
历史子分或资格标记不能开放技术建议。完整公式、窗口、缺失原因和回放绑定边界见
[事件源时间与自动建议资格](EVENT_SOURCE_TIMING.md)。身体与轨迹窗口的后续迁移见下文，不代表准确性验收。

## 公共规则

- 双视角先还原 ROI 缩放，保持各自镜像方向；肩宽比仍受视角/透视影响，是未标定投影代理。
- 身体宽度：每帧有效肩宽/髋宽（至少 4px）的中位数，再取事件中位数。
- `confidence` 是启发式证据质量，不是准确概率；`accuracy_validated=false`。
- `fresh_front_pose_v2_finite_observations`：新鲜正面观测须提供有限数值坐标、范围为[0,1]的模型分数，并通过原有分数、观测来源和源帧资格。布尔值、数值字符串、NaN和Infinity不授权测量。退化或非有限几何返回null，不把非法余弦夹成0°；关节角先归一化向量再计算点积。原始观测保持不变，旧XY记录仍明确未核验。
- `observation_qualification`按实际选中窗口保存被拒绝的关节、源帧及原因；整帧容器异常使用`joint=null`和记录级原因。缺值时相应原因进入`contract.missing_reasons`，有其他合格样本时保留拒绝证据，不将它们自动解释为整个指标缺失。
- 缺失值保持 null；`missing_reasons` 描述缺失，`coaching_exclusion_reason` 单独记录评分禁用原因。旧模块未提供详细原因时明确返回 `insufficient_inputs_or_qualification`，不编造原因。
- OSD资格版本`osd_observation_qualification_v6_finite_values`：方向角与上升比分别记录`fields`。端点重合时方向角为空，合格上升比0保留为独立canonical metric；已知源帧号不匹配、非法坐标/分数及容器异常拒绝参与测量。缺来源的历史字段不能充分核验。
- 球拍框必须是有限的四个数值坐标且宽、高为正；声明的模型分数须在[0,1]。异常主观测不能借用兼容框提升测量资格。动力链拍峰仍要求原有>=0.5模型分数，非有限速度断开片段；检测框中心不等于真实拍头。
- 双视角解析保留`observed`、`recovered_from_mirror`、`source_frame_id`和`confidence_source`；低分、旧点和镜面补点不能因重新包装而成为新鲜测量。该更改不新增推理或三维重建。
- 报告将证据百分比标为“证据参考…·未校准”，复核建议显示“复核提示”；过滤阈值不能批准未经验证的评分或技术规则。标明含义不等于完成概率校准。
- 实时分段排除已发布事件尾部，但测量可从有上限的原始帧缓存读取触球窗口；不改变传统事件统计窗口。`analysis_build`记录公式/资格/评分版本、源文件哈希与生成时间，旧报告不自动重写。
- 控制台统计行按动作、单位、方法分组；行数与指标种类数分开显示，两者都不是独立准确性验收数或五维评分数。五维阻断原因不构成评分规则；自动规则集合仍为空。
- 人工重算保留指定源帧，缺失/重复观测不借邻帧；采集器沿用事件触球锚点。阶段回位候选按0.08秒低运动支持与至少3个连续源观测判断，缺时钟时明确降为观测数启发式。人工区间跨度、实际样本数和源媒体时长分别记录，详见`EVENT_SOURCE_TIMING.md`。
- 人工标注页面使用`manual_annotation_identity_v2_event_links`，后端共享安全整数契约；非法输入不保存、不下载、不评估。来源事件与标注身份、重复链接及当前卡片冲突在导入/恢复前校验；显式空来源不继承旧事件ID，空触球/峰值帧不借旧值。草稿可缺锚点，正式导出/评估需完整开始/触球/结束顺序。历史页面不自动升级。
- 挥拍评估比较参考一致性，旧accuracy字段在新输出中保留为null，匹配比例有独立命名；未复核页面不显示比例。模型辅助来源与复核定稿不批准独立准确性，详见`EVALUATION_REFERENCE_CONTRACT.md`。
- Python/CLI评估入口采用`evaluation_identity_v1_strict_source_frames`：完整输入先校验，声明帧号不取整或强制转换，显式null不借旧字段，真实0帧保留；重复身份、反序区间与非法容差拒绝。合法旧别名仅建立内存规范副本，输出不能覆盖原证据。此资格不证明帧的实际存在、曝光或独立准确性。
- 报告入口采用`report_identity_v1_strict_event_and_coach_ids`：模型与Coach文档均先校验唯一整数身份及声明帧别名，再按原ID关联。模型缺少触球锚点时不从Coach补回；时间轴缺锚点不生成0帧标记或定位。`report_identity.source_binding_verified=false`明确仅完成声明ID校验，跨会话来源绑定和剩余身份消费者继续由T04跟踪。
- 身体指标使用`body_and_trajectory_windows_v1_source_time`：触球/端点±0.08秒，准备屈膝取起始至触球的前半媒体时间；肩线基线取前四分之一并限于0.16秒，回位参考选触球后0.40秒最近的事件内实际观测。窗口裁剪到事件，样本中位数不是时间加权中位数。
- 完整、严格递增的文件PTS及源帧身份才授权当前身体窗口；显式无效、估计或不可用时钟不借用兼容时间。旧记录保留明确未核验的时间/帧窗路径。`window_evidence`保存时间依据、选中源帧、实际范围、缺口和拒绝原因；样本可用比例不声称时间覆盖率。
- 无效源时间无法定位到窗口，OSD 保守计入覆盖分母并拒绝展示；`unlocated_time_frames` 留下证据。时钟倒退/重复同样拒绝。
- 动力链遇到关键点缺失或时间断点切段，每段独立微分、三点中位滤波和找峰；不足 6 个速度样本的段不能借用其他段。多个可用段保守拒绝合并。
- 每对峰值区间 A=[a0,a1]、B=[b0,b1]，延迟范围为 `[b0-a1-cadence, b1-a0+cadence]`。范围包含零则先后不可分辨。双视角取区间包络，肩—拍包含球拍峰宽。这个范围不是统计置信区间。
- 足部跨度和髋位移门槛按事件身体宽度归一化（0.2、0.04），去除相邻步长作为噪声的假设；仍是未经误差标定的规则。当前所有自动技术评分曲线禁用，人工确认评分保留。二维与三维差异仍是已知限制。

## 指标目录

运动特征、分段、球质量窗口及实时等待/去重的2026-10-05更新见
[MOTION_SOURCE_TIME.md](MOTION_SOURCE_TIME.md)。旧位移信号与源时间归一候选分开保存，
原帧参数明确作为25Hz历史调参单位。下表与v4实现定义保持一致，并共同应用上述数值资格规则。未标定空间含义和独立准确性仍待验证。

| ID | 公式 | 坐标系 | 窗口 | 有效条件 |
|---|---|---|---|---|
| hip_shoulder_separation | abs(wrap180(shoulder_line_angle-hip_line_angle)); sample median | front image plane | contact source PTS +/-0.08s, clipped to event | observed shoulder and hip endpoints; reported source time; projected separation only |
| shoulder_turn | sample median atan2(back_shoulder_width,front_shoulder_width); fallback absolute shoulder image angle | ROI coordinates restored to source pixel scale for width proxy; front image for fallback | contact source PTS +/-0.08s, clipped to event | both source-scale widths >15 px for dual proxy; reported source time; no calibrated 3D interpretation |
| shoulder_turn_change | max abs(wrap180(angle-circular_sample_median(baseline))) | front image shoulder-line orientation | event start through contact; baseline first quarter source elapsed interval capped at0.16s | finite observed shoulder angles and observed baseline; baseline unwrapped relative to first angle |
| preparation_knee_flexion | sample median(180-mean(available left/right hip-knee-ankle angles)) | front image joint coordinates | first half of start-to-contact source elapsed interval | reported source time; at least one complete nondegenerate leg; side availability can vary |
| arm_extension | sample median hitting-side shoulder-elbow-wrist interior angle | front image joint coordinates | source PTS +/-0.08s at contact if contact score>=.35, otherwise peak; clipped to event | reported source time; complete nondegenerate observed arm |
| contact_lateral_distance | abs(ball_x-body_center_x)/event_body_width; closest supported source-time sample | front original image pixels; body center=mean of shoulder and hip midpoints | contact source PTS +/-0.08s, clipped to event | ball and torso center plus positive body scale; reported source time; coaching gates are separate |
| weight_transfer | distance(sample_median_center(start),sample_median_center(contact))/event_body_width | front image Euclidean displacement, not physical weight transfer | each endpoint source PTS +/-0.08s, clipped to event | both torso centers and positive body scale; reported source time |
| balance_drift | distance(sample_median_center(contact),sample_median_center(early_recovery))/event_body_width | front image Euclidean displacement, not balance stability | endpoint source PTS +/-0.08s; recovery observed PTS nearest contact+0.40s within event | both torso centers and positive body scale; reported source time |
| takeback_depth | max min(2.5,abs(back_wrist_x-back_shoulder_mid_x)/back_shoulder_width) | back ROI projection restored to source pixel scale | start through contact | back wrist and shoulders; width >20 px; not physical depth |
| scapular_retraction | max min(3,back_shoulder_width/front_shoulder_width) | ROI coordinates restored to source pixel scale | start through contact | both shoulder widths >15 px; not anatomical scapular motion |
| racket_head_speed | distance(current_box_center,previous_box_center)/source_dt | original image pixels per source second | contact and immediately preceding frame | fresh consecutive boxes; positive same-basis source dt; no km/h calibration |
| racket_max_speed | max valid box-center pixel speed | original image pixels per source second | event window | same source-time and observation requirements as contact speed |
| brush_angle | atan2(max(0,lowest_y-contact_y),abs(lowest_x-contact_x)) | front original image box centers; y increases downward | source time contact [-.48,0] seconds | candidate/confirmed contact; >=5 observations; >=.8 coverage; qualified contact box; no clock gaps; noncoincident chord |
| drop_depth_ratio | max(0,lowest_y-contact_y)/event_body_width | front original image; normalized by body width | source time contact [-.48,0] seconds | qualified contact and track coverage plus positive body scale; coincident chord permits zero rise but no direction |
| stance | median atan2(abs(right_ankle_y-left_ankle_y),abs(right_ankle_x-left_ankle_x)) | front original image ankle-line inclination | source time contact [-.12,+.12] seconds | >=3 fresh ankle pairs score>=.5, span>=.2*event_body_width; coverage>=.8; range<=10 degrees; valid clocks |
| leg_drive | (max(hip_mid_y)-contact_hip_mid_y)/event_body_width | front original image vertical displacement | source time contact [-.6,0] seconds | >=5 fresh hip pairs score>=.5; coverage>=.8; contact observed; rise>.04*event_body_width; valid clocks |
| kinematic_sequence | circular angle derivatives; per-continuous-run median3; peak>=90% max interval; pair lag intervals including cadence | independent original front/back image lines and racket box centers; source seconds | contact [-.6,+.16] seconds | source-time contract; point score>=.5; line>=12px; >=6 samples per run; coverage>=.6; reject broad/boundary/ambiguous peaks and view conflicts |
| swing_quality_score | practice scoring policy; automatic curves disabled pending independent validation; confirmed manual rubric only | dimensionless policy score, not physical measurement | event | practice policy eligibility; no calibrated accuracy probability |

## 验证

`test_measurement_value_validity.py`覆盖NaN关节假0°/180°、非法分数、无效位移、球拍框及拍峰输入、异常容器、双视角来源标记和严格派生测量JSON。`test_evidence_quality_display.py`直接验证实时和独立报告渲染的证据含义与复核状态。严格派生JSON不等于所有历史原始日志已完成数值规范化；原始观测、空间精度和分数校准另行验证。

`test_body_metric_source_windows.py`覆盖非均匀时间窗口、兼容时钟无关性、源帧间隔、准备时间分割、回位、缺失锚点、短缺口补点和旧峰值降级。旧兼容路径只保留`legacy_candidate_peak_frames`，不再输出按FPS换算的峰值毫秒或OPTIMAL/DISCONNECTED技术结论。


`test_metric_formula_regressions.py` 包含 ±180° 角度、遮挡断段、无效时钟覆盖、球拍宽峰传播和 JSON/OSD/报告契约一致性回归。合成回归证明这些程序错误得到约束，不代替真实视频与独立标注的误差评估。

旧 `ALGO_2.0_DESIGN_AND_PROGRESS.md` 是历史设计记录；当前操作定义以本契约及实现为准。动力链研究边界参见 `KINEMATIC_SEQUENCE_CROSS_VALIDATION.md`。


### 历史源帧资格补充（OSD v5）

动力链髋、肩及球拍观测携带明确 `source_frame_id` 时，必须与当前记录帧号一致；
冲突即视为无效观测并断开微分片段，即使 `observed=True` 也不能覆盖此条件。
缺少源帧号的历史记录仍按旧格式读取，不代表来源已验证。该规则不提高置信度。
人物选择恢复的侧身点仍需独立通过角度的 12px 线长检查；该角度阈值尚是像素启发式，
未完成定位误差与分辨率校准，不可据此宣称跨尺度动力链准确性。
