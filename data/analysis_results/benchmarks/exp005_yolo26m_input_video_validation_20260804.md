# exp-005 YOLO26m Core ML：input_video.mp4 兼容性验证

- 日期：2026-08-04
- 输入：`data/input_video.mp4`
- 输入 SHA-256：`a62fe2cd73b739eed5f86e214ab262cc8ffe07a36fca449de327d1e33aa2c07b`
- 视频：2560×1440、25 FPS、10秒、250帧
- 候选配置：`configs/yolo26_tennis_exp005_candidate.yaml`
- 对照配置：`configs/yolo26_tennis_config.yaml`（exp-004）

## 完整管线结果

| 项目 | exp-004 YOLO26n | exp-005 YOLO26m |
|---|---:|---:|
| 完成帧数 | 250/250 | 250/250 |
| 异常退出 | 否 | 否 |
| 总处理时间 | 12.68秒 | 21.30秒 |
| 平均吞吐 | 19.72 FPS | 11.74 FPS |
| 15 FPS门槛 | 通过 | 未通过 |

exp-005 输出视频：

`data/analysis_results/benchmarks/exp005_yolo26m_input_video_validation.mp4`

SHA-256：`65a60455aea695741505310216f8d279c6e0f0c121f048681c0ddd06b9f8924a`

exp-004 对照视频：

`data/analysis_results/benchmarks/exp004_yolo26n_input_video_control.mp4`

SHA-256：`5b737e681e257649b3bf9cdca83e8dad21b01580ee59d588b3fe8f100b4948df`

## 单次统一推理统计

为排除现有兼容 wrapper 每帧重复调用统一模型两次的影响，另外按每帧只调用一次
`detect_unified` 复算：

| 指标 | exp-004 | exp-005 |
|---|---:|---:|
| 原始网球候选覆盖帧 | 250 | 250 |
| 原始球拍候选覆盖帧 | 103 | 42 |
| 最终主球输出帧 | 250 | 250 |
| 最终球拍输出帧 | 132 | 103 |
| 原始网球候选数 | 415 | 358 |
| 原始球拍候选数 | 115 | 44 |
| 推理 p50 | 5.17 ms | 24.89 ms |
| 推理 p95 | 5.54 ms | 25.64 ms |

exp-005 的球拍原始覆盖明显低于 exp-004，与 v002 val 上的球拍回退结论一致。两者主球均
覆盖250帧，但该数字不能解释为运动球 Recall，因为画面中长期存在静止实体球和镜像球。

## 证据边界

`data/input_video.mp4` 已包含姿态骨架、球/拍框、圆圈、文字等程序覆盖层，并非干净原始视频。
当前运行时开启覆盖层恢复，因此输出画面和250帧主球覆盖会受到既有标记影响。该视频适合验证
运行时兼容、稳定性和吞吐，不适合验证纯模型 Precision/Recall 或陌生场景泛化。

## 性能瓶颈

当前 `main.py` 先调用 `BallDetectionWrapper.predict_ball`，再调用
`RacketDetectionWrapper.detect_rackets`；两个 wrapper 都会执行一次 `detect_unified`，因此同一帧
统一检测器被推理两次。exp-005 单次约25 ms，重复调用是完整管线降至11.74 FPS的主要原因。

exp-005 结束时还出现 `MILCompilerForANE` 编译失败提示，但 Core ML 允许的其他计算单元完成了
全部250帧；候选配置使用 `compute_units: ALL`。这仍属于需要后续治理的加速风险。

## 结论

exp-005 已证明可以被 tennis_analyzer 加载并完成两类统一检测接口和250帧完整处理，但当前：

1. 完整管线低于15 FPS门槛；
2. 球拍覆盖明显低于 exp-004；
3. 输入视频含预绘制覆盖层，不能作为纯模型效果验收；
4. 仍需干净原视频和至少1,500帧产品回归。

因此保留为显式备选模型，不切换默认配置。
