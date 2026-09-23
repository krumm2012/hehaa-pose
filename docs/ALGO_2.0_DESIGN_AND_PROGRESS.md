# 算法 2.0：虚拟双机位融合与高级网球生物力学（设计与进度文档）

**版本**：v2.0-draft  
**更新日期**：2026-09-22  
**当前分支**：`algo-2.0`  
**关联代码库**：`tennis_analyzer`，参考借鉴 `https://github.com/krumm2012/Tennis-Vision`

---

## 一、 背景与设计目标

### 1.1 背景与现状
在室内网球训练仓（如 Camera04 场景），摄像头自天花板斜上方俯拍场地，后墙配有全景大平镜。
- **历史方案局限**：原系统中镜面被视作干扰与噪点源（配置了 `mirror_ball_filter: shadow` 与镜像惩罚），同时单机位正面拍摄在选手侧身击球（Side-on）时会导致肩宽投影塌陷（Side-on Collapse），且向后引拍和反手击球时击球臂常被躯干遮挡。
- **算法 2.0 创新**：将镜面反射“变废为宝”，转化为**物理 100% 毫秒级硬同步的虚拟背面机位**。无需引入单帧需耗时 1.58 秒的重型 SAM-3D 模型，即可通过真实背面图像获取完整后背与引拍动作。
- **借鉴 `Tennis-Vision`**：
  1. 移植基于身体相对坐标系与躯干轴中线投影的击球分类算法（`classify_forehand_backhand`）。
  2. 移植基于双腕间距与肩宽比的双手握拍（Two-handed Backhand）判定（`TWO_HANDED_MAX_GAP`）。
  3. 移植触球手腕-球间距物理门控（`max_contact_distance`），彻底过滤空挥假动作。

### 1.2 目标产物
1. **机位解耦组件**：从单路 2.5K 视频/RTSP 实时分流出正面（Front）与背面（Back，已做水平镜像翻转矫正）两路标准流。
2. **前后双视角姿态融合引擎**：无缝运行双路 Core ML 姿态估计，支持侧身抗退化与遮挡自愈。
3. **后背特色生物力学指标**：引拍深度（Takeback Depth）、肩胛收缩度（Scapular Retraction）、360° 无退化转肩分离角（X-Factor）。
4. **双视角同步视频渲染与教练报告**：输出 Side-by-Side 双画面视频，并在控制面板和时间轴中联动展示。

---

## 二、 架构设计与核心数据流

```
                     ┌───────────────────────────┐
                     │ Camera04 原始视频 (1440p)  │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │     DualViewManager       │
                     │  (多边形裁剪 + 镜像水平翻转)  │
                     └──────┬─────────────┬──────┘
                            │             │
              正面帧 (Front) │             │ 背面帧 (Back Flipped)
                            ▼             ▼
                     ┌───────────┐   ┌───────────┐
                     │ FrontPose │   │ BackPose  │ (yolo26m-pose / Core ML)
                     └─────┬─────┘   └─────┬─────┘
                           │               │
                           └───────┬───────┘
                                   │ 关键点归一化
                                   ▼
                     ┌───────────────────────────┐
                     │ DualViewBiomechanicsEngine│
                     │  1. 正反手判定 (中线跨越)    │
                     │  2. 双手握拍检测 (双腕间距)   │
                     │  3. 触球距离门控 (剔除空挥)   │
                     │  4. 抗侧身退化转肩角 (X-Factor)│
                     │  5. 盲区手臂遮挡自愈         │
                     │  6. 后背引拍深度 & 肩胛收缩   │
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │     DualViewRenderer      │ │    LocalRealtimeCoach     │
       │ (Side-by-Side 对比视频/图片)│ │  (融入后背动力链指导建议)   │
       └───────────────────────────┘ └───────────────────────────┘
```

---

## 三、 核心几何与数学公式

### 3.1 躯干轴与中线跨越投影（正反手物理判定）
* **肩部轴向向量**：
  $$\vec{V}_{\text{axis}} = P_{\text{L\_shoulder}} - P_{\text{R\_shoulder}}$$
* **躯干中心点**：
  $$P_{\text{center}} = \frac{1}{2}(P_{\text{L\_shoulder}} + P_{\text{R\_shoulder}})$$
* **击球手腕投影偏移**：
  $$\text{side} = \frac{(P_{\text{wrist}} - P_{\text{center}}) \cdot \vec{V}_{\text{axis}}}{\|\vec{V}_{\text{axis}}\|}$$
  * 右手选手：$\text{side} < 0$ 为正手（手腕在右半身）；$\text{side} > 0$ 为反手（手腕跨越中线至左半身）。

### 3.2 双手握拍几何门控
* 双腕欧氏距离：
  $$\text{gap} = \|P_{\text{L\_wrist}} - P_{\text{R\_wrist}}\|$$
* 当 $\text{gap} < 0.45 \times \|\vec{V}_{\text{axis}}\|$ 且两手同向靠近球拍时，判定为**双手持拍动作**。

### 3.3 抗侧身塌陷转肩角（Anti Side-On Collapse）
* 单目前景机位视角下，当身体垂直于镜头时，$\|\vec{V}_{\text{axis}}\|_{\text{proj}} \to 0$。
* 融合视角下，利用正反双视角法向互斥性：
  $$\theta_{\text{shoulder}} = \text{atan2}(V_{\text{axis\_front}}, V_{\text{axis\_back}})$$
  在 $[0^\circ, 180^\circ]$ 全域内连续且无奇点，彻底解决侧身时投影计算失效的问题。

---

## 四、 迭代任务分解与进度跟踪看板

| 任务编号 | 阶段 | 模块 / 目标 | 涉及文件 | 状态 | 验收标准 / 交付物 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **TASK-00** | 基线准备 | 工作区基线快照归档 | git status / commit | 🟢 已完成 | 已将基线 34 个改动文件归档至 `algo-2.0` 分支 |
| **TASK-01** | Phase 1 | 编写 `DualViewManager` | `dual_view_manager.py`<br>`configs/dual_view_config.yaml` | 🟢 已完成 | 稳定分流 Front 与 Back(水平翻转) 画面，具备双向高精度坐标映射 |
| **TASK-02** | Phase 1 | 单元测试与离线切分验证 | `test_dual_view_manager.py` | 🟢 已完成 | 7 项测试全部通过，支持 `49.35.mp4` 离线切分 |
| **TASK-03** | Phase 2 | 移植 `Tennis-Vision` 判定 | `dual_view_biomechanics.py` | 🟢 已完成 | 完整实现解剖脊柱中线跨越、双手握拍间距比与触球物理距离门控 |
| **TASK-04** | Phase 2 | 正反手与击球门控测试 | `test_dual_view_biomechanics.py` | 🟢 已完成 | 6 项击球分类测试全部通过，有效排除空挥假动作 |
| **TASK-05** | Phase 3 | 双视角姿态互补与遮挡自愈 | `dual_pose_estimator.py`<br>`test_dual_pose_estimator.py` | 🟢 已完成 | 在 `49.35.mp4` 上跑通双路 17 关键点检测与遮挡手腕自动补全 |
| **TASK-06** | Phase 3 | 后背动力链指标开发 | `dual_view_biomechanics.py` | 🟢 已完成 | 输出抗侧身塌陷转肩角、后背引拍深度与肩胛收缩率 |
| **TASK-07** | Phase 4 | Side-by-Side 视频与渲染 | `dual_view_renderer.py`<br>`test_dual_view_renderer.py` | 🟢 已完成 | 实现双视角骨骼绘制、HUD 动力学仪表盘与端到端渲染 |
| **TASK-08** | Phase 4 | 全流程批处理验证 | `scripts/process_algo2_dual_view.py` | 🟢 已完成 | 成功将 `49.35.mp4` 完整处理并导出为 `algo2_dual_view_biomechanics.mp4` |
| **TASK-09** | Phase 5 | 帧级特征管线双视角对接 | `swing_motion_features.py` | 🟢 已完成 | 支持提取自愈姿态 (`healed_pose`) 与后背引拍/肩胛/转肩等双视角特征 |
| **TASK-10** | Phase 5 | 事件分类器双视角优先判定 | `swing_event_classifier.py` | 🟢 已完成 | 融合解剖轴中线跨越与双手持拍，支持 `dual_view_transverse_projection` 规则 |
| **TASK-11** | Phase 5 | 双视角生物力学指标聚合 | `swing_biomechanics.py` | 🟢 已完成 | 引入 `dual_view_2d_v1` schema，聚合后背引拍深度、肩胛收紧度与 360° 抗塌陷转肩角 |
| **TASK-12** | Phase 6 | 智能教练双视角规则扩展 | `local_realtime_coach.py` | 🟢 已完成 | 新增后背引拍与肩胛收缩纠错建议，保持确定性且每条不超过 15 个汉字 |
| **TASK-13** | Phase 6 | 全管线集成与回归验证 | `test_algo2_pipeline_integration.py`<br>`scripts/process_algo2_dual_view.py` | 🟢 已完成 | 254 项测试全绿通过，在 `49.35.mp4` 上精准输出正手判定与后背引拍分析报告 |
| **TASK-14** | Phase 7 | 复杂长视频多动作分割 | `dual_view_biomechanics.py`<br>`dual_pose_estimator.py` | 🟢 已完成 | 解剖归一化遮挡自愈、多事件动态能量分割，在 `40.16.mp4` 实测 >40 FPS |
| **TASK-15** | Phase 7 | 网球击球接触窗口物理重构与随挥误判治理 | `configs/dual_view_config.yaml`<br>`swing_event_classifier.py`<br>`swing_event_segmenter.py`<br>`dual_view_renderer.py`<br>`scripts/process_algo2_dual_view.py` | 🟢 已完成 | 1. 彻底剔除随挥收拍对动作定性的污染，建立触球核心窗口机制；<br>2. 镜面 ROI 精确收缩杜绝前景人脸穿透；<br>3. 实现 Two-Pass 视频渲染与 PingFang 中文教练 HUD；<br>4. 在 `40.16.mp4` 修正为 100% 精准正手（Forehand），254 项测试全绿。 |
| **TASK-16** | Phase 8 | 统一球/拍感知、物理触球反弹检验、静止微动门控与 HUD 交互重构 | `dual_view_renderer.py`<br>`swing_event_classifier.py`<br>`scripts/process_algo2_dual_view.py`<br>`test_algo2_pipeline_integration.py` | 🟢 已完成 | 1. 接入 CoreML ANE YOLO26n 统一检测器 (6.8ms)，正面视口叠加动态球轨迹拖尾与青色球拍框；<br>2. 建立物理触球与轨迹反弹检验，精准判定真实击球与空挥；<br>3. 引入手腕邻域球拍优选与峰值窗口生物力学蓄力门控，完美解决第 197 帧站立误检并归位 `READY STANCE`；<br>4. 顶部 HUD 彻底消除文字重叠；<br>5. 256 项自动化测试全绿通过，产出 `algo2_40_26_biomechanics.mp4`。 |
| **TASK-17** | Phase 9 | 三个梯队高级生物力学指标、击球特写卡片与综合技术评分标准落地 | `swing_biomechanics.py`<br>`dual_view_renderer.py`<br>`scripts/process_algo2_dual_view.py`<br>`docs/SWING_QUALITY_SCORING_STANDARD.md` | 🟢 已完成 | 1. 落地三个梯队：拍头动力学挥速、刷球仰角与下潜、步法站位分类、垂直蹬地率、双机位 3D 相对深度与动力学链时序；<br>2. 实现击球瞬间特写遥测卡片与 10 帧子弹时间定格；<br>3. 建立 100 分制单拍综合技术评分（Swing Quality Score）与四级段位（PRO/ADVANCED/INTERMEDIATE/DEVELOPING），详见 [`docs/SWING_QUALITY_SCORING_STANDARD.md`](./SWING_QUALITY_SCORING_STANDARD.md)；<br>4. 治理镜面跳变与有球训练走动误检，260 项自动化测试全绿。 |
| **TASK-18** | Phase P1 | 镜面与人像遮挡标定工具集成、Web HTML 5维雷达图与动力学链时序升级 | `local_control_panel.py`<br>`local_control_panel.html`<br>`calibrate_mirror.py`<br>`mirror_calibration.html`<br>`dual_view_manager.py`<br>`swing_report_builder.py`<br>`test_dual_view_manager.py`<br>`test_swing_report_builder.py` | 🟢 已完成 | 1. 控制台原生集成镜面标定工具（`/mirror-calibration` 路由与 API）；<br>2. 交互式人像遮挡区（`mask_polygon`）标定与 Backview 实时半透明暗色隐私遮罩渲染；<br>3. Web HTML 报告升级：原生 SVG 5 维技术雷达图（转肩/引拍/延展/挥速/蹬地）、四级段位徽章（PRO/ADVANCED/INTERMEDIATE/DEVELOPING）与 100 分制仪表、动力学链传递延时条（$\Delta t_{\text{hip}\to\text{sh}}$ 和 $\Delta t_{\text{sh}\to\text{rkt}}$）、击球定格特写与遥测指标网格；<br>4. 全量自动化单元测试增至 265 项且 100% 通过。 |
| **TASK-19** | Phase P0 | 生产实时化与主工程对接（离线批处理 ➔ 实时运行） | `main_pipe.py`<br>`frame_processor.py`<br>`realtime_swing_runtime.py`<br>`qwen3_tts_sidecar.py`<br>`test_algo2_realtime_pipeline.py` | 🟢 已完成 | 1. 深度集成 Algo 2.0 虚拟双机位管线到生产级 `main_pipe.py`（支持 `--algo2-dual-view` / `--dual-view` 开启，保持原有单机位 100% 向后兼容）；<br>2. 实时推理多进程架构：ANE YOLO-pose 前后双机位姿态、遮挡自愈姿态合成与镜面球拍误检空间过滤；<br>3. 实时分析多进程渲染：Side-by-Side HD ($1080 \times 720$) 实时拼合、隐私遮罩、荧光动态球轨迹拖尾与击球瞬间特写遥测卡片平滑悬浮；<br>4. 在 `40.26.mp4` 生产流水线上实测稳定达到 25.4 ~ 26.3 FPS（Hardware Videotoolbox 编码），低延迟输出实时击球事件与三梯队生物力学指标；<br>5. 全量自动化单元测试增至 270 项且 100% 通过。 |

**状态图例**：  
- ⬜ 待开始 (Pending)  
- 🟡 进行中 (In Progress)  
- 🟢 已完成 (Done)  
- 🔴 遇到阻碍 (Blocked)

---

## 五、 测试与验收基线数据集

1. **基线视频 1**：`/Users/krum5539/Desktop/Camera/49.35.mp4`
   - 规格：2560x1440, 25/50fps, 10秒, 正手击球与后背引拍。
   - 验证结果：
     - 正面视角与镜像视角高保真切分，水平翻转矫正无失真。
     - 动作类型精准识别为 `Forehand`（置信度 90.0%，判定规则 `dual_view_transverse_projection`）。
     - 后背引拍深度比 `2.5562`，肩胛骨收缩比率 `1.2673`，抗侧身塌陷转肩角 `35.95°`。
     - 导出视频：`/Users/krum5539/Desktop/Camera/algo2_dual_view_biomechanics.mp4`（双视角同屏骨骼 + HUD 仪表盘，耗时 11.79s，吞吐率 21.2 FPS）。
2. **基线视频 2**：`/Users/krum5539/Desktop/Camera/test/40.16.mp4`
   - 规格：2560x1440, 25.14fps, 250帧 (~10秒), 连续两次单手正手击球（右手）。
   - 深入力学排查与修正：
     - 原先错误将随挥阶段（Follow-through 扫过对侧肩胛，双腕靠近）误判为双手反拍。重构为“触球前动力链窗口（Pre-impact & Contact Window）”判定后，100% 恢复物理真值。
     - 镜面 ROI 收紧为 `[0.36, 0.11, 0.54, 0.35]`，彻底消除前景选手头部入侵镜面的双骨骼重叠杂讯。
   - 验证结果：
     - 耗时 6.87 秒完成 Two-Pass 全流程处理与 H.264 导出（Pass 1 姿态 45.9 FPS，Pass 2 渲染 176 FPS）。
     - 精准捕获并自动分割出两次击球事件，均真实、确定性分类为 **`Forehand`**：
       - **事件 #1 (帧 64~130, 击球点 99)**: `Forehand` (置信度 90.0%)，引拍深度比 `1.7248`, 肩胛收紧度 `1.2522`, 转肩角 `47.61°`, 展臂 `82.85°`, 教练建议：`[limited_arm_extension] 挥拍时手臂再舒展`。
       - **事件 #2 (帧 165~218, 击球点 187)**: `Forehand` (置信度 100.0%)，引拍深度比 `1.6294`, 肩胛收紧度 `1.0056`, 转肩角 `41.09°`, 展臂 `157.63°`, 教练建议：`[limited_knee_flexion] 准备时适当降低重心`。
     - 导出视频：`/Users/krum5539/Desktop/Camera/test/algo2_40_16_biomechanics.mp4`。
3. **基线视频 3**：`/Users/krum5539/Desktop/Camera/test/40.26.mp4`
   - 规格：2560x1440, 25.1 FPS, 250帧 (~10秒), 包含真实击球与站立准备状态。
   - 治理与进化验证：
     - **球/拍感知与轨迹**：全流程集成 YOLO26n ANE 统一检测器，实时勾勒正面视角动态荧光网球轨迹拖尾与持拍框。
     - **第 197 帧站立误检根治**：手腕关联优选球拍杜绝后墙镜面虚像跳跃，配合击球峰值窗口转肩引拍物理门控，成功将原先被误判为 `Forehand (Event #3)` 的第 197 帧准确判定为 `READY STANCE`。
     - **HUD 交互排版**：正面/背面视角标题两端靠齐，中间置入独立胶囊徽章与智能教练建议，实现零文字碰撞。
     - 验证输出：2 次真实物理挥拍事件：
       - **事件 #1 (帧 10~69, 击球点 36)**: `Forehand` (置信度 99.7%)，引拍深度比 `1.5215`, 肩胛收紧度 `1.2368`, 转肩角 `45.85°`, 展臂 `142.75°`, 教练建议：`挥拍时手臂再舒展`。
       - **事件 #2 (帧 98~170, 击球点 132)**: `Forehand` (置信度 100.0%)，引拍深度比 `1.5330`, 肩胛收紧度 `1.4367`, 转肩角 `42.56°`, 展臂 `150.30°`, 教练建议：`准备时适当降低重心`。
     - 导出成果视频：`/Users/krum5539/Desktop/Camera/test/algo2_40_26_biomechanics.mp4`（Pass 1 耗时 8.71s / 28.7 FPS，Pass 2 渲染耗时 1.36s / 183.8 FPS）。
4. **生产级实时化端到端验证（P0）**：
   - 接入主干生产程序 `main_pipe.py --algo2-dual-view --realtime-swing-events`，实时双路 ANE 姿态估计与 YOLO26 统一感知。
   - 实测在 Apple Silicon 芯片上维持 25.4 ~ 26.3 FPS 吞吐，低于 100ms 延迟，实时生成 Side-by-Side HD 视频、动态轨迹与悬浮遥测卡片。
5. **单元回归测试套件**：全量 275 项自动化测试覆盖所有动力学、机位解耦、控制面板及生产级主流水线集成模块，在 Python 3.14 与 Python 3.11 (CoreML) 双环境均 100% 通过。

---

## 六、 本地动作纠错与智能教练系统优化（Local Realtime Coach 2.0）

### 1. 业务痛点与排查（基于 40.16.mp4 实战发现）
在用户使用 `40.16.mp4` 于 Web 控制面板进行实时分析时，系统侦测到 3 次正手击球，但在前端界面与实时报告中输出建议全部为：
- Swing #1 (54.9分): `确保来球完整入镜` (`ball_track_gaps`)
- Swing #2 (77.3分): `减少球拍遮挡` (`racket_track_gaps`)
- Swing #3 (79.2分): `减少球拍遮挡` (`racket_track_gaps`)

经深度排查，定位两大根本原因：
1. **控制面板门槛过严**：前端将 `min_confidence` 误设为 `0.9`，而人体关键点骨骼计算的生物力学置信度分布在 0.82 ~ 0.88 之间（如屈膝 0.82、转肩 0.82、转肩角 0.88），导致所有核心动作问题被 100% 误杀，回退到球拍遮挡兜底。
2. **采集告警一票否决**：旧有 `_blocking_advice` 对 `ball_track_gaps`（未入镜）实施了硬阻断，即便人体骨骼姿态 100% 完整清晰（`pose_frame_ratio = 1.0`），技术纠错也被无条件遮蔽。

### 2. 优化方案与工程落地
1. **解耦阻断机制，技术建议优先输出**：
   - 阻断条件收紧为**仅人身姿态丢失**（`pose_frame_ratio < 0.65` 或 `confidence < 0.55`）时才真正硬阻断；
   - 只要人体姿态清晰，**技术建议（technique）优先输出**；当存在 `ball_track_gaps` 且 `max_suggestions > 1` 时，第 1 条作为提示，后续第 2、3 条并行输出动作指导。
2. **新增算法 2.0 高级力学纠错规则库**：
   - **动力学链时序断裂**（`disconnected_kinetic_chain`，优先级 95）：
     - 判定：`kinematic_sequence.details.sequence_quality in ("DISCONNECTED", "SUBOPTIMAL")`
     - 纠错播报：`"用身体核心带动球拍发力"`
   - **下肢垂直蹬地不足**（`limited_leg_drive`，优先级 89）：
     - 判定：`leg_drive.drive_ratio < 0.15`
     - 纠错播报：`"击球瞬间双腿蹬地发力"`
   - **拍头下潜/刷球不足**（`limited_brush_drop`，优先级 87）：
     - 判定：`drop_depth_ratio < 0.25` 或 `brush_angle < 15.0°`
     - 纠错播报：`"击球前拍头下潜刷球"`
3. **准备期转肩变化（`shoulder_turn_change`）阈值微调**：
   - 将 `min_shoulder_turn_change_deg` 从 12.0° 平滑调优至 10.0°，提高对临界转体好球的包容度。
4. **控制面板（Web UI）参数防护**：
   - `local_control_panel.html` 输入框设置推荐值 `placeholder="0.45"`，增加防呆 Tooltip 提示；
   - 默认建议条数推荐为 `3 条`。

### 3. 40.16.mp4 优化前后效果对比

| 击球事件 | 评分与段位 | 优化前输出建议 | 优化后输出建议 |
| :--- | :--- | :--- | :--- |
| **Swing #1**<br>(触球第 38 帧) | **54.9分**<br>(DEVELOPING) | `[确保来球完整入镜]` | 1. `[采集] 确保来球完整入镜`<br>2. `[技术] 用身体核心带动球拍发力`<br>3. `[技术] 准备时适当降低重心` |
| **Swing #2**<br>(触球第 96 帧) | **77.3分**<br>(ADVANCED) | `[减少球拍遮挡]` | 1. `[技术] 用身体核心带动球拍发力`<br>2. `[技术] 准备时适当降低重心` |
| **Swing #3**<br>(触球第 191 帧) | **79.2分**<br>(ADVANCED) | `[减少球拍遮挡]` | 1. `[技术] 用身体核心带动球拍发力`<br>2. `[技术] 准备时适当降低重心`<br>3. `[技术] 提前转肩充分引拍` |

### 4. 自动化测试验收
* 新增 4 组针对性单元测试（动力学链断裂、蹬地不足、拍头下潜、采集与技术共存）；
* 全套单元测试总数扩充至 **275 项，100% 通过**（耗时 3.392s）。


