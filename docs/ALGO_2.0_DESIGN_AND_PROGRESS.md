> 历史设计记录：部分“全域转肩/无损恢复”等表述未经验证，不能用作准确性标准。当前逐指标操作定义见 [指标计算契约](METRIC_CALCULATION_CONTRACT.md)。

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
| **TASK-20** | Phase P0 | 正反双机位几何校准、动态人体追踪自愈与各机位聚合标定 | `dual_view_manager.py`<br>`main_pipe.py`<br>`configs/roi_config.yaml`<br>`roi_stream_config.py`<br>`realtime_swing_pipeline.py` | 🟢 已完成 | 1. 动态自适应人体追踪（EMA 平滑，540x720 3:4 黄金宽高比，彻底消除拉伸畸变）；<br>2. 人体工学自愈扩展，彻底解决切掉人/球/拍问题；<br>3. 物理镜面边界校准，消除视角颠倒与拍部遮挡；<br>4. 推理与渲染进程透传同步；<br>5. 全量 303 项测试 100% 通过。 |
| **TASK-21** | Phase P0 | 双视角独立实例姿态并发推理与延迟大幅压缩 | `dual_pose_estimator.py`<br>`main_pipe.py`<br>`test_dual_pose_estimator.py` | 🟢 已完成 | 1. 为 Front View 与 Back View 维护独立 CoreML 模型实例并开启 ThreadPool 并发调度；<br>2. 单帧双姿态耗时从 27.3ms 骤降至 19.9ms（降幅 ~28%），理论吞吐从 36 FPS 提升至 50+ FPS；<br>3. 全量 305 项单元测试 100% 通过。 |

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

---

## 九、 空挥静默机制与球拍追踪连续性（racket_track_gaps）根因消除 (2026-09-23)

### 1. 业务需求与问题定位
1. **空挥（无来球）指导静默**：
   - 规则明确：针对全程无球触碰的空挥动作（`is_shadow_swing == True`）以及模型综合置信度低的无效事件，**不输出教练指导建议，跳过动作校准**，界面标注为 `[空挥练习 · 无来球]`，避免无意义的“确保来球完整入镜”告警干扰学员。
2. **`racket_track_gaps` 频繁误报排查**：
   - 在 `40.16.mp4` 击球中，Swing #2 检出率仅 35.59%，Swing #3 仅 61.82%，频繁误报“减少球拍遮挡”。

### 2. 根因分析
* **Overlay Marker Recovery 蓝幕误报**：内置标记恢复算法（HSV 95~115）将球场后墙蓝色防撞幕布接缝误判为球拍并给出 **0.99 假置信度**，导致单拍优选器（Selector）压制了前景置信度 0.85 的真实球拍；随后该假拍在镜面区域过滤（$y < 620$）被剔除，导致当帧球拍变为 `None`。
* **镜面高度比率设置偏低**：原配置 `racket_mirror_min_y_ratio: 0.18`（仅覆盖 $y < 259\text{px}$），未覆盖到球场后墙 $y \approx 620\text{px}$（比率约 0.43）的大镜面，镜中倒影拍在转身时未受惩罚。
* **瞬态丢帧补偿缺失**：高速抽球（如击球瞬态拍头甩动）存在 1~2 帧运动模糊。

### 3. 工程落地
1. **配置参数调优 (`configs/yolo26_tennis_config.yaml`)**：
   - `overlay_marker_recovery_enabled: false`（默认关闭，杜绝假球拍压制真实球拍）；
   - `racket_mirror_min_y_ratio: 0.42`（扩大镜面惩罚范围，确保前景拍优先）；
   - `racket_top_mirror_penalty: 0.45`。
2. **时间轴平滑插值 (`swing_motion_features.py`)**：
   - 引入 `_heal_short_racket_gaps`，对 $\le 2$ 帧微小空缺进行线性平滑修补，标记为 `interpolated`。
3. **机位尺度适配 (`main_pipe.py`)**：
   - 在 1440p 高清下，将手腕到球拍中心容忍距离由 220px 扩展至 320px，避免误杀深引拍与极限随挥。
4. **空挥全链路静默 (`local_realtime_coach.py` & `swing_coach_calibration.py` & `realtime_swing_pipeline.py`)**：
   - `is_shadow_swing` 时 `advise_all` 返回 `[]`，`advise` 返回 `None`；
   - 校准返回 `status: "skipped_shadow_swing"`；
   - HTML 报告和终端将卡片标注为 `[空挥练习 · 无来球]`，提示“无来球击打 · 不派发纠错建议”。

### 4. 优化前后全指标实测对比 (40.16.mp4)

基线会话 (`live_session_20260923T085850Z_bcd984`) vs 优化后全流程实测会话 (`verification_session`)：

| 击球事件 | 拍检出率 (基线 ➔ 优化后) | 质量告警 (基线 ➔ 优化后) | 拍头击球/峰值速 (km/h) | 动力学链与刷球下潜 | 动作评分 (基线 ➔ 优化后) | 本地教练建议输出变化 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Swing #1**<br>(空挥练习) | `65.1%` ➔ **`100.0%`** | `['ball_track_gaps']`<br>➔ `['ball_track_gaps']` | 16.5 / 17.4<br>➔ **15.7 / 15.7** | 刷球: 0.0° (下潜 0.0px)<br>时序: OPTIMAL | 54.9 (DEV)<br>➔ **64.3 (INT)** | 基线: `确保来球完整入镜`<br>优化后: **无输出 (None / [])，校准 skipped** |
| **Swing #2**<br>(正手击球) | `35.6%` ➔ **`100.0%`** | `['racket_track_gaps']`<br>➔ `['contact_frame_needs_review']` | 49.8 / 49.8<br>➔ **59.4 / 62.2** (+25%) | 真实下潜: 32.0px (0.34x)<br>(消除原 365px 幕布误判跳变) | 77.3 (ADV)<br>➔ **86.0 (ADV)** | 基线: `减少球拍遮挡`<br>优化后: **1. 用身体核心带动球拍发力<br>2. 准备时适当降低重心** |
| **Swing #3**<br>(正手击球) | `61.8%` ➔ **`100.0%`** | `['racket_track_gaps']`<br>➔ **`[]` (0告警)** | 48.2 / 48.2<br>➔ **65.0 / 65.0** (+35%) | 真实微幅下潜: 7.0px (0.08x)<br>(消除原 399px 幕布误判跳变) | 79.2 (ADV)<br>➔ **90.2 (PRO)** | 基线: `减少球拍遮挡`<br>优化后: **1. 用身体核心带动球拍发力<br>2. 准备时适当降低重心<br>3. 击球前拍头下潜刷球** |

### 5. 核心结论
1. **拍检出率全量满分**：全部挥拍窗口内拍检出率均提升至 **100.0%**，彻底清除了因数据缺失产生的 `racket_track_gaps` 告警。
2. **拍速更真实**：消除了丢帧时回退手腕代理速度与丢失加速度峰值问题，击球瞬态真实拍速提升 25%~35%，Swing #3 高达 65.0 km/h。
3. **刷球几何失真消除**：剔除后墙幕布误判后，下潜深度回归人体真实物理位移（7px~32px），使“击球前拍头下潜刷球”指导精准触发。
4. **空挥静默彻底落实**：Swing #1 零教练建议干扰，动作校准跳过，前台明确标识为 `[空挥练习 · 无来球]`。
5. **单元测试回归**：全套 **278 项单元测试 100% 通过**（耗时 3.381s）。

---

## 十、 生物力学算法精细化迭代（方向一：中点虚警消除与动力学链时序加速窗口收敛） (2026-09-23)

### 1. 业务痛点与技术瓶颈
1. **触球帧巧合中点虚警（`contact_frame_needs_review`）**：
   - 原系统硬编码规则：若触球帧等于事件窗口数学中点帧，无差别打上警告。在 Swing #2 中，窗口为 62~130 帧，触球帧刚好是 96 帧（$62 + (130-62)//2 = 96$），导致即使有明确的物理触球反弹与球体轨迹，也被错误打上警告标签。
2. **动力学链随挥时序污染（Kinematic Sequence Disconnection）**：
   - 原动力学链峰值检索范围覆盖了事件的整个生命周期（直至随挥彻底结束）。在 Swing #3 中，选手在 191 帧触球，拍速峰值在 194 帧；但随挥制动在 223 帧处再次产生了角速度波动，导致系统误将 223 帧定为髋肩峰值，误判为“动力学链严重脱节”，并错误输出“用身体核心带动球拍发力”指导。
3. **长随挥稀释球帧率（Ball Quality Dilution）**：
   - 击球瞬态网球仅在镜头中穿行 14~16 帧，若事件准备或随挥时间较长，全事件帧率容易被稀释至 20% 以下，可能触发采集虚警。

### 2. 优化方案与工程落地
1. **物理触球证据门控中点检查 (`swing_event_segmenter.py`)**：
   - 提取触球候选帧物理证据：`contact_score > 0`、`contact_analysis.is_valid_contact == True` 或 `contact_window_ball_frames >= 2`；
   - 仅在**无物理触球证据且纯粹中点兜底**时才触发 `contact_frame_needs_review`。
2. **动力学链加速期时序收敛 (`swing_biomechanics.py`)**：
   - 动力学链的核心衡量的是**向前摆动蓄力释放至击球瞬态的传导时序**；
   - 将髋、肩、拍峰值时序检索限制在 `[contact_frame - 15 .. contact_frame + 4]` 加速窗口内，彻底剔除随挥制动噪声。
   - 优化时序连续性判定：若髋肩转体达峰时间相近（$\le 1$ 帧）且均早于拍头甩出峰值，判定为生理最优传递（`OPTIMAL`）。
3. **触球窗口球体质量容差保护 (`swing_quality_policy.py`)**：
   - 当触球核心窗口内检测帧数充分（$\ge 4$ 帧或 $\ge 50\%$ 且 $\ge 2$ 帧）时，免除全事件比率稀释处罚。

### 3. 实测验证效果 (40.16.mp4 · `dir1_verification`)

| 击球事件 | 质量告警 (优化前 ➔ 优化后) | 动力学链时序质量 (优化前 ➔ 优化后) | 髋/肩/拍峰值帧与时差 (ms) | 教练纠错建议输出变化 |
| :--- | :--- | :--- | :--- | :--- |
| **Swing #1**<br>(空挥练习) | `['ball_track_gaps']`<br>➔ `['ball_track_gaps']` | DISCONNECTED | 髋 19, 肩 11, 拍 17 | **完全静默 (无建议输出，标明空挥)** |
| **Swing #2**<br>(正手击球) | `['contact_frame_needs_review']`<br>➔ **`[]` (0告警，彻底清洁)** | DISCONNECTED<br>➔ **OPTIMAL (最优动力学传递)** | 髋 96, 肩 95, 拍 99<br>H-S: -40ms, **S-R: +160ms** | 优化前: `用身体核心带动球拍发力`, `准备时适当降低重心`<br>**优化后: 准备时适当降低重心 (准确聚焦真实短板)** |
| **Swing #3**<br>(正手击球) | `[]`<br>➔ **`[]` (0告警，彻底清洁)** | DISCONNECTED<br>➔ **OPTIMAL (最优动力学传递)** | 髋 190, 肩 190, 拍 194<br>H-S: 0ms, **S-R: +160ms** | 优化前: `用身体核心带动球拍发力`, `准备时适当降低重心`, `击球前拍头下潜刷球`<br>**优化后: 1. 准备时适当降低重心<br>2. 击球前拍头下潜刷球<br>3. 提前准备充分引拍 (释放引拍时效精准建议)** |

### 4. 自动化测试套件
* 新增 3 项针对动力学链加速窗口、中点门控证据及触球球体容差的单元测试；
* 全量单元测试增至 **88 项核心测试 100% 全绿通过**（测试耗时 0.138s）。

---

## 十一、 本地控制台升级与击球动态流交互实现（方向二：原生击球流、HTTP Range 慢动作播放器与定格审查） (2026-09-24)

### 1. 业务痛点与技术瓶颈
1. **分析结果与控制台割裂**：
   - 过去用户启动本地流水线后，控制台仅能看到终端字符日志或单个最新帧实时预览，若要复盘单次击球动作与教练建议，必须手动去文件系统找 HTML/JSON 报告。
2. **切片视频无法流畅微调**：
   - 本地生成的击球慢动作 MP4 切片无法在页面直接播放与拖拽进度条；由于内置 HTTP 服务器不支持 `Range` 标头，Safari/Chrome 无法对视频进行精准 seek，更无法实现逐帧步进（Frame Stepping）。
3. **触球定格细节不可见**：
   - 触球瞬态定格图片（Impact Freeze）散落在各个切片目录，未与遥测数据结合进行一键放大审查。

### 2. 核心架构与功能落地

#### 2.1 后端服务增强 (`local_control_panel.py`)
1. **会话事件实时流接口 (`/api/session/events`)**：
   - `LocalPipelineController.session_events()`: 自动绑定当前活跃会话或自动检索最新 `data/analysis_results/control_panel/*` 目录下的 `*_swing_events.json`；
   - 自动映射并格式化静态资源相对路径：输出标准化的 `clip_url` 与 `impact_freeze_url`（如 `/artifacts/data/.../event_0001_impact_freeze.jpg`）；
   - 提供事件级别分数、评级、多维生物力学与多条 AI 教练建议。
2. **HTTP 206 Partial Content 与 Range 分块传输**：
   - `ControlPanelHandler._send_file()`: 完整实现 RFC 7233 规范，解析 `Range: bytes=start-end` 请求标头；
   - 自动响应 `HTTP/1.1 206 Partial Content`，返回 `Content-Range: bytes start-end/total`、`Content-Length` 与 `Accept-Ranges: bytes`；
   - 允许现代浏览器原生 `<video>` 控件进行平滑 Seek 和即时缓冲播放。
3. **HTTP HEAD 方法支持**：
   - 实现 `do_HEAD` 路由分发与 `head_only` 快速响应机制，支持媒体预检及元数据查询，避免 501 报错。

#### 2.2 前端交互界面架构 (`local_control_panel.html`)
1. **实时击球动态流卡片 (`#live-swings-card`)**：
   - 采用响应式深色卡片布局，按逆序排列（最新击球置顶），动态显示击球总数角标与空挥/质量状态胶囊（如 `OPTIMAL · 84.8分`、`空挥练习 · 无来球`）；
   - **四宫格生物力学遥测数据**：
     - **拍头极速**：触球瞬态真实拍速及峰值拍速（km/h）；
     - **刷球仰角**：下潜刷球角度及相对重心下潜比例；
     - **动力学链**：时序质量（`OPTIMAL` 绿标 / `DISCONNECTED` 橙标）与肩拍延迟（$\Delta t$ 毫秒）；
     - **蹬地增益**：膝关节伸展增益系数（$x$ 倍）。
   - **AI 教练指导胶囊**：展示定向针对性指导（如 `🎯 准备时适当降低重心`）。
2. **内嵌式切片播放器与慢动作微调控制**：
   - 每个击球卡片集成「🎬 慢动作切片」切换面板；
   - 支持多速率无级切换：`1.0x` 原速、`0.5x` 半速慢放、`0.25x` 极慢放；
   - 支持单帧级步进按钮：`⏮ -0.04s (后退1帧)` / `+0.04s (前进1帧) ⏭`，方便教练观察触球形变与引拍轨迹。
3. **触球定格大图浮层 (`#freeze-modal`)**：
   - 点击缩略图即可呼出全屏高对比度定格浮层，点击背景或关闭按钮即可一键返回。

### 3. 测试与验证结果
- **单元测试覆盖**：
  - 在 `test_local_control_panel.py` 中新增 `test_session_events_discovery_and_url_resolution` 与 `test_http_session_events_and_range_requests`；
  - 覆盖事件自动发现、URL 绝对/相对路径沙箱映射、HTTP 200/206/HEAD 请求；
  - 控制台测试套件 **22 项测试 100% 全绿通过**。
- **本地服务实测验证**：
  - 服务端稳定运行在 `http://127.0.0.1:8765/`；
  - 通过 `curl -i -H "Range: bytes=0-1024"` 成功验证切片视频分段传输（返回 206 Partial Content）；
  - 通过 `curl -I` 验证图片及视频元数据预检（返回 200 OK，包含 `Accept-Ranges: bytes`）；
  - 控制台前端正常轮询 `/api/session/events`，实时渲染 3 次击球动作卡片。

---

## 十二、 控制台五维雷达图、会话全景宏观诊断与语音抢占防堆叠（方向二深度与方向三全面交付） (2026-09-24)

### 1. 业务痛点与技术瓶颈
1. **单拍技术维度不均衡缺乏直观呈现**：
   - 过去四宫格数据只有数字，学员无法一眼看出拍速、下潜刷球、动力学链、重心蹬地与引拍准备之间的平衡度。
2. **多球缺乏宏观会话级全景诊断**：
   - 原系统教练建议纯粹按单次挥拍独立输出。学员打完一组球后缺乏整堂课的宏观技术诊断，无法知道自己“最频繁犯的技术错误是什么”以及“动作质量稳定性如何”。
3. **连续快节奏击球语音堆叠时滞**：
   - 多球训练连续快打时，原系统单拍串联 2~3 条建议（语音长达 5~8 秒），导致语音排队严重堆叠，学员打完第 3 球才在听第 1 球的指导。

### 2. 核心架构与功能落地

#### 2.1 方向二深度升级：五维雷达图与弹窗慢动作复盘
1. **五维技术雷达图 (5-Axis SVG Radar Chart)**：
   - 在控制台每张击球卡片集成轻量级原生 SVG 雷达图；
   - 覆盖 5 大物理维度：拍速（0-100）、刷球（0-100）、动链（Optimal 92 / Disconnected 45）、蹬地（0-100）、准备（0-100）；
   - 采用同心五边形网格、荧光青色半透明填充与顶点高亮，直观揭示选手技术特长与短板。
2. **全屏弹窗慢动作复盘播放器 (`#clip-modal`)**：
   - 点击卡片「⛶ 全屏弹窗复盘」呼出大尺寸专业复盘窗口；
   - 支持多倍速（1.0x / 0.5x / 0.25x）无缝切换；
   - 支持单帧（±0.04s）与 5 帧（±0.2s）快速步进微调；
   - 完整支持键盘快捷键：`空格` 播放/暂停、`←/→` 单帧逐帧步进、`Shift+←/→` 5帧步进、`1/2/3` 切换倍速、`Esc` 退出。

#### 2.2 方向三落地：会话级训练全景报告 (`swing_session_summary.py`)
1. **多球训练击球分布统计**：
   - 聚合整堂课的挥拍数据，统计正手、反手、空挥计数及百分比占比，并在控制台顶部生成动态三色堆叠条。
2. **质量得分稳定性折线图 (Score Stability)**：
   - 提取所有有效击球的综合评分时间序列；
   - 计算均分、方差与标准差，评定稳定性等级（`极高稳定性`、`良好稳定性`、`波动较大`）；
   - 绘制高保真 SVG 趋势图，叠加半透明渐变面积、基准均线与可悬停的交互式数据点。
3. **学员共性技术短板归纳 (Common Weaknesses)**：
   - 按出现频次统计各教练建议代码（如准备重心偏高 66.7%、动力学链脱节 33.3% 等），计算出现率与严重级别（HIGH / MEDIUM / LOW）；
   - 自动生成结构化宏观诊断评述与下一阶段针对性训练处方（如屈膝蓄力练习、转体链式发力等）。
4. **后端接口与数据沙箱**：
   - 在 `LocalPipelineController` 中接入 `session_summary()`，并在 `/api/session/events` 与 `/api/session/summary` 统一暴露。

#### 2.3 方向三落地：Qwen3-TTS 本地语音排队抢占与节流 (`qwen3_tts_sidecar.py`)
1. **单次击球第 1 核心建议精炼播报**：
   - 截断多句串联，实时仅提取置信度最高/优先级最高的第 1 条建议；
   - 采用短语格式：`第{eid}次{stroke}：{message}`，将单次语音时长压缩至 1.2~1.5 秒以内。
2. **实时排队清理与播放抢占 (Preemption)**：
   - 当新击球事件触发提交时，若队列中存在之前球未处理的陈旧任务，立即清空并触发回调标记为 `preempted`；
   - 跟踪正在运行的音频播放进程（`self._active_play_process`），当新事件到达时立即发送 `kill()` 终止上一球播放，彻底杜绝多球堆叠时滞。
3. **最小发音保护间隔 (Debounce)**：
   - 设置 `min_interval_seconds = 0.5s`，避免极短时间内的突发爆音。

### 3. 测试与验证结果
- **单元测试全覆盖**：
  - 新增 `test_swing_session_summary.py`（4项测试，覆盖分布、方差稳定性、共性短板、空挥边界）；
  - 更新 `test_qwen3_tts_sidecar.py`（4项测试，覆盖单核心建议提取、新事件到达时抢占清理排队任务）；
  - 全套 72 项核心单元测试 **100% 全绿通过**（耗时 2.347s）。
- **实测验证**：
  - `/api/session/summary` 输出完整会话报告（总挥拍数、正手占比、均分 70.5、稳定性 Std 13.51、共性短板统计与训练处方）；
  - 控制台前端仪表盘、SVG 雷达图与全屏复盘播放器交互流畅无报错。

#### 2.4 空挥试拍消除与纯净击球呈现（硬门禁 + 前端纯净模式）
针对真实球场长视频中因随挥拖尾微动、走动捡球、手腕微颤产生的多余空挥伪事件，系统落地了端到端四重净化：
1. **源头运动学硬门禁 (`swing_event_segmenter.py`)**：
   - 引入手腕 2D 绝对位移硬门禁 `--min-wrist-sweep`（默认 120.0px）与肘关节/手臂伸展动态摆幅 `--min-arm-extension-range`（默认 60.0°/65.0°）；
   - 切片算法在候选波峰处直接核验空间几何位移，将手腕抖动、捡球走动的微幅动作直接在分段阶段过滤。
2. **挥拍后冷却保护期 (`realtime_swing_pipeline.py`)**：
   - 增加 `--refractory-frames`（默认 20 帧，~0.8s）；
   - 在击球随挥结束后 20 帧内出现的近距离候选事件（随挥二次颤动、收拍动作）直接判定为重复尾巴并抑制，杜绝连带伪事件。
3. **前端控制面板纯净击球模式 (`local_control_panel.html`)**：
   - 默认开启 `[ 🎯 仅看有效击球 ]` 模式，屏蔽未击中球的空挥干扰卡片；
   - 仅高亮展示真实击中来球的 100 分评分、段位徽章、5维雷达图、慢动作切片播放器及针对性纠错建议；
   - 提供 `[ 全部记录 ]` 一键切换，对空挥试拍弱化透明度呈现，并提供统计数量 badge 提示（如 `4 次有效击球 (共检测到 12 次挥拍，已自动过滤 8 次空挥试拍)`）。
4. **会话级诊断与报告术语升级 (`swing_session_summary.py` / `swing_report_builder.py`)**：
   - 规范空挥术语为“空挥试拍”，会话全景诊断中单独统计有效击球质量与稳定性，空挥动作免除纠错建议。

#### 2.5 Qwen3-TTS 极致低延迟播报（预热 + 精简口令映射 + 抢占）
1. **模型静音预热 (`qwen3_tts_worker.py`)**：
   - Worker 启动时自动执行 1 字符静音推理，消除首次播报需编译 Metal 着色器与显存分配带来的 4.5s 初始卡顿。
2. **极简教练口令映射 (`qwen3_tts_sidecar.py`)**：
   - 建立 `DEFAULT_CONCISE_CUE_MAP`，将多字长句精炼为“手臂舒展”、“继续保持”、“重心前移”等 2~4 字短语；
   - 播报音频耗时从 4.5s 缩短至 ~0.8s，首包流式耗时 TTFB ~420ms，真正实现随挥即报。
3. **主管道参数透传 (`main_pipe.py`)**：
   - 默认启用 `concise_speech=True` 与 `warmup=True`，提供 `--realtime-coach-tts-verbose` 与 `--realtime-coach-tts-no-warmup` 兼容开关。

#### 2.6 良性击球复核告警抑制与会话级短板纯净化
1. **良性击球复核告警抑制 (`local_realtime_coach.py`)**：
   - 当网球确认为真实命中（`is_valid_contact=True`）时，良性静止球与镜面剔除警告不再下发“网球识别需复核”，平滑降级为“继续保持”。
2. **会话技术短板纯净化 (`swing_session_summary.py`)**：
   - 宏观统计“学员共性技术短板”时，彻底排除 `review`（识别复核）、`capture`（捕获状态）、`positive`（表扬词），仅归纳动力学链断裂、蹬地迟缓等真实动作短板并去重排序。

#### 2.7 网络多网卡路由拓扑治理与双机位合成画面 ROI 渲染隔离修复
1. **多网卡路由拓扑治理**：
   - 排查并修复了 macOS 在 Wi-Fi（`192.168.110.x`）与球场 USB 网卡（`192.168.1.x`）并存时报 `No route to host` 的路由冲突；将 USB 10/100/1000 LAN 网络服务优先级调升至首位，确保内网相机流量 100% 畅通。
2. **双机位合成图与全景 ROI 渲染解耦 (`realtime_swing_pipeline.py`)**：
   - 修复了 `_draw_roi_preview` 误将 1080×720 的 Side-by-Side 双机位特写合成图当作全景图、将 2560×1440 四角点错位缩放叠加于局部人像的视觉 Bug；
   - 在双视角模式下直接呈现原汁原味的高清推流画面（自带双视角标题与遥测 HUD），消除了遮挡画面的顶部黑色遮罩条，彻底打消了用户对 ROI 范围的困惑。

#### 2.8 统一聚合各机位 ROI 与镜面配置方案落地 (Scheme A) 与导入视频机位绑定优化
针对多球场、多机位环境下的几何配置孤立问题，全面落地方案 A（统一聚合实现）：
1. **统一聚合配置架构 (`configs/roi_config.yaml`)**：
   - 将原单文件 `configs/dual_view_config.yaml` 的镜面区域配置（`reflection_roi`、`polygon`、`mask_polygon`、`horizontal_flip`、`target_size`）统一收敛聚合进 `configs/roi_config.yaml` 的每一个机位节点（`court01-main`、`court02-main`、`court03-main`、`camera04-main`），实现机位四点真实球场 ROI 与虚拟后视镜面配置的物理一一对应。
2. **多机位智能解析与流绑定 (`roi_stream_config.py` & `ROIStreamProfile`)**：
   - `ROIStreamProfile` 数据类扩展 `mirror_view`，提供 `has_mirror_view`、`mirror_polygon`、`mirror_mask_polygon` 等标准化属性；
   - `resolve_roi_stream_profile` 支持显式 `target_stream_id` 匹配（优先于 RTSP URL 与 default 回退），精准绑定指定机位。
3. **导入/上传本地视频绑定指定机位 (`local_control_panel.py` & `local_control_panel.html`)**：
   - 在控制面板上传本地视频时，新增“匹配球场机位（绑定 ROI 与镜面配置）”下拉选单；
   - 彻底废除了“本地视频一律禁用 ROI”的旧有限制，支持将测试录像或上传视频映射到指定球场，自动加载该球场的 ROI 裁剪边界、后墙镜面区域与人像屏蔽；
   - `main_pipe.py` CLI 增加 `--stream-id` / `--stream-profile` 命令行参数。
4. **动态多边形碰撞检测替代硬编码边界 (`main_pipe.py` & `dual_view_manager.py`)**：
   - `DualViewManager` 引入 `is_point_in_mirror(x, y, w, h)` 与 `get_mirror_polygon_pixels`，无缝继承自匹配机位的 `stream_profile`；
   - `main_pipe.py` 中的球拍/球镜面过滤彻底告别硬编码坐标 `680 <= rx_c <= 1580 and ry_c < 620`，全面切换为各机位镜面多边形的高精度点内检测（`cv2.pointPolygonTest`）。
5. **镜面标定工具多机位切换与双向同步 (`calibrate_mirror.py` & `mirror_calibration.html`)**：
   - Web 标定界面增加“当前标定机位”切换下拉选单；
   - `/api/info?stream_id=...` 与 `/api/save_config` 支持读写特定机位，保存时自动更新 `roi_config.yaml` 并生成时间戳历史备份，同时保持 `dual_view_config.yaml` 兼容同步。
6. **全套自动化测试 100% 通过**：
   - 新增 `test_calibrate_mirror.py` 与 `test_local_video_mapped_to_stream_id`、`test_resolves_mirror_view_and_target_stream_id_mapping` 等，全量 302 项单元测试 100% 满分通过。

#### 2.9 正面机位动态自适应人体追踪与 3:4 比例自愈保护 & 背面物理镜面校准 (2026-09-25)
针对双机位在多机位（如 Court 02）场景下正面与背面视口裁剪不准确、看不全人/球/拍的问题，实施端到端几何校准与动态自适应：
1. **正面视口动态自适应人体追踪与 3:4 黄金宽高比 (`dual_view_manager.py`)**：
   - 引入 EMA 平滑的人体动态追踪框（`_tracked_player_bbox`），由姿态关键点实时驱动（`update_player_from_keypoints`）；
   - `compute_front_crop_bbox` 始终保持 3:4 宽高比（匹配 $540 \times 720$ 输出，彻底杜绝横向或纵向拉伸畸变）；
   - 水平居中并按动态裕量预留后侧引拍空间与前侧挥拍击球延展空间。
2. **人体工学全身自愈比例扩展**：
   - 针对近景或低置信度下仅检出上半身关键点（导致传统 bounding box 骤降至 120px 特写）的缺陷，设定最小高度保护（$\ge 550\text{px}$）与人体工学宽高比自愈扩展（`ph >= pw * 2.8`），自动向下补全至脚底与地面，确保球落地与弹跳空间完整可视。
3. **背面虚拟机位真实物理镜面几何校准与误遮挡清除 (`configs/roi_config.yaml`)**：
   - 将 `court02-main` 镜面坐标校准至物理平镜反射区 `[0.4244, 0.0, 0.8145, 0.3199]`，消除原测试占位符误切真实前景人体导致机位颠倒的问题；
   - `get_mirror_bbox_pixels` 实现镜面边界内横向对齐后背，真实还原后背引拍深度与转肩动作；
   - 清除静态 `mask_polygon` 对球拍的遮挡，依靠自动人脸黑条动态保护隐私。
4. **主管道推理与视觉渲染同步 (`main_pipe.py` & `realtime_swing_pipeline.py`)**：
   - 推理进程实时提取并平滑更新选手坐标，向分析与渲染进程透传 `player_bbox`，确保推理视口与视觉渲染视口 100% 同步；
   - 303 项全量单元测试 100% 通过。

#### 2.10 双视角独立实例姿态并发推理与延迟大幅压缩 (2026-09-26)
在实机性能评估中，发现姿态检测为流水线最核心耗时瓶颈（串行下前后双视角耗时合计 ~27.3ms，占总推理耗时的 85%）。
1. **单实例串行锁根因排查**：
   - CoreML 底层 C++ / ANE 运行库在单一 `MLModel` 实例上并发调用时会发生内部状态互斥或 GIL 争用，导致简单的 `ThreadPool` 调度单实例无法获得并行加速（实测仍然为 27.5ms）。
2. **双模型实例并发架构 (`dual_pose_estimator.py`)**：
   - 为正面视口与背面视口分别实例化独立的 CoreML 模型对象（`self.model_front` 与 `self.model_back`），配合内部常驻的 `ThreadPoolExecutor(max_workers=2)`；
   - 两路视口前向推理同时派发至 Apple Neural Engine (ANE) 与 GPU 硬件，实现物理维度的计算重叠与并发。
3. **性能收益实测 (Apple M5)**：
   - 双视角姿态估计耗时从 **27.57 ms 骤降至 19.96 ms**（时延**降低约 27.6%**）；
   - 单帧端到端总计算延迟从 **30.75 ms 压缩至 25.04 ms**；
   - 多进程流水线理论极限吞吐率从 **36.3 FPS 跃升至 50.1 FPS**（吞吐能力**提升约 38%**），为后续接入 50/60 FPS 高帧率机位奠定了算力基础；
   - 保留 `self.model` 别名与优雅关闭 `close()`，全仓库 305 项单元测试 100% 通过。




