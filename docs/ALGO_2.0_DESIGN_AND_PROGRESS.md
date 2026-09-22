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
3. **基线数据集 3**：`Tennis-Vision` 中的基准测试片段（反手、双手反拍、侧身击球）。
   - 重点验证：正反手分类准确度、双手握拍识别率、抗侧身塌陷稳定性。全量 254 项自动化单元测试全绿通过。
