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

**状态图例**：  
- ⬜ 待开始 (Pending)  
- 🟡 进行中 (In Progress)  
- 🟢 已完成 (Done)  
- 🔴 遇到阻碍 (Blocked)

---

## 五、 测试与验收基线数据集

1. **基线视频 1**：`/Users/krum5539/Desktop/Camera/49.35.mp4`
   - 规格：2560x1440, 25/50fps, 10秒, 正手击球与后背引拍。
   - 重点验证：镜面区域切分、水平翻转、正面手腕引拍盲区的背面补全。
2. **基线数据集 2**：`Tennis-Vision` 中的基准测试片段（反手、双手反拍、侧身击球）。
   - 重点验证：正反手分类准确度、双手握拍识别率、抗侧身塌陷稳定性。
