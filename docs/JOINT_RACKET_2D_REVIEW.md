# 2D肢体与七点球拍联合分析

2026-10-20。用户明确参考图基于3D pose；当前实时可用数据以2D人体、主球与球拍七点为主。本工具从同源保存日志离线生成联合复核，尚未接入实时HUD；不增加推理调用，也不宣称实时开销已经验收。

## 指标与轨迹

`joint_racket_2d_analysis.py` 是不加载模型的计算模块。原始源像素中分别保存真人/镜像的12个肢体点、7个球拍点，绘制肩—肘—腕—握把—甜区—拍头联系及历史轨迹。

- 当帧、observed、非镜面恢复、分数≥0.5且坐标有限的肢体点才进入观测。
- 球拍候选分数≥0.25、握把分数≥0.5；每视角选距合格手腕最近的握把，门限为图像对角线0.05倍。第二候选与第一候选距离差不足对角线0.005倍时拒绝。一个候选竞争真人/镜像两侧时两侧都拒绝。这些是候选诊断参数，未经独立准确率验证。
- 人物归属始终标为候选，不修改Shadow业务选择；源图与镜像源坐标分别绘制，不额外翻转或融合。
- 肘膝角、髋肩连线像面夹角、拍杆像面角、腕握距离/本视角肩宽可作为二维几何观测。
- 右腕与甜区速度使用连续源帧、合格点和媒体PTS。小于等于局部中位间隔四分之一或大于三倍间隔时拒绝差分。没有插值、缺值置零、平滑补点或强制峰值排序。
- 真人主球—甜区像面距离只提供邻近程度；镜像不复用真人主球。最小像面距离不能确认触球。
- 速度是px/s，关节夹角不是3D角，髋部抬升不能命名为蹬地功率。右臂完整链 `complete.arm_chain` 指右肩/右肘/右腕；左臂单独统计。

## 连续至少3帧不可用

统计每个点、完整肢体链及完整七点球拍的源帧连续缺失。原始球拍候选为空、握把归属失败和单点低置信分开记录。即使画面有拍，归属不明确也可能无法测量。

CSV保留session、track、首末源帧、连续数量、首末PTS、PTS跨度、原因、事件重叠与触球候选±0.12秒重叠。恰好3帧另有计数。各点及完整链的记录会重叠，不能将所有行数相加解释为独立缺帧次数。

源日志缺帧另按审查中ffprobe已解码的源帧数检查，包括首尾缺失。缺少源日志的区间不会与前后丢点拼接。PTS跨度为末帧时间减首帧时间；例如25Hz连续3帧的首末跨度80ms，3个帧槽为120ms。两者不能混用。事件外球不在画面可能正常，不把覆盖率当召回率。

## 人工触球标识

界面支持逐帧、慢放、真人/镜像、轨迹与无覆盖原图。模型候选帧保持不变；人工结果另存：

- `unreviewed`：待审核；`unknown`：看不清；`not_applicable`：没有触球。
- `interval`：仅能定位可能区间，不声称确认单帧。
- `confirmed`：由审核人明确设置锚点，必须在人工区间内，并填写可见事实与证据视角。

浏览器可保存/导入/导出草稿；点击“保存到本机新收据”后服务端校验源哈希、事件ID、帧号、PTS、审核人和判断，在产物目录 `human_reviews/` 创建不可变收据。confirmed锚点的PTS由服务端从已绑定源数据推得。模型事件不被覆写，人工确认不自动授予教练规则或测量准确性。

本工具使用新的 `tennis.joint-racket-contact-review.v1` schema，不直接提交旧的V3 Coach API；目前不把新收据自动导入V3或用于重算生产事件。需要重算时，应另建人工结果分支并显式选择人工区间，保留原锚点。

## 现有km/h公式推导

设地面单应性为H，源像素齐次点为pᵢ=(uᵢ,vᵢ,1)ᵀ，齐次归一化π(a,b,c)=(a/c,b/c)：

$$g_i=\pi(Hp_i).$$

相机中心C=(Cₓ,Cᵧ,Hc)，物点高度hᵢ，假定针孔相机、水平地面。物点处于C与地面交点gᵢ的同一射线上。由高度分量：

$$H_c+\lambda_i(0-H_c)=h_i \quad\Rightarrow\quad \lambda_i=1-\frac{h_i}{H_c}.$$

因而物点的水平坐标为：

$$q_i=C_{xy}+\left(1-\frac{h_i}{H_c}\right)(g_i-C_{xy}).$$

若具备各帧真实高度和相机位置，三维端点位移的平均速率估计为：

$$v_{3D}\approx3.6\frac{\sqrt{\|q_1-q_0\|^2+(h_1-h_0)^2}}{t_1-t_0}\quad[\mathrm{km/h}].$$

这仍是端点弦长/时间，不是弯曲轨迹弧长的精确平均速率。当前实现只有水平投影，并对两个端点使用**同一个当前估计高度**：

$$\widehat h=\mathrm{clip}\!\left(1.40\frac{y_{ground}-y_{racket}}{y_{ground}-y_{shoulder}},0.15,2.70\right).$$

其中地面y取可用脚踝最大y、肩y取平均。图像脚踝—肩垂直距离不足25px或缺点时高度不可估，可能退回地面投影。默认相机高2.40m，Cxy=(0,0)。实际代码缩放α为：

$$\alpha=\begin{cases}1,&\widehat h\le0\text{ 或 }\widehat h\ge H_c\text{ 或 }H_c\le0\\\max(0.20,1-\widehat h/H_c),&\text{其他}\end{cases}.$$

因此当前同高度实现：

$$\widehat v_{raw}=3.6\alpha\frac{\|g_1-g_0\|}{\Delta t}.$$

换算位移>4m或速率>90m/s先拒绝，剩余速度用固定半径R=1.20m、向心加速度限值a=1050m/s²限幅：

$$\widehat v_{out}=3.6\min(\widehat v_{raw}/3.6,\sqrt{aR})\approx\min(\widehat v_{raw},127.8).$$

地面H不能独立确定空中物点的高度与三维相机参数。上述肩高、相机高、半径、限值均为代码假设；同高度忽略垂直运动，限幅也不能证明数值真实。当前拍点还主要是框中心。因此不把该输出命名为真实拍头速度，不用于疲劳诊断。

## 运行

从冻结的前次审查输入建立一个新目录，不覆盖已有产物：

```sh
venv_yolo26/bin/python scripts/build_joint_racket_2d_review.py \
  --audit data/analysis_results/data_chain_audit_20261020/audit.json \
  --output data/analysis_results/joint_racket_2d_review_20261020_v2
venv_yolo26/bin/python scripts/serve_joint_racket_2d_review.py \
  --directory data/analysis_results/joint_racket_2d_review_20261020_v2 --port 8770
```

打开 http://127.0.0.1:8770/ 。源视频按需解码，24帧内存缓存，不复制1250张图片。服务仅监听本机。停止服务后页面不能继续解码原图；按上述命令重启即可。

验证：`venv_yolo26/bin/python -m unittest test_joint_racket_2d_analysis -v`。
# 2026-10-20 速度估算参数调整

动作特征提取默认使用交互示意中选定的相机高度 **3.10 m**、同高度拍点假设 **1.20 m**。
缩放系数为 `1 - 1.20 / 3.10 = 0.612903`。示例地面投影位移 1.15 m、两帧间隔
39 ms，对应 `3.6 × 0.612903 × 1.15 / 0.039 = 65.06 km/h`。
1.15 m 和 39 ms 仅为示例，运行时仍使用各帧实际投影位移和源视频时间差。

`extract_motion_features` 的 `camera_height_m` 和 `assumed_racket_height_m`
允许覆盖默认参数。逐帧 `racket_height_m` 优先；无逐帧高度时采用 1.20 m 假设。
显式传入 `assumed_racket_height_m=None` 恢复先前的 2D 姿态高度推断，传入 0
可检查地面投影速度。输出新增 `racket_height_source`、
`racket_speed_camera_height_m`、`racket_speed_is_geometric_estimate`，供复核假设来源。
20% 修正下限、约 127.8 km/h 封顶及异常位移拒绝规则沿用现有实现。
本次只改变重新分析的计算；已保存的历史分析和复核页面数据需重新生成后才会更新。

# 2026-11-04 双视图双向合并与运动学速度曲线重构方法

针对单视角下肢体动作曲线断线、触球瞬间甜区速度丢失以及单帧跳变尖峰压扁纵轴的问题，在不破坏原始单视角观测契约（`metrics` 与 `tracks` 保持原样）的前提下，新增 `dual_view_fusion` 双向合并层与运动学曲线重建流水线。

## 1. 跨视角躯干平面仿射与侧身共线门控 (`mirror_geometry.py`)

在 [`estimate_cross_view_torso_transform`](../mirror_geometry.py) 中建立 `Front <-> Mirror Back` 的双向 2D 平面仿射变换 $p_{\text{dst}} = A p_{\text{src}} + t$：

- **4 点最优化拟合（Optimal）**：当双肩与双髋 4 个躯干锚点在双视角均可见，且双视角横向宽度 $\ge 15\text{ px}$、纵向脊柱高度 $\ge 25\text{ px}$ 时，求解最小二乘仿射矩阵 $A \in \mathbb{R}^{2\times 2}, t \in \mathbb{R}^2$。
- **侧身共线与剪切门控（Shear & Spine Gate）**：当球员处于极致侧身转体（Unit Turn，如 `f=180..182`）时，双肩与双髋近似共线于同一竖直平面，无约束 6-DOF 仿射会产生病态纵向剪切（例如 $s_y \to 2.89$）。为此引入四项几何门控：
  1. 手性行列式：$\det(A) > 0$（全景平镜反射保持图像平面同向手性）；
  2. 拟合残差：$\text{RMSE} < 25.0\text{ px}$；
  3. 剪切门控：$|a_{01}| < 0.45 s_x$ 且 $|a_{10}| < 0.45 s_y$；
  4. 脊柱比例门控：$0.65 r_{\text{spine}} \le s_y \le 1.45 r_{\text{spine}}$，其中 $r_{\text{spine}} = H_{\text{dst}} / H_{\text{src}}$ 为双视角脊柱高度比。
- **各向异性脊柱回退（Anisotropic Spine Fallback）**：一旦任一门控不满足或锚点不足 4 个，立即回退到以躯干质心对齐、纵向采用脊柱比 $s_y = \text{clip}(r_{\text{spine}}, 0.35, 2.8)$、横向采用相机俯角透视比 $s_x = 1.295 s_y$（`back -> front`）或 $s_x = 0.772 s_y$（`front -> back`）的对角刚体缩放模型。

## 2. 四级双向肢体与 7 点球拍合并流水线 (`joint_racket_2d_analysis.py`)

在 [`merge_dual_view_frame`](../joint_racket_2d_analysis.py) 中按置信度优先级执行四级互补：

1. **双向肢体关节自愈（Bidirectional Limb Healing）**：
   - 遍历 `left/right_shoulder, elbow, wrist, hip, knee, ankle`。当某一视角关节缺失而对侧视角观测合格时，通过跨视角躯干仿射映射补全（标记为 `back_projected_to_front` 或 `front_projected_to_back`）。
   - 解决了镜像视角因球网横带与身体遮挡导致的左手腕（覆盖率从 `34.2% -> 94.2%`）和右手腕（`52.5% -> 96.8%`）断线问题。
2. **自愈手腕找回原生 7 点球拍候选（Healed-Wrist Native Candidate Rescue）**：
   - 许多帧 YOLO26-Pose 实际上检出了高质量的 7 点球拍候选框，仅因单视角手腕漏检而在关联阶段被判为 `no_qualified_wrist`。利用 Step 1 自愈后的手腕重新执行就近握把匹配（门限 `0.065 * diag`）与小臂共线平局决胜，在 5 个会话中直接找回 **83 帧** 全部 `v=2` 的原生 7 点球拍（标记为 `healed_wrist_candidate_rescue`）。
3. **跨视角预测导向弱候选找回（Cross-View Guided Weak Candidate Rescue）**：
   - 对置信度处于 `[0.15, 0.25)` 的弱球拍候选框，若其握把同时与本视角手腕及对侧视角投影过来的握把吻合，则予以激活（标记为 `relaxed_threshold_candidate_rescue`）。
4. **持拍手腕锚定跨视角球拍投影（Wrist-Anchored Cross-View Racket Projection）**：
   - 当某视角因身体遮挡完全无候选框（如正手引拍阶段真人挡住球拍，仅镜像可见）时，为消除 35° 俯角下躯干平面与伸展手臂平面之间的深度视差，**不直接使用躯干绝对平移映射球拍**，而是以目标视角持拍手腕 $w_{\text{dst}}$ 为锚点，仅对源视角球拍各点相对于源握点 $g_{\text{src}}$ 的偏移向量做各向异性缩放：
     $$\hat{p}_{\text{dst}}^{(k)} = w_{\text{dst}} + \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \left(p_{\text{src}}^{(k)} - g_{\text{src}}\right), \quad k \in \{0,\dots,6\}$$
   - 导出为 `v=1`（遮挡可见性标记）的跨视角补全球拍（标记为 `back_projected_to_front` / `front_projected_to_back`）。

## 3. 运动学速度曲线去噪与触球短缺口桥接 (`_compute_merged_kinematics`)

在 [`_compute_merged_kinematics`](../joint_racket_2d_analysis.py) 中对合并后的 `right_wrist` 与 `sweet_spot` 轨迹执行四步运动学重建，结果存入 `frame['dual_view_fusion']['kinematics'][view]`：

1. **单帧瞬移折返抑峰（1-Frame Teleportation Spike Suppression）**：
   - 扫描时间窗口内的最近邻有效观测点 $(i_{\text{prev}}, i, i_{\text{next}})$。若单步跳变 $\max(d_{\text{in}}, d_{\text{out}}) > \tau_{\text{spike}}$（手腕 $42\text{ px}$、甜区 $75\text{ px}$）且折返总路程满足：
     $$d(p_{i_{\text{prev}}}, p_i) + d(p_i, p_{i_{\text{next}}}) > 2.1 \max\big(10.0,\, d(p_{i_{\text{prev}}}, p_{i_{\text{next}}})\big)$$
   - 则判定 $p_i$ 为单帧误匹配尖峰（如 Session 0 `f=199` 的 `5262 px/s` 与 `f=10` 的 `2669 px/s`），使用源 PTS 时间权重 $\alpha = \Delta t_1 / (\Delta t_1 + \Delta t_2)$ 线性内插修正，并标记为 `spike_suppressed_interp`。
2. **高速动态模糊短缺口桥接（Motion-Blur Short-Gap Bridging）**：
   - 对触球前后因极高角速度动态模糊导致的残余短缺口（手腕 $\le 2$ 帧，球拍甜区 $\le 3$ 帧，如 `f=21..22`、`f=192`），在校验源日志帧号连续且两端总位移符合人体挥拍物理上限后，按源视频 PTS 时间戳插值桥接（标记为 `short_gap_bridged`），找回被抹除的触球鞭打主峰（`3470–3737 px/s`）。
3. **PTS 加权中心差分（PTS-Weighted Central Difference）**：
   - 源视频帧间隔在 `30ms / 41ms / 48ms` 之间交替波动，单步后向差分会将检测微抖放大为周期锯齿。连续有效帧统一采用中心差分：
     $$v_i = \frac{\|p_i - p_{i-1}\| + \|p_{i+1} - p_i\|}{t_{i+1} - t_{i-1}}$$
4. **三点零相位平滑（3-Point Zero-Phase Filter）**：
   - 对连续有效速度序列施加 `[0.25, 0.50, 0.25]` 零相位对称核平滑，不引入峰值相位滞后，完整保留“引拍掉拍头次峰 → 拍柄引导滞后谷值（Slot Lag） → 触球鞭打主峰”的动力链传递时序。

## 4. 复核面板与数据集挖掘命令

- **并排双视图拼接复核页面**（`scripts/joint_racket_2d_review.html`）：
  - 支持 `双视图并排拼接对比 (Front | Mirror Back)` 模式；
  - 勾选 `☑ 双视图合并补全 (Merged)` 后，主画布以青色/橙色虚线连接补全关节与球拍历史轨迹，底部 `#plot` 以实线绘制合并去噪速度曲线（青点=补全/桥接，粉点=抑峰，半透明虚线=原始单视角对照）。
- **重新构建复核数据与挖掘 YOLO26 训练样本**：
  ```sh
  venv_yolo26/bin/python scripts/build_joint_racket_2d_review.py \
    --audit data/analysis_results/data_chain_audit_20261020/audit.json \
    --output data/analysis_results/joint_racket_2d_review_20261020_v2

  venv_yolo26/bin/python scripts/mine_dual_view_dataset.py \
    --audit data/analysis_results/data_chain_audit_20261020/audit.json \
    --output data/analysis_results/dual_view_mined_20261020
  ```


