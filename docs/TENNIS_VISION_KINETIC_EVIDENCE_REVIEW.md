# Tennis-Vision 历史迭代与动力链验证证据审查

日期：2026-10-04。只读审查 Tennis-Vision 源码及已保存运行报告；不重跑 GPU、不修改历史模型、标注或发布结果。本报告使用本地一手实现和实验记录，未将视觉顺眼、优化收敛、单测通过当作现实运动学准确性。

## 结论

Tennis-Vision 已形成较完整的来源绑定、原生模型回放、镜面对应审计、固定划分对照及失败保留机制，适合成为 tennis_analyzer 的离线核验平台。现有九段 SAM3D/MHR 重建不能直接成为“真实动力链”真值：相机/镜面未独立实测，原始时间采样存在不均匀间隔，髋肩参与镜面拟合，同系列视频和同模型误差相关，球拍大量帧由先验补足。下一轮优先产出 **髋—躯干峰值时序的独立误差基准**，随后才扩展手臂、球拍与教学判断。

48.43 的手—拍柄接触和拍面方向已有用户人工验收，必须尊重该限定验收，不要求重复确认；其范围不包括自动厘米测距、相机、人体三维和动力链时序。其余八段仍是候选。[验收记录说明](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MHR_FIT_PROGRESS.md:222)

## 两条算法线应分开看

### 球场与回合分析线

广播视频流程是 YOLO/ByteTrack 人员、TrackNet 三帧球热图、球场关键点/单应、事件候选、接触/落地分类、Viterbi 回合解码、轨迹与球飞行重建。它已经明确地把同一平面的投影、空中球、事件误差和人体姿态区分。球场自洽重投影不足以证明球场正确；曾有错误球场仍取得14/14 RANSAC内点。因此球场速度、回合3D和身体动力链不共用一种“准确率”。[技术总览](/Users/krum5539/Documents/Tennis-Vision/docs/public/TECHNICAL_OVERVIEW.md:1)

### 新视频人体、镜面、手—拍分析线

原片保留；统一成最高2560宽、25 fps输入；云端真人 SAM3D、镜中独立 SAM3D、双人物 SAM2传播；下载源视频和模型/源码哈希；镜面用二维关节配准；人体保持真人根节点，镜中只提供有界手部约束；球拍以原图轮廓、掌内握点、双侧轮廓和 SO(3) 时序损失拟合。SAM2 遮罩存在不证明镜中3D存在；反射纹理好看不证明关节几何准确。[双视角流程及产物](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MULTIVIEW_GPU_ITERATION.md:3)

镜中 SAM 在水平翻转图上推理，输出 X 还原；解剖 MHR 编号保留，不再交换左右手。YOLO COCO镜面对应的编号规则是另一层，不能混用这两种交换规则。[实现](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/generate_multiview.py:59)、[说明](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MULTIVIEW_GPU_ITERATION.md:24)

`multiview_constraints` 用反射后核心关节误差做门控，消除逐帧中位平移后比较形状；仅修正手指、保留手腕、躯干和根节点，权重最高0.2、单关节修正不超过8mm。这不是双视角三角化的人体躯干，也没有重新求解全身运动。[源码](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/multiview_constraints.py:19)

## 历史迭代里最应保留的证据

| 阶段 | 已验证结果 | 对动力链迭代的含义 |
| --- | --- | --- |
| 九段原生MHR回放 | 9/9回放一致，最大网格欧氏误差约1.02×10⁻⁶m，83个镜中缺帧保持缺失 | 只验证参数能复现保存的模型；微米级回放残差不是人体微米级精度 |
| v7→v9联合优化 | v7接触改善但留出15.24→35.47 canonical px、角加速度P95 4.73→16.63°/帧²；v8归一化修复后新人工保留误差29.26→7.63，但加速度P95仍4.73→9.24；v9未继续改善 | 损失、二维位置、接触、时序必须分项评估；不能凭某项变好发布整体准确 |
| v11镜中热点消融 | 两组120步完成，均拒绝；第183帧角加速度14.38→3.82，但旧保留P95 16.92→25.66 canonical px | 加强平滑或删除镜中观测可让曲线更顺，同时损伤真实点约束 |
| 镜面/相机审计v2 | 原生投影roundtrip中位约0.000057px；镜面人体核心保留中位15.15、P95 65.80原图px | 自一致残差可极小，跨证据误差仍很大 |
| 仅改镜面法向 | 三训练帧五点极线误差中位2.30→0.84px，但旧保留62.53→92.85、新保留13.89→19.08，拒绝替换 | 必须将标定点与验收点隔离，不能让髋肩自己校准镜面再证明自己 |
| 时轴修复 | 原始250帧→统一249帧，旧平均fps映射152/249不同；精确PTS全部249帧匹配实际FFmpeg滤镜 | 同帧编号不足以证明同一曝光；导数及峰值需实际PTS和重复源帧标记 |
| 单帧186冲突 | 固定握点三维候选真人中位22.11→3.07原图px，但柄底21.91、镜中32.54px；自由平移可到1.30px，却偏离原握点约85cm | 二维贴合与解剖/深度之间有强歧义，不能把自由度放开到贴合后称真值 |
| 固定9cm时序对照 | 新6帧真人中位13.68→10.59→9.86px，P95 34.19→29.00→39.22；握点尾部与角加速度也退步 | 用户参数是可控假设，不是独立测量；时序候选仍拒绝 |
| 八段批处理 | 1993帧、1989显示球拍、4隐藏；各段柄轴P95约39.76–71.42°；仅46/249个49.33帧为强观测，203为估计 | 可见球拍与完整播放不等于拍头速度可信；先验证髋肩，再扩展拍头 |

来源：[原生回放审计](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MULTIAGENT_GPU_ITERATION.md:20)、[v7–v9台账](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MHR_FIT_PROGRESS.md:50)、[v11对照](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MHR_FIT_PROGRESS.md:158)、[相机审计 JSON](/Users/krum5539/Documents/Tennis-Vision/output/offline_iterations/20261002_joint_evidence_review/mirror_camera_audit_v2/mirror_camera_report.json)、[单帧核查](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MHR_FIT_PROGRESS.md:183)、[固定9cm报告](/Users/krum5539/Documents/Tennis-Vision/output/offline_iterations/20261002_joint_evidence_review/fixed_grip_9cm_v1/REPORT.md:1)、[八段报告](/Users/krum5539/Documents/Tennis-Vision/output/offline_iterations/20261002_grip_collection_9cm_v1/REPORT.md:1)。canonical px与原图px不可在表外直接相互比较。

相机审计v2实际原片PTS间隔中位 **41.25ms**、P95 **49.92ms**、最大 **97.42ms**。早期文档记载的编码50fps不能当作真实均匀50Hz曝光；统一25fps也不会创造新观测。精确映射算法强制检查零起点、源PTS递增、统一PTS序列及EOF，拒绝不符合预期的归一化流程。[时轴算法](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/source_frame_alignment.py:24)、[实际统计](/Users/krum5539/Documents/Tennis-Vision/output/offline_iterations/20261002_joint_evidence_review/mirror_camera_audit_v2/mirror_camera_report.json)

当前镜面拟合使用真人SAM髋肩膝踝与镜中YOLO点，每第五帧留出；`measured_camera=False`。这验证了同视频模型之间的配准泛化，未建立物理相机真值。[拟合范围及留出](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/estimate_mirror.py:43)

地面/镜面四角拟合支持用户实测宽长及 ground-to-camera 变换，但焦距仍取模型预测、主点固定图像中心、未包含实测畸变，返回仍为 `paired_ground_estimate`、`measured_camera=False`。有地面尺寸不等于已完成相机标定。[四角标定实现](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/mirror_calibration.py:17)

## 下一轮最小实验与可验收产物

### 0. 先冻结数据与测量定义

用九段现有片作回归集，在任何新训练前冻结源视频SHA、attempt、模型/代码版本、标定版本、原曝光PTS与归一化映射、raw真人/镜中关节、缺失掩码、球拍观测和先验比例。历史看过或调过的片不能称全新盲测。调试集、标定集、最终新采集测试集分别建manifest，隔离脚本预测与评估；每个序列都保留拒绝和漏检分母。

先定义：骨盆轴向角速度、躯干轴向角速度、峰值时间、Δt=t躯干−t骨盆；球拍若加入，明确拍头线速度而不是检测框中心。两髋连线/两肩连线的三维方向变化同时包含身体倾斜，不等于绕身体纵轴的旋转。使用至少非共线的解剖点构造骨盆/躯干坐标系，固定左右/前后/上方向及角速度分量定义；真值和被测算法采用同一定义。旋转求导在统一三维坐标中计算，显示用平滑版本与测量用版本分别保存。

### 1. 使用现有九段做一次输入修复隔离对照

最小A/B只改精确PTS映射与重复帧处理，其余模型、镜面、训练/留出、滤波参数完全冻结；在新目录重生成与源帧相符的球拍/二维人体观测，比较旧输入和修复输入。报告每段重复/漏帧、接触帧变动、两视角髋肩曲线差、峰值稳定性、留出投影中位/P95、覆盖率。历史文档明确“时轴代码修复”没有证明“历史拟合收益”。此实验只能确定修复时轴对模型一致性的收益。[历史状态](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MHR_FIT_PROGRESS.md:175)

### 2. 一次独立标定与新测试采集

最小试行可先采同人同拍10–20次挥拍，覆盖慢/正常/快及至少两种朝向，作为确定流程可行性的pilot，不能据此宣称跨人群准确率。同步记录经过独立标定的动作捕捉或多相机参考、原始真人+镜面视频、同步闪光/时间信号；检查实际曝光PTS、重复帧和快门模糊。建议参考设备采用真正120–200Hz或更高采样，这是试行设计，不是当前已验证的合格门槛。

单独估计相机fx/fy/cx/cy、畸变、镜面法向/距离、地面上方向和尺度；用非人体标定板/固定空间参考拟合，再在未参与拟合的多个位置/深度的参考点验收。标定时不使用本次要评价的髋肩动态曲线。若用户提供的三维参考本身来自同一SAM3D镜面流程，则仍只能做重建一致性，不是独立真值。

### 3. 在同一盲测集做消融与真实误差表

至少对比：前视二维、镜中二维、未融合的各自SAM3D、标定后双视角三维、三维加时序模型。所有方案共用真值、事件窗口和缺失分母。记录每个挥拍的骨盆/躯干角速度曲线、峰值区间、Δt及参考值；报告峰值时间MAE/P95、Δt偏差/MAE/P95、可分辨样本的先后符号错误率、有效覆盖率/拒绝率。按挥拍或整段bootstrap，避免把连续帧视作独立样本；另列慢/快、朝向、遮挡分组。

从参考高采样原始序列按真实曝光时刻模拟25/50/100Hz采样，保持同一预先规定滤波策略，测峰值时间误差和符号误判。不能用25fps插帧产生的“100fps”评价高速采样收益。平滑强度以真值峰值偏移和尾部误差选择，不以曲线是否漂亮选择。

### 4. 从误差基准导出置信度和教学门控

现阶段证据分数只做来源质量/双视角一致性摘要。待独立数据到位，定义“|Δt估计−Δt参考|≤E”的可验证事件，在调参集建立成功概率校准，在新测试集画可靠性曲线、报告coverage与误差；E按教学所需时序精度和实测分辨率定，不预设成经验生理常数。峰值区间重叠、误差界跨越零、双视角不一致或采样不足时输出无法分辨；负Δt不能直接解释为不连贯。

本报告建议的采集数量、频率和后续门限都是待pilot校准的实验设计。当前已有15cm/6cm躯干与手部门控、8mm修正、角步长/加速度门限也只是既有实现阈值，不是动力链准确性验收标准。[现有门控](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/multiview_constraints.py:26)、[分项验收说明](/Users/krum5539/Documents/Tennis-Vision/viewer/video_import/MHR_FIT_PROGRESS.md:31)

## 最先值得落地的接口

让 Tennis-Vision 导出独立 `kinematic_reference_bundle`，包含视频/原片SHA、source_frame_id/PTS、camera/mirror/world标定来源与验收状态、原始各视角关节和骨段坐标系、测量轨迹与平滑轨迹、reference类型/设备及同步误差、contact及峰值人工/设备依据、train/eval角色。tennis_analyzer只消费该可追溯bundle进行同定义对比，不把Viewer mesh、默认9cm或融合后序列直接指定为ground truth。

先把“同帧、同物理定义、独立标定、独立参考、盲测误差”打通，才能回答动力链是否准确。纹理、握拍预览和动作评分可以继续迭代，但它们的视觉与工程验收不能替代这个基准。
