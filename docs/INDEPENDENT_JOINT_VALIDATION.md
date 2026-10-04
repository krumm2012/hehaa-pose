# 独立关节点验证

此流程用于 Tennis Analyzer 二维姿态位置误差评估，不验证三维动力链或技术评分。

## 固定证据

- `validation/baseline_cases.json` 锁定已有帧记录与事件记录的 SHA-256。
- `scripts/audit_pose_resolution.py` 重新运行模型，保存视频、模型、代码哈希及原图坐标。
- `--continuous` 使用连续 250 帧与动态 ROI；默认模式使用 30 个独立帧与固定 ROI。两种模式分别报告，不混合解释。
- 分辨率差异是稳定性，不是误差真值。原分辨率不能自动当真值。

## 独立标注

打开 `data/analysis_results/kinematic_validation/joint_labels_20261004/index.html`。
按人物自身左右选择关节，正面人物与镜面背影分开。不可辨认时明确标记，不能猜坐标。
标注页不显示模型点；当前帧含触球候选与已发现异常帧，属于针对性样本，不能代表全体挥拍准确率。
建议两名标注者各自导出文件，保留原稿，另行仲裁分歧。不要把模型输出拷贝进真值文件。

页面未填写项保持缺失。导出前填写标注者编号并复核已填点。新版生成器支持本地自动保存、刷新恢复、JSON 导入续标、最近20个修订导出、前后帧快捷键和下一未填项。草稿按视频哈希、帧范围、模式及模型版本隔离；来源或坐标不匹配时拒绝导入，独立标注不能导入辅助模型建议。浏览器存储不可替代文件备份，请定期导出。

2026-10-05 新版连续22帧页面为 `independent_temporal_resume_v2_20261005/` 与 `assisted_temporal_resume_v2_20261005/`。旧页保留，不自动升级其脚本。生成器拒绝覆盖已存在的标注页；需要修改界面时生成新版本，再导入匹配来源的标签。批量确认保留确认状态；坐标、可辨认性或删除修改撤销全局确认。

## 评估命令

从仓库根目录运行：

```sh
venv/bin/python scripts/evaluate_joint_labels.py \
  --labels /absolute/path/to/exported_joint_labels.json \
  --predictions data/analysis_results/kinematic_validation/pose_temporal_fixed_20261004/audit.json \
  --tolerance-px 10 \
  --output /absolute/path/to/evaluation.json
```

10px 仅为调用示例，不是已验证标准。应预先确定容差并保留所有版本的结果，不能看完结果后调整门槛宣称提高准确率。

必须同时看：

- **位置误差**：人工可辨认且模型新鲜合格输出的配对点计算距离。
- **合格输出率**：模型合格输出 / 人工可辨认点；缺输出保留在分母。
- **错误输出率**：超过预设容差 / 合格配对输出。
- **总体容差内成功率**：容差内输出 / 人工可辨认点；隐藏预测不能提高此值。
- **不可辨认点上的模型输出**：只作单独计数，不能标记为已证明错误，因为真值未知。
- **未标注数**：明确保留，不能当作无误差样本。

评估按分辨率、视角、关节分别统计。草稿或未确认文件不产生准确性统计。哈希/坐标契约不一致或真值坐标非法时拒绝执行。

## 尚未完成

真实人工标签尚未提供，因此真实误差、错误输出率与评分校准仍未知；目前通过的是工程回归和稳定性检查。

## 模型辅助复核（单独记录）

已接收 `assisted_joint_review_received_20261004/assisted_joint_review.json`：
6 帧、130 个批量接受点、14 个未标注点，无坐标修改。这是辅助复核样本，
不能代替上述独立真值，也不能校准模型置信度或动力链准确率。

用 `scripts/compare_assisted_joint_review.py --review FILE --predictions FILE --output FILE`
单独检查分辨率一致性。输出资格包含当前源帧、原图坐标边界和模型分数；
所有已接受可见点保留在分母中。原分辨率与其自身预标注零偏移仅为数据一致性。
本次半分辨率正面保留 69/69 点，背面保留 49/61 点；暂不据此修改实时默认分辨率。

### 2026-10-05 标注计划修订

最新无模型提示连续帧页：`data/analysis_results/kinematic_validation/independent_temporal_plan_v3_20261005/index.html`。源帧175–196共22帧，显式 `requested_joints` 为左右肩、左右髋；双视角计划176项，空草稿评估状态为 `pending_independent_confirmation`。旧v2页面缺计划字段，评估会使用传统十二关节分母；保留旧页，最终四关节采集使用v3，不能把空草稿视为已确认标签。

## 两位标注者的分歧仲裁

`scripts/adjudicate_joint_labels.py` 比较两份独立原稿，生成原帧对照页，并校验最终裁决。
两份原稿必须对应同一视频哈希、帧尺寸及关节计划；最终发布要求两位不同标注者
各自确认原稿。模型辅助文件及带模型来源的点不能转换为独立参考。

```sh
venv/bin/python scripts/adjudicate_joint_labels.py create \
  --left /absolute/path/reference_A.json \
  --right /absolute/path/reference_B.json \
  --source /absolute/path/original_video.mp4 \
  --output data/analysis_results/kinematic_validation/adjudication_new_version \
  --tolerance-px 10
```

这里的10px仅用于查看分歧大小，不是准确性阈值。任何非零坐标差异、可辨认性分歧及
漏标均需显式裁决；不按容差自动平均。红色为A原稿、蓝色为B原稿、黄色为最终裁决，
可接受一份原稿、在原帧重新点选或标记不可辨认，并填写依据。两份原稿尚未确认时
只允许查看，不能发布最终标签。页面支持本地保存、刷新恢复和导入续标；修改裁决或
仲裁者编号会撤销整体确认。需保存导出的`joint_adjudication_decisions.json`。

```sh
venv/bin/python scripts/adjudicate_joint_labels.py finalize \
  --left /absolute/path/reference_A.json \
  --right /absolute/path/reference_B.json \
  --plan data/analysis_results/kinematic_validation/adjudication_new_version/plan.json \
  --decisions /absolute/path/joint_adjudication_decisions.json \
  --output /absolute/path/adjudicated_labels_new_version.json
```

最终标签可交给上面的`evaluate_joint_labels.py`。原稿、计划和裁决均以完整JSON的规范化
SHA256绑定；输入改变后必须重新建立计划并确认。生成器和最终发布拒绝覆盖已有输出，
原输入保持不变。裁决完成仅说明流程完成，不证明坐标真值、三维动力链或技术评分准确。
合成数据只用于工程验收，真实双人标签仍待提供。
