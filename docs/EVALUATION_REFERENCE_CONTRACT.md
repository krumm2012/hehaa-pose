# 挥拍参考对照与准确性资格

版本：`evaluation_reference_v1_review_agreement`。2026-10-05。
实现：`evaluation_reference_policy.py`、`swing_evaluation.py`。

评估器比较预测事件与提供的参考标注。比较成功或人工复核定稿不能证明参考独立，
也不能证明动力链、关节位置或技术评分已经准确。当前没有独立准确性批准路径。

## JSON字段

- 保留实际事件匹配计数、源帧差值、区间IoU及原有比较规则。
- `summary.stroke_type_match_ratio`：匹配事件中类型相同的比例。
- `summary.contact_within_tolerance_ratio`：匹配触球候选落入源帧容差的比例。
- `stroke_type_accuracy`、`contact_accuracy`保留为null兼容字段，不再承载未核验准确率。
- `precision`、`recall`、`f1`仍是事件对参考的统计；V2只有整段检查完成且没有待复核项才发布。
- `metrics_finalized`表示对照复核定稿，独立性另由`reference_provenance`记录。
- `reference_provenance`保存声明来源、模型提示是否可见、对照复核状态，以及
  `independence_verified=false`和`accuracy_validated=false`。导入文件的自行声明不批准独立性。
- V1缺少完整时间轴复核契约，原计数可审计，完成状态仍未核验。

## 页面显示

实时与独立报告统一采用对照标签：事件Precision/Recall/F1、类型匹配比例、触球帧容差匹配比例。
未复核或完成状态缺失时全部比例显示“待复核”；完成后仅显示有限、范围[0,1]的数值。
旧JSON即使保存100%也遵循这个显示资格，不改写原件。

模型预填的标注编辑器导出`source.reference_method=model_assisted_review`及
`model_predictions_visible=true`。用户在该页继续修改仍属于有模型提示的复核；
直接评估导入原件保留其声明，声明独立不代表已核验。

触球容差按源帧号而不是测量毫秒；采样、缺帧、实际曝光及参考误差另需验证。
不能把这里的匹配比例升级为动力链真实准确率或技术评分校准。

## 验证及后续

`test_evaluation_reference_policy.py`覆盖未复核默认值、完成的辅助复核、自行声明独立、
V1旧记录、两种页面和导出来源。原匹配算法的计数、误差、IoU及事件Precision/Recall/F1回归保留。
旧测试的“准确率”断言改为对应参考匹配比例；新增测试单独验证准确率字段为空。

独立位置、动作、触球、动力链及教练评分验收仍需真实标注、预定协议和数据分组。
此外，CLI评估器的部分旧帧号解析仍采用int转换，继续由T04审查；本契约不宣称完成所有身份消费者迁移。
