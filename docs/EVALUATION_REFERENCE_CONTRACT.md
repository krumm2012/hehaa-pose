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
## 评估入口的源帧身份（2026-10-05）

`evaluation_identity_v1_strict_source_frames`用于公开Python评估函数、文件写入器和CLI。
V1/V2在匹配前校验完整输入；已声明帧号只能为0至2^53-1的整数，
小数（含110.0）、布尔值、字符串、负数及非有限数值均拒绝，不取整或转换。
模型事件ID必须声明且唯一；V1参考按唯一事件ID匹配，V2参考按唯一非空字符串
annotation_id保留身份。旧V2没有annotation_id时按事件ID或原稿行号建立稳定标签，
在valid_hit过滤之前处理，不能因过滤而改变行身份。source_event_id只保留来源线索，
不能替代V2时间匹配，也不能批准独立性。

模型主字段为顶层start/contact/end_frame，随后是frames内同名_frame及短名；
参考主字段为frames内start/contact/end，随后是顶层_frame及frames内_frame。
按字段是否存在选取，不按真值选取：显式null保持缺失，真实0帧保留。
主字段缺失时仍支持合法旧别名；已声明的非法别名也不能被合格主字段掩盖。
仅建立内存规范副本，不回写原始文档；缺边界不能生成区间IoU或对应边界误差，
仍可凭其他已声明观测进行对照。反序区间或触球超出已声明区间直接拒绝。

事件数组、frames、来源和复核布尔字段有类型检查；不静默丢弃异常行，
不以字符串"false"批准复核完成。帧容差必须是非负安全整数，IoU门槛必须是
[0,1]有限数值；不截断或夹到合法值。比较JSON的settings.identity_policy以及
analysis_build.evaluation_identity_policy保存版本，运行快照新增契约源码哈希。

非法输入不创建或替换评估文件。输出路径不得覆盖事件或参考原件，含符号链接、
硬链接。CLI以状态2和明确原因拒绝。`test_evaluation_identity_contract.py`通过
公开评估、真实文件写入和CLI覆盖上述反例；合法数据保留匹配、差异、IoU及对照比例。

这里验证的是声明身份和输入资格，不证明源帧实际存在、曝光同步或真实触球。
人工重算与直接Coach派生也在计算前使用该文档校验，并对逐帧日志frame_id使用
同一整数契约；模型、来源链接和日志身份不再int转换。原件哈希和引用保留，
非法日志不发布新修订。日志中合法重复身份仍由锚点/源时间资格明确拒绝测量，
不能通过去重制造新观测。内部阶段表的JSON字符串键按已校验日志ID查表还原，
不把任意字符串解析成新的源帧。`test_manual_review_identity_contract.py`覆盖直接验证、
直接派生及实际修订发布边界。其他固定FPS、身份消费者、设备时钟与独立误差
仍由T04及其他待办跟踪。

## 报告与编辑器身份（2026-10-05）

`report_identity_v1_strict_event_and_coach_ids`复用公开模型事件身份校验，
在独立报告组装及两种HTML渲染入口验证完整事件文档。模型及Coach的event_id
必须是已声明、唯一的非负安全整数；不以int转换或默认0来建立关联，
重复Coach不能按后一行覆盖。声明帧别名均校验，内存副本保留null与真实0，
不会回写原JSON。模型缺少触球锚点时也不从另一个Coach文档补回。

输出report_identity明确Coach只按声明event_id关联，source_binding_verified=false。
整数相同不能证明不同文件来自同一会话；源内容哈希、会话绑定及其他消费者仍待验收。
合法报告的观测量和评分限制保留；非法输入在渲染前拒绝，不替换已存在报告。

`manual_annotation_identity_v2_event_links`内联到实时与独立编辑器。
JSON中的event_id/source_event_id严格为整数，annotation_id为非空字符串；
旧稿缺annotation_id时按合法event_id或原稿行号生成稳定标签。
来源字段显式null保持人工补充事件；仅字段未声明时兼容合法旧event_id链接。
DOM属性只解析程序写入的规范十进制整数文本，不能把任意JSON字符串转为身份。

导入和恢复先对完整文件校验帧别名、峰值帧、复核布尔字段、唯一标注ID和
唯一来源链接，检查链接存在于当前模型卡片，以及是否与未覆盖卡片ID冲突。
通过后才更新确认框、清除旧人工卡片并应用新稿；错误文件保留已有卡片及草稿。
恢复保留annotation_id，重复导入不会堆积人工事件；新人工ID避开已有manual编号。
保存、下载及评估采用同一身份校验。可选峰值不作为正式复核的必填锚点，
但已声明的非法峰值不能静默清空，显式空峰值也不能被旧模型峰值补回。

时间轴缺触球或峰值锚点时不绘制对应标记，缺起止边界不编造事件区间；
空定位请求不跳到0帧，实际0帧仍可定位。这里验证的是源身份传递，
媒体PTS与传感器曝光、派生视频帧映射及真实触球准确性仍是独立验收项。

`test_report_identity_contract.py`覆盖真实报告输入、写入及实时渲染边界；
`test_report_frontend_identity.py`执行两种实际生成页面的导入、恢复、导出和
时间轴函数；既有标注、工作流、参考对照测试继续验证兼容行为。
