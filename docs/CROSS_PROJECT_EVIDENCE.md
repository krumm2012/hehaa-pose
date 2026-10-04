# Analyzer 与 Tennis-Vision 同源证据关联

`scripts/build_cross_project_review.py` 在离线生成新对照页，不进入实时推理路径。
原事件、原始帧记录及Vision历史拟合保持原版本；关联结果单独保存。

关联使用实际视频字节SHA256、Analyzer明确源帧ID、Vision原生重建的视频哈希与
逐帧tracking索引。相同文件名、相近平均FPS和模型一致性不能建立帧对应。
仅在同一输入文件、帧索引及来源契约全部可核验时，显示对应的Vision数组帧号。
缺少某个事件锚点的FrameRecord时，该锚点保持缺失。

```sh
venv/bin/python scripts/build_cross_project_review.py \
  --manifest /absolute/path/analyzer/evidence_manifest.json \
  --vision-result /absolute/path/vision/result \
  --vision-source /absolute/path/vision/source.mp4 \
  --reconstruction /absolute/path/vision/reconstruction.npz \
  --viewer-url http://127.0.0.1:18769/datasets/DATASET_ID/result/viewer.html \
  --analyzer-report-url http://127.0.0.1:8765/artifacts/PATH/report.html \
  --output data/analysis_results/kinematic_validation/cross_project_new_version
```

页内一键打开各自报告与Viewer，并列出开始、触球候选、运动峰值、结束的源帧号。
当前Viewer入口使用已有页面；对应帧号供手动定位，不宣称已自动同步三维播放器。
输出目录必须为新目录；`association.json`保存输入哈希、契约状态与阻断原因。

生成器还会读取该Viewer同目录的`mesh_meta.json`，核对实际服务的视频SHA与帧数。
入口属于其他视频、无法读取元数据或发生重定向时，不提供可点击的Vision入口。
这项检查验证入口的源视频身份，不证明当前纹理、姿态版本或三维模型准确。

## 原片与标准化副本

2026-10-05实际核查：50.03的Analyzer原输入为250帧，容器声明的`r_frame_rate=50/1`
与平均帧率`1600000/63813`不同；Vision标准化副本为249帧、25fps，视频SHA256不同。
此时原输入与标准化副本的关联状态为`pending_source_frame_mapping`，不按平均FPS换算帧号。

可为同一个Vision输入另生成Analyzer验证会话，逐帧对照两套输出：

```sh
venv/bin/python scripts/run_pixel_acceptance.py \
  --manifest /absolute/path/frozen_analyzer/evidence_manifest.json \
  --input-source /absolute/path/vision/source.mp4 \
  --output data/analysis_results/kinematic_validation/pixel_cross_project_new_version \
  --max-frames 249
```

这个参数只借用冻结机位几何，要求新输入尺寸相符。新会话记录参考与实际输入哈希，
不同输入明确为`unverified_derived_relationship`，不会证明标准化副本对应原片的哪次曝光。
原片和派生片之间的逐帧映射仍需单独保存与验收；原片人工标签不能直接套到新副本。

未指定`--input-source`时，原视频必须存在且SHA256与冻结manifest一致；不符时在
创建结果目录及启动推理之前拒绝。FrameRecord的确定性重放允许源视频不可用，
但重新从像素运行检测必须另行核验视频。显式替换输入保留独立来源和几何检查。

同源帧关联只帮助定位证据。人物身份、身体段定义、三维误差、曝光时间、峰值精度与
技术评分仍需独立验证；Vision模型重建保持`accuracy_validated=false`，不能作为独立真值。
