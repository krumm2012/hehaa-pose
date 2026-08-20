# 人工复核闭环页面：技术设计

## 范围

本阶段只实现 Swing 人工真值导入、评估回写和人工 Coach 重算页面。检测框标注与 YOLO
训练留在后续任务中。原始 `final_events.json`、实时 Coach 和事件日志保持只读。

## 运行方式

新增本地 `manual_review_workflow.py`，从一个会话目录启动仅绑定
`127.0.0.1` 的 HTTP 服务：

```bash
python3 manual_review_workflow.py \
  --session-dir /tmp/tennis_rtsp_calibration \
  --open
```

服务复用该目录中的 `final_report.html`、`final_events.json` 和
`final_frames.jsonl`，并提供同源 API。使用 HTTP 而不是直接扩展 `file://` 的原因是浏览器
不能安全地执行 Python、原子写文件或重新聚合逐帧证据。

## 页面流程

1. 用户选择 `swing_manual_annotations_v2.json`。
2. 页面先显示 schema、事件数、待复核数及源会话匹配结果。
3. 用户点击“生成评估与人工 Coach”。
4. 后端原子保存标注并运行 Swing 评估。
5. 如果仍有 `needs_review`，只生成 provisional 评估并阻止人工 Coach。
6. 如果标注已完成，后端按人工边界重新聚合质量、生物力学和 Local Coach。
7. 页面并列显示原始实时 Coach 与人工校准 Coach，并列出发生变化的字段。

## 输出契约

- `final_manual_annotations.json`：本次导入的人工作业副本；
- `final_evaluation.json`：人工评估结果；
- `final_manual_events.json`：人工边界派生事件和 Coach；
- `final_manual_review_state.json`：页面状态、产物路径和错误；
- 原始 `final_events.json`、`final_report.html` 不被 workflow 覆盖。

## 人工事件重算

人工事件以 `source_event_id` 匹配原事件，然后覆盖人工确认的 start/contact/end/type。
系统从完整 FrameRecord JSONL 提取运动特征，按人工范围重算：

- pose / ball / racket 覆盖率；
- 触球窗口网球覆盖；
- 检测警告；
- 单摄像头生物力学；
- Coach calibration；
- Local Coach 建议。

人工结果必须写入 `review_provenance`，包含源事件 ID、标注 ID、变更字段和标注文件哈希。

## API

### `GET /api/manual-review/state`

返回事件列表、当前评估、人工 Coach 和 workflow 状态。

### `POST /api/manual-review/evaluate`

请求体为 `swing_manual_annotations_v2`。响应返回校验结果、evaluation 和人工事件。
错误会使用 4xx，并给出可直接显示的中文信息。

## 安全边界

- 仅监听 `127.0.0.1`；
- 只允许访问指定 session 目录内文件；
- 标注 JSON 有大小上限；
- 所有写入使用临时文件替换；
- 源会话不匹配、边界越界、事件重复或复核未完成时不生成正式 Coach。

## 测试

- 单元测试覆盖 schema、会话、边界、pending、正式评估和 Coach 重算；
- HTTP 测试覆盖 GET/POST 与目录穿越拒绝；
- 浏览器测试覆盖错误文件、pending 文件、完成文件和 Coach 对比刷新。

