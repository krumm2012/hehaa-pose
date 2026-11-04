# 项目推进 · 2026-10-31

本次从 10 月 20 日冻结的五段日志、原视频和事件快照继续推进，不修改原始证据、模型候选触球帧或发布标签。

## 已实现

- `session_speed_trend.py` 按单位、策略、时间来源、标定和高度假设记录速度组。选择完整同源的 km/h 或 px/s 序列，不将部分 km/h 与部分 px/s 拼接。两种值同时存在时，优先完整米制序列，否则独立使用完整像素序列；缺值仍缺失，零值保留。
- 趋势至少需要六拍、基线及近期各两拍，并检查动作/训练条件、事件重叠、标定/方法一致性及人物画面尺度。缺少证据时阻断比较；基线零值不计算百分比。输出仅为 `SPEED_STABLE`、`SPEED_DECREASED`、`SPEED_INCREASED` 描述，不将速度变化命名为生理疲劳。动力链抖动当前不作合并推断。
- 帧特征及事件拍速保留几何假设、标定矩阵哈希与拍框中心测量身份。默认仍为此前选定的相机高 3.10 m、无逐帧高度时拍点假设 1.20 m。
- `joint_racket_2d_live.py` 已接入 `main_pipe.py` 的实时渲染路径：显示正面肩—肘—腕—握把—甜区—拍头联系和最近 24 帧轨迹；连续至少三帧不可用时提示，并把诊断写入逐帧日志。缺失处断线，源帧编号不连续时清空历史；不把真人和镜像观测融合。关联仍为候选，未成为业务选择依据。
- 存在七点数据且保存帧记录的运行路径默认显示此层。`racket_pose.render_joint_2d: false` 可在启动 YAML 中关闭；legacy 无七点时不启用。标注视频、实时报告预览与控制台 `/api/live-preview` 沿用同一渲染输出。
- 人工复核增加“下一拍待审核”和已保存人工收据进度，浏览器草稿不会被计为已完成人工参考。服务端按内容哈希及源绑定验证收据。

## 新产物

`data/analysis_results/project_advancement_20261031_v1/`：

- `index.html`：五段重算报告入口。
- 每会话 `events.json`、`report.html`、`speed_comparison.json`：固定原事件边界，重新计算特征和生物力学观测；移除旧网络建议/Coach意见，不伪称建议已重新验证。9/15 拍有接触候选帧的 km/h，剩余缺值保留。
- `contact_review/`：五段原图、轨迹与人工区间/锚点复核入口；`contact_review_queue.csv` 将触球窗口附近有连续缺点的事件优先列出。
- 每会话 `contact_review_draft.json`：可导入的空白草稿，全部为 `unreviewed`。截至生成时 **0/15** 完成人工复核；模型辅助复核即便完成，也不能直接充当独立盲标验证。
- `live_2d_frame_*.jpg`：保存日志重放的实际轨迹预览。重放统计的 update 用时只覆盖 CPU 计算，不含模型、渲染、编码、浏览器或网络，不是新的 25FPS 验收。
- `heldout_exclusions.json`、`independent_source_inventory.json`：现有开发/验收源及已知项目素材排除清单。仅凭哈希不同不足以证明未参与训练/调参；派生视频也不属于独立素材。

```sh
venv_yolo26/bin/python scripts/advance_joint_racket_review.py \
  --audit data/analysis_results/data_chain_audit_20261020/audit.json \
  --output data/analysis_results/project_advancement_NEW
venv_yolo26/bin/python scripts/serve_joint_racket_2d_review.py \
  --directory data/analysis_results/project_advancement_20261031_v1/contact_review --port 8771
```

打开 `http://127.0.0.1:8771/`，填写审核人，逐拍检查原帧和前后帧，记录区间、能确认的锚点及可见事实；看不清可标 unknown。完成后保存本机新收据。不要将“区间”视为已确认的单帧真值。

## 独立验证工具与仍待输入的工作

`independent_racket_validation.py` / `scripts/evaluate_independent_racket.py` 支持：

- 核对预测和人工参考的源视频 SHA-256、源像素坐标、零基源帧及图像大小；拒绝已知开发集源、重复帧、模型辅助参考和无审核来源的参考。
- 七点逐点的匹配数量、缺失预测数、平均/中位/最大像素误差，以及显式容差内点数占**可见人工参考总数**的比例，防止只统计容易检出的点。
- 人工独立区间对应的触球候选误差，分别报告区间外帧差与源 PTS 毫秒差。对应关系须人工复核，本工具不声称已经测得事件检测 Precision/Recall。
- 二维关键点和触球误差不能证明 km/h 或三维动力链准确；真实速度误差需要独立尺度/高度及速度参考。

产物目录有空白 `independent_reference_template.json`。准备独立预测文件时使用 schema `tennis.independent-racket-prediction.v1`，包含 `model_sha256`、相同源/坐标字段、`frames: [{frame_id, points: {sweet_spot: [x,y], ...}}]`、`contacts: {reference_id: predicted_frame}`。
人工参考 `frames` 包含源 PTS，`contact_pairs` 为 `{reference_id,start_frame,end_frame}`；未看清的点使用 null。

```sh
venv_yolo26/bin/python scripts/evaluate_independent_racket.py \
  --prediction /absolute/path/frozen_predictions.json \
  --reference /absolute/path/blind_human_reference.json \
  --exclusions data/analysis_results/project_advancement_20261031_v1/heldout_exclusions.json \
  --point-tolerance-px 5 --output /absolute/path/new_errors.json
```

上例 5px 只是运行参数示例，不是已验证的质量门槛。工具始终保留 `active_or_exclusive_allowed: false`，不自动放行模型。

目前尚无经确认的独立素材和人工盲标参考；人工触球复核、独立验证和完整实时性能复验仍待完成。旧640 Shadow正式版的性能验收不自动覆盖本次新渲染层。

## 本次验证

145项定向回归通过，新增模块语法检查及diff空白检查通过；浏览器核验真人视窗、下一拍待审核导航及0/15收据进度。没有创建真实人工确认收据，没有重新运行模型或授予完整实时性能/模型质量资格。

## 人工确认接入进度

已导入用户 krum5538 对 live_session_20261019T160611Z_35063f 的3拍人工确认，锚点为16、103、188；全批进度3/15。生成独立人工锚点重算分支及 human_review_summary.html / JSON / CSV，原模型候选不变。第1拍旧拍框几何估算由缺值变为45.0km/h，第2拍40.1km/h，第3拍仍缺值；七点甜区三拍均有合格像素速度。两种测量点与单位分别保留，不跨链补值。

复核页增加加载本会话已保存标注入口；服务端按源绑定和收据内容哈希恢复。其余12拍未发现对应触球标注文件，保持unreviewed；没有将模型候选或AI观察伪装为人工确认。
