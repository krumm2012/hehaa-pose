# Tennis Analyzer 项目工程指南

本文件为 Codex 与 Antigravity 共用的根目录规则入口，仅保留跨任务硬约束与文档索引。

## 1. 项目定位与活跃入口

面向 Apple Silicon 本机、单机位 + 后墙镜面反射双视角的室内网球训练智能分析系统。

- **实时与离线分析主链路**：`main_pipe.py`、`pipeline/`（`cli.py`、`inference_worker.py`、`analyzer_worker.py`、`evidence.py`）、`dual_view_manager.py`、`dual_pose_estimator.py`、`pose_estimator_yolo26.py`、`dual_view_biomechanics.py`、`yolo26n_unified_detector.py`、`frame_processor.py`、`realtime_swing_runtime.py`、`realtime_swing_pipeline.py`
- **本地控制面板与持久化**：`local_control_panel.py`、`control_panel/`（`http_handler.py`、`db_store.py`、`session_store.py`、`settings.py`、`trajectory_refiner.py`）、`local_control_panel.html`、`player_compare.html`、`static/js/`、`templates/reports/`
- **人工复核与校准**：`manual_review_workflow.py`、`scripts/serve_joint_racket_2d_review.py`
- **权威配置文件**：`configs/yolo26_tennis_config.yaml`（主配置）、`configs/roi_config*.yaml`（机位 ROI 与镜面标定）
- **历史遗留勿误改**：早期 `main.py`、HSV `ball_tracker.py` 及 `ARCHITECTURE.md` 历史章节已废弃，勿作为新功能入口或架构依据。

## 2. 运行环境与验证命令

- **Python 虚拟环境**：主链路与测试一律使用 `./venv_yolo26/bin/python`（Python 3.9），TTS 独立模块使用 `./venv_qwen3_tts/bin/python`（Python 3.11）；禁止用系统 `python3` 运行主链路或单测。
- **默认快速回归**：改动代码后先跑全套快速检查：
  ```bash
  ./scripts/run_fast_tests.sh
  ```
  （依次校验 `node -c static/js/*.js`、`node --test tests/*.cjs` 与 11 个核心 Python 单测）
- **定向单元测试**：按修改模块补充运行对应单测，例如：
  ```bash
  ./venv_yolo26/bin/python -m unittest test_stitched_dual_pose.py test_auto_calibrate_manual_review.py
  ```
- **本地服务与版本核验**：
  - 控制面板：`./venv_yolo26/bin/python local_control_panel.py --port 8766`
  - CLI 会话校订：`./venv_yolo26/bin/python manual_review_workflow.py --session-dir <dir> --port 8767`
  - 2D 联合复核：`./venv_yolo26/bin/python scripts/serve_joint_racket_2d_review.py`（默认端口 `8770`）
  - 修改服务端 Python 代码后**必须重启服务进程**，并通过 `GET /api/status` 核验 `service_build`（哈希在模块导入时冻结）。校准与报告页面必须通过 HTTP 服务访问，禁止使用 `file://`。

## 3. 核心工程铁律

- **Core ML 计算单元（Apple Silicon M5 / macOS 26.7.1）**：严禁对 YOLO26 Core ML 模型设置 `CPU_AND_GPU` 或 `CPU_ONLY`——`iOS15–18` 模型会在 `MPSGraphExecutable` MLIR pass 触发 C++ `SIGABRT` 崩溃，`iOS26` 虽不崩溃但单次慢 4.3×（`41.97 ms` vs ANE `9.84 ms`）。全线保持 **`Det(ANE) + Pose(ALL)`**。
- **姿态预处理与 2 合 1 拼图单次推理**：测量主链路保持 `pose_resize_policy: direct_source` + `pose_preprocess_mode: letterbox` + 浮点亚像素坐标（禁止 `int()` 截断）；默认启用 `pose_stitched_single_pass: true`（正/背 2 合 1 拼图单次推理与零拷贝缓冲复用，可通过该开关设 `false` 回退双实例并发）。
- **背面镜面姿态双守卫**：`dual_pose_estimator.py` 背面分支写入缓存前，必须保留 `_enforce_backview_anatomical_consistency` 与 `_suppress_backview_foreground_occluded_ankles`，防止单帧误翻腿污染跨帧缓存。
- **数据、时间与指标契约**：遵守 `docs/adr/0003-source-media-time-contract.md` 的源媒体时间契约，保留跨视角自愈溯源标记（`allow_recovered` / `is_recovered`）；涉及挥拍指标、分段或教练规则计算前，必须先核对 `docs/METRIC_CALCULATION_CONTRACT.md`。

## 4. 安全、Git 与仓库卫生

- **凭据与隐私保护**：禁止将 `.env.local`、`.camera-credentials.local.env`、RTSP 账号密码或 LLM API Key 写入代码、日志、命令行示例或提交记录。
- **产物与模型体积控制**：禁止提交 `data/analysis_results/` 下的运行时输出产物；未经用户明确要求，不要修改或重导受 Git 跟踪的 `.mlpackage` / `.pt` 模型权重。
- **Git 提交约定**：采用 Conventional Commits 格式 `<type>(<scope>): <subject>`（如 `feat(pose): ...`、`fix(report): ...`）；未经用户明确指示，不要自动执行 `git commit` 或 `git push`。

## 5. 按需深读索引

| 主题 | 查阅文档 |
| --- | --- |
| Core ML 实测数据与双视角姿态避坑 | `docs/agents/pose-coreml-lessons.md` |
| 系统运行架构与模块边界 | `docs/SYSTEM_DESIGN.md` |
| 本地部署、CLI 验收与故障排查 | `docs/OPERATIONS.md` |
| 关键架构决策记录（时间/缓冲/共享内存） | `docs/adr/` |
| 挥拍生物力学指标与计算契约 | `docs/METRIC_CALCULATION_CONTRACT.md` |
| 功能需求快照与验收基线 | `docs/REQUIREMENTS_BASELINE.md` |
