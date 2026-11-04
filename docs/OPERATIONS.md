# 本地运行与运维手册

更新：2026-10-20；当前正式版为[640 Shadow stable](releases/640-shadow-stable-20261019.md)，性能收据绑定`0b5f4e5`；本文是发布后文档同步。运行前阅读 [系统设计](SYSTEM_DESIGN.md) 与 [需求基线](REQUIREMENTS_BASELINE.md)。

## 1. 环境与启动检查

当前视频环境为 `venv_yolo26`（Python 3.9），TTS 独立使用 `venv_qwen3_tts`（Python 3.11）。推荐 Apple Silicon Mac；模型、音频设备和摄像头权限需在运行机器就绪。

```bash
cd /Users/krum5539/Documents/tennis_analyzer
git status -sb
venv_yolo26/bin/python --version
venv_yolo26/bin/python main_pipe.py --help
venv_yolo26/bin/python local_control_panel.py --help
```

模型检查主配置 `configs/yolo26_tennis_config.yaml`：默认 exp004 Core ML 球/拍模型与 yolo26m-pose。检查实际模型文件、manifest 和加载日志。部分 Core ML 文件已受 Git 跟踪，`.gitignore` 中的 `*.mlpackage/` 不会取消既有跟踪；不能只看忽略规则判断模型是否分发。

首次安装 TTS 可运行 `scripts/setup_qwen3_tts_mlx.sh`；MLX-Audio 已固定到本机验证的 commit `6a92a1f20b033b7a2a6caa30db30146e58cca322`。安装前记录环境版本，首次模型下载和加载可能较慢。

## 2. 推荐：控制面板

```bash
cd /Users/krum5539/Documents/tennis_analyzer
venv_yolo26/bin/python local_control_panel.py --port 8766 --open
```

访问 http://127.0.0.1:8766/ 。未指定`--port`时程序默认仍为8765，本文显式使用8766。页面选 Court 或自定义流，临时填写凭据，检查 ROI 预览，再启动。也可预先设置 `TENNIS_RTSP_USERNAME` 和 `TENNIS_RTSP_PASSWORD` 环境变量。DeepSeek 密钥使用 `DEEPSEEK_API_KEY`；控制台支持从 `.env.local` 读取所配置的密钥变量。不要将真实凭据写入提交或排障记录。

2026-10-05：`GET /api/status` 新增 `service_build`，记录控制进程PID、模块加载时间、
控制源码SHA256及所加载的指标/时序/观测/评分策略。哈希在模块导入时冻结，修改磁盘
源码后不会将旧进程伪装成新版；需重启服务再核查。此字段用于运行版本验收，不代表
模型误差或传感器曝光时间已经标定。分析会话状态仍由顶层`state`与`pid`表示。

初次排障可逐项开启DeepSeek、TTS、HDMI和录像以定位开销；这不能作为关闭既有能力后的完整性能验收。正式验收的旁路组合与范围见[版本说明](releases/640-shadow-stable-20261019.md)。记录每次实际生效配置。

基础默认仍是`legacy`、`buffered`、`barrier`、`queue`；旧版不加载七点。使用正式640需在原控制面板显式选`640_shadow`/`shadow`、连续处理、同帧并行和共享源帧，人体缩放保持`legacy`。本地视频绑定实际机位后应用相应ROI/镜面；未绑定时不套摄像头ROI。七点模型不输出人体，不能替代双人体推理。

640的25FPS资格限已验收M5/32GiB、配置和CPU资源可用条件。active/exclusive仍未取得该模型的检测质量授权，不能仅修改mode绕过；首次安装、独立CLI YAML与收据核验见[正式版安装说明](releases/640-shadow-stable-20261019.md)。

停止分析使用页面“停止”，等待进程结束并检查证据清单。关闭控制服务使用其终端 Ctrl+C。TTS 关闭会取消待播请求并回收 worker；停止后的最后一条语音可能被中断，文字结果保留。

## 3. 命令行视频与 RTSP 验证

本地视频路径为占位符，替换为存在的干净原视频。每次使用新的输出目录。

```bash
venv_yolo26/bin/python main_pipe.py \
  --config configs/yolo26_tennis_config.yaml \
  --input '/绝对路径/测试视频.mp4' \
  --processing-mode buffered --racket-pose-mode legacy \
  --output data/analysis_results/manual_check_001/output.mp4 \
  --no-save-video \
  --realtime-swing-events --realtime-coach \
  --realtime-analysis-interval 5 --realtime-settle-frames 15 \
  --evidence-manifest data/analysis_results/manual_check_001/evidence_manifest.json
```

`--no-save-video` 关闭整段标注录像，独立挥拍片段仍可能生成。RTSP 测试将 input 替换为可达地址，例如 `rtsp://127.0.0.1:8554/16-10`，保持`--processing-mode buffered`逐帧处理；该地址要求本地已有推流服务，不由本项目创建。只有显式接受丢帧时才用旧`realtime`策略；`buffered`会覆盖旧丢帧标志，不能把`--drop-stale-frames`当作默认优化。

语音参数为 `--realtime-coach-tts`；只保存音频增加 `--realtime-coach-tts-no-playback`。DeepSeek 使用 `--deepseek-coach`。不要将 API Key 写在命令参数中。

## 4. 产物与人工校准

控制台会话位于 `data/analysis_results/control_panel/<session_id>/`。保留事件 JSON、逐帧 JSONL、事件 JSONL、证据 manifest、人工标注与评估；报告、片段和音频便于复核。音频位于 `*_coach_audio/`。

从控制面板的报告链接进入人工校准，完成整段复核、处理待复核项，再导入或评估标注。只运行 `swing_evaluation.py` 不会自动执行全部 Coach 重算流程。完整闭环需要逐帧证据，页面下载 JSON 本身不会训练模型。

CLI 产生的已结束会话可单独启动兼容工作流服务，使用不同端口避免冲突：

```bash
venv_yolo26/bin/python manual_review_workflow.py \
  --session-dir data/analysis_results/manual_check_001 \
  --port 8767 --open
```

历史会话文件命名需能被 `discover_session_paths()` 识别；缺少事件或帧日志时先核对产物路径。不要用 `file://` 页面替代 HTTP 校准入口。

## 5. 故障定位

| 现象 | 核查与处理 |
| --- | --- |
| 端口占用 | `lsof -nP -iTCP:8766 -sTCP:LISTEN`核对本文端口；省略`--port`时另查8765。优先复用已有控制台，其他服务改用8767等空闲端口 |
| RTSP 无画面 | 先用预览验证地址、凭据、网络及推流服务；随后检查 Reader 连接/读取超时和重连日志 |
| 启动失败 | 找第一个子进程异常及完整 traceback；最终 PipelineProcessError 是汇总结果 |
| Torch/LibreSSL 警告 | 与致命 traceback 分开；记录环境版本，勿据警告直接判定模型失败 |
| DeepSeek unavailable | 检查密钥是否存在、旁路错误原因、网络和接口配置；本地建议应继续生效 |
| TTS pending / 无声音 | 检查独立 Python、模型缓存和音频设备；首请求默认 120 秒、后续 30 秒，可通过 coach_tts 配置修改 |
| 声音断续 | 检查 coach_tts 的 first_audio_ms、audio_underflows 和 playback_error；生成与播放已独立，持续生成过慢仍会欠载 |
| 报告不更新 | 先恢复自动刷新，再核对修改时间；写线程错误会通过运行时健康检查上报并停止异常会话 |
| 控制令牌无效 | 刷新控制面板并重新从报告入口进入，获取新令牌 |
| 评估 provisional | 确认整段已检查、待复核项已处理及标注完整；不要手改指标状态 |
| 七点Shadow退出/超时 | 连续Shadow本轮失败；保留原因后停止，再单独启动legacy。DeepSeek/TTS旁路失败与此策略不同 |
| 关闭慢 | 核查片段编码、TTS 请求、队列排空与清单哈希；强制终止可能留下不完整产物 |

## 6. 回归、备份与版本回退

```bash
venv_yolo26/bin/python -m unittest -q \
  test_runtime_resilience \
  test_qwen3_tts_sidecar test_realtime_swing_runtime \
  test_realtime_swing_pipeline test_local_control_panel \
  test_manual_review_workflow test_pipeline_supervision \
  test_local_realtime_coach test_deepseek_realtime_coach

venv_yolo26/bin/python replay_evidence_bundle.py \
  data/analysis_results/manual_check_001/evidence_manifest.json --verify-only
venv_yolo26/bin/python replay_evidence_bundle.py \
  data/analysis_results/manual_check_001/evidence_manifest.json
```

重放从 FrameRecord 开始，不重新推理视频。CLI 返回 2 表示最低重放集合不可用，3 表示重放语义不一致；还要查看打印的 `valid` 和具体文件校验结果，不能只依据退出码判断所有补充产物完整。

升级前记录 Git commit、生效配置、模型 manifest、Python 和依赖版本，并保留一份成功会话。回退时停止当前会话，在独立目录准备已验证代码及匹配模型/依赖，运行上述回归和固定视频验证后再切换。不要对有未提交变更的仓库做强制重置。

备份应包含必需JSONL、事件、人工标注、manifest及引用的外部视频；只备份manifest不足以重放。先停会话、核对保留清单；已验证成功的工程编码副本可在SHA256和解码帧数核验后清理，附省略回执。人工审核、标注、失败视频、逐帧日志和最终验收记录保留，原生成manifest仍是历史记录，不代表已清理视频仍在本地。

2026-10-20本机清理后，分析产物约8.07GiB、项目约15GiB；目录体积会继续变化。文件系统透明压缩保持原字节和路径，普通读取/HTTP继续可用；`*.jsonl.gz`或`*.journals.tar.gz`则需按回执无损恢复后重放，不要把压缩文件直接当JSONL传入。清理不要直接套用根目录`cleanup.sh`，它会递归删除缓存并涉及历史分析目录。
