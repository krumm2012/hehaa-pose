# 640 Shadow stable · 2026-10-19

文档同步：2026-10-20。正式版本为 [`640-shadow-stable-20261019`](https://github.com/krumm2012/hehaa-pose/releases/tag/640-shadow-stable-20261019)，不是 Prerelease。
性能验收绑定源码提交 `0b5f4e59125c3dbb3f21c19954ba39a033e74aea`，构建身份为
`06fb3441cf47b89a2eae0a681c9198189502c1f180dce730fd6df113734f5d1f`。
本文是发布后的文档同步，不改写原标签或历史验收结论。

## 发布范围

这是显式启用的 640 七点球拍 Shadow：新模型保存和显示七点观测，球选择、旧球拍选择、挥拍事件与 Coach 沿用旧链路。
默认仍为 `legacy`，默认旧版不启动七点工作进程。
`stable_25fps_accepted: true` 仅指下述硬件、配置和素材的性能验收；
`quality_gate_passed: false`、`active_or_exclusive_allowed: false` 保持不变。
不能将 Shadow 性能收据用于 active/exclusive 检测质量授权。

| 配置项 | 正式验收值 |
| --- | --- |
| 控制面板模型 / 球拍方案 | `640_shadow` / `shadow` |
| 处理 | `buffered`，每个已解码源帧处理，stride=1 |
| 调度 / 球拍传输 | `barrier` / `shared_memory` |
| 新球拍模型 | Core ML，全源帧独立letterbox至640×640，`ALL`，置信阈值0.25，关键点阈值0.5 |
| 人体 | 正面与镜面两路，Core ML `ALL`，`pose_resize_policy: legacy` |
| 球检测 | exp004，960输入，`ANE` |
| ROI | 冻结验证素材使用 Court02；现场必须绑定并复核实际机位 |
| 环境 | Apple M5 / 32 GiB；Python3.9.6、coremltools9.0、NumPy2.0.2、OpenCV4.13.0；默认OpenCV线程数10，推理线程池8 |

基础YAML和控制面板未选择640时仍为 `legacy`、`buffered`、`barrier`、`queue`。
选择640模型不会自动把IPC切为共享源帧；需手动选择 `shared_memory`。
共享传输免去源图通过队列序列化，预处理仍有必要的裁剪、缩放和颜色转换。
`interframe`、七点球拍分区裁剪和人体`direct_front/direct_source`不属于本次正式性能配置。

## 安装与身份核验

适用于已有球检测模型、双人体模型及Python依赖的macOS项目环境。
Release仅额外分发640球拍模型，不包含所有基础模型、虚拟环境或原视频。
先保存未提交工作，在目标安装目录检出正式标签；不要对当前工作目录强制重置：

```bash
git checkout 640-shadow-stable-20261019
mkdir -p /tmp/tennis-640-shadow-stable-20261019
gh release download 640-shadow-stable-20261019 --repo krumm2012/hehaa-pose \
  --dir /tmp/tennis-640-shadow-stable-20261019
```

在下载目录核验全部附件：

```bash
cd /tmp/tennis-640-shadow-stable-20261019
shasum -a 256 -c SHA256SUMS
```

回到项目根目录后解压模型。若同路径已有候选，先核对目录身份；不同的旧候选需另存，避免覆盖实验：

```bash
tar -xzf /tmp/tennis-640-shadow-stable-20261019/racket-pose-640-shadow-stable-20261019.tar.gz -C .
venv_yolo26/bin/python - <<'VERIFY'
from scripts.verify_stable_shadow_release import verify
print(verify('.', '/tmp/tennis-640-shadow-stable-20261019/stable-contract.json'))
VERIFY
```

`verify()`检查模型身份与发布范围，不运行模型或重新授予性能/准确率资格。
旧 `scripts/verify_shadow_release.py` 仍可核对同一模型目录，但显示的是历史preview身份合同；正式资格以下载的 `stable-contract.json` 与 `validation-summary.json` 为准。
本仓库的[正式身份合同](640-shadow-stable-20261019.json)与已发布附件 `stable-contract.json` 字节一致。

模型为 `models/racket_pose/candidates/bn-repaired-640-v1/racket_pose_bounded_fp16_640.mlpackage`。
项目目录哈希：`0dec1259f50b2cfb286f3ed5408a2f5f589749a0ddaf7a9b8a4139f29c1a025e`。
输入RGB640×640、输出`[1,26,8400]`；七点为butt_cap、grip_center、throat、sweet_spot、head_top、frame_left、frame_right。
目录身份与压缩包文件SHA256是不同校验；两者都需要匹配。

## 控制面板启动

```bash
venv_yolo26/bin/python local_control_panel.py --port 8766 --open
```

打开 http://127.0.0.1:8766/ 。未指定端口时程序默认仍是8765，本文显式使用8766。
在原工作台选择本地视频或流，绑定实际机位，检查四点ROI、镜面及隐私遮罩。
选择640七点球拍影子模型、连续处理、同帧并行和共享源帧，开启原有事件/Coach/报告与需要的录像能力。
修改只在下次启动生效；核对启动身份为640×640 / ALL / shadow及上述模型哈希。

## 命令行启动

`main_pipe.py`没有`--racket-model-profile`参数。需要先用控制面板同一配置函数生成独立运行YAML；不要只把基础960配置的mode改成shadow后当作640运行。
以下示例仅在源视频确属Court02时使用`court02-main`，其他机位必须替换。
每次使用新目录，避免覆盖已有会话：

```bash
venv_yolo26/bin/python - <<'CONFIG'
from pathlib import Path
import yaml
from racket_model_profiles import apply_model_profile
root = Path.cwd()
config = yaml.safe_load((root / 'configs/yolo26_tennis_config.yaml').read_text())
apply_model_profile(config, '640_shadow', 'shadow', root)
config['pipeline_perf']['pose_resize_policy'] = 'legacy'
output = root / 'data/analysis_results/640_shadow_manual_001'
output.mkdir(parents=True, exist_ok=False)
(output / 'runtime.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
CONFIG
venv_yolo26/bin/python main_pipe.py \
  --config data/analysis_results/640_shadow_manual_001/runtime.yaml \
  --input '/绝对路径/Court02原视频.mp4' --stream-id court02-main \
  --output data/analysis_results/640_shadow_manual_001/output.mp4 \
  --processing-mode buffered --racket-pose-mode shadow \
  --inference-schedule barrier --racket-ipc shared_memory \
  --algo2-dual-view --no-dual-view \
  --realtime-swing-events --realtime-coach \
  --evidence-manifest data/analysis_results/640_shadow_manual_001/evidence_manifest.json
```

`--algo2-dual-view`启用两路人体分析，`--no-dual-view`关闭旧原图/处理结果双窗口布局，两者不矛盾。
本地文件完整非节流分析可用`offline`，但它不是上述buffered实时性能验收。

## 验收结果及历史慢轮

| 项目 | 结果 |
| --- | --- |
| 九段原视频，各三轮 | 27/27严格通过；最低25.124 FPS，最差名义输入→输出P95 97.2ms |
| 不间断长测 | 15,250源帧及编码帧完整；稳定25.069 FPS，P95 88.6ms，RSS增长3.28% |
| 旧版/shadow业务字段对照 | 9/9一致；业务一致性不代表每轮旧版吞吐均通过 |
| 网络/语音故障固定复验 | 48.43与49.43各交替3轮，加50.03对照，共7/7通过；最低25.142 FPS，最差P95 69.0ms |
| 定向回归 | 133项通过 |

原48.43/49.43故障轮曾为21.964/21.890 FPS，P95为1444.9/1455.5ms，原测量与失败视频保留。
按用户要求，在同一冻结构建上进行上述固定七轮串行复验，全部结果计入，未修改模型、ROI、参数或门槛。
同期整机CPU空闲率44–85%、中位73%（含预热和后核验）；没有强制关闭用户其他应用。
CPU争用是环境线索，不是已证明的唯一根因；共享高负载仍可能降低吞吐。

延迟是预热后本地名义输入节奏到输出提交，不是相机曝光到屏幕呈现。
同一套本地工程素材不能证明跨球员/机位泛化、真实触球或真实三维测量；真实TTS成功合成/扬声器并发性能及上游RTSP损失仍需独立验收。

完整公开收据：[validation-summary.json](https://github.com/krumm2012/hehaa-pose/releases/download/640-shadow-stable-20261019/validation-summary.json)、[release-manifest.json](https://github.com/krumm2012/hehaa-pose/releases/download/640-shadow-stable-20261019/release-manifest.json)。
本机报告位于`data/analysis_results/racket_pose_acceptance_20261018/stable-shadow-release-20261019/final-0b5f4e5/index.html`，通过控制台HTTP artifacts入口访问；公开发布不依赖本机产物一直存在。

## 故障、回退与材料保留

连续Shadow逐帧等待同源新结果。工作进程加载失败、退出或超时会使本轮会话失败，不是故障后旧链路无阻塞继续的线上旁路。
停止会话后，将模型选择恢复为“当前已配置模型”、球拍方案改为`legacy`，重新启动；旧版不加载七点进程。
DeepSeek/TTS的失败保留本地文字建议，状态分别记录；这与七点工作进程失败策略不同。

保留原素材、逐帧日志、人工审核和标注、失败视频及最终发布核验记录。
成功的工程编码副本可在原清单SHA256与解码帧数验证后清理，需附省略回执，不把生成时manifest误当当前所有文件仍存在。
JSON/JSONL可采用保持原字节和路径的文件系统透明压缩；gzip/tar日志需在重放前按保留映射无损恢复。
截至2026-10-20，本机分析产物约8.07GiB、项目约15GiB；这是清理时快照，不是长期容量承诺。

历史：[preview说明](640-shadow-preview-20261019.md)、[候选冻结过程](640-shadow-candidate-20261019.md)。这些记录不覆盖本页正式版范围，也不能授权active/exclusive。
