# 🎾 Tennis Analyzer - AI 网球动作与轨迹全解析系统

一个集成了尖端计算机视觉技术（基于深度学习）的专业网球视频分析系统。它不仅能追踪球的运动轨迹，还能精确识别球员姿态、挥拍动作，并针对 Apple Silicon 进行了极致的性能优化。

---

## 🌟 核心功能

*   **🏃‍♂️ 高精度姿态检测 (Pose Estimation)**
    *   支持 **YOLO26m-pose (CoreML)** 与 YOLOv8-pose 自动切换。
    *   实时识别 17 个关键点，分析身体重心与动作一致性。
*   **🎾 智能球检测与追踪 (Ball Tracking)**
    *   **色彩编码系统**：自动区分红色（运动球）与蓝色（静止球）。
    *   **轨迹补全**：基于历史帧的渐变动态轨迹线，清晰展示球路。
*   **🎯 ROI 兴趣区域优化 (ROI Analytics)**
    *   支持 4 点描线定义非矩形场地区域。
    *   **计算量缩减 50%+**：仅在关键区域内进行高精度检测。
*   **🏓 挥拍与击球分析 (Swing Detection)**
    *   自动识别正手、反手挥拍类型。
    *   精准捕捉击球瞬间，计算击球点位。
*   **🎬 精彩瞬间捕捉 (Highlight Clips)**
    *   自动生成包含击球瞬间前后视频片段、数据分析 JSON 及关键帧截图。

---

## 🚀 性能革命

针对 Mac (Apple Silicon) 进行了深度适配，实现了从“几乎不可用”到“生产就绪”的质变：

| 指标 | 基础模式 | 性能优化模式 (Apple Silicon + ROI) | 改进倍数 |
| :--- | :--- | :--- | :--- |
| **推理速度** | 0.99 FPS | **26.03 FPS** | **26.3x ⚡** |
| **设备功耗** | 极高 (CPU 满载) | 低 (利用 Neural Engine) | **-50% 🔋** |
| **静止球干扰** | 严重 | 95% 过滤成功率 | **有效提升** |

---

## 📂 项目结构

```text
tennis_analyzer/
├── configs/            # 预设 YAML 配置文件
├── models/             # 模型存放区 (.mlpackage 和 .pt)
├── data/               # 默认输入输出及精彩瞬间目录
├── pose_estimator.py   # 统一姿态估计接口（支持 YOLOv8/v26）
├── ball_tracker.py     # 球追踪与运动状态判断核心
├── roi_manager.py      # ROI 区域选择与坐标转换
├── racket_detector.py  # 球拍检测与挥拍逻辑
├── main.py             # 统一入口程序
├── cleanup.sh          # 项目清理脚本 (用于移除缓存与临时文件)
└── archives/           # 历史归档文件
```

---

## 📄 分析与优化报告

项目包含多份详细的技术报告，涵盖性能优化、配置分析与算法演进：
*   **[PERFORMANCE_OPTIMIZATION_SUMMARY.md](file:///Users/krum5539/Documents/tennis_analyzer/PERFORMANCE_OPTIMIZATION_SUMMARY.md)**: 性能优化成果总览。
*   **[CONFIG_OPTIMIZATION_SUMMARY.md](file:///Users/krum5539/Documents/tennis_analyzer/CONFIG_OPTIMIZATION_SUMMARY.md)**: 运动球检测配置优化。
*   **[DETECTION_ISSUE_ANALYSIS.md](file:///Users/krum5539/Documents/tennis_analyzer/DETECTION_ISSUE_ANALYSIS.md)**: 检测漏报与误报深度分析。
*   **[PIPELINE_TEST_RESULTS.md](file:///Users/krum5539/Documents/tennis_analyzer/PIPELINE_TEST_RESULTS.md)**: 多线程/异步流水线测试对比。

---

## 🛠️ 快速开始

### 1. 环境准备
```bash
# 进入目录
cd tennis_analyzer
# 激活环境 (推荐 Python 3.10+)
source venv/bin/activate
# 安装核心依赖
pip install -r requirements.txt
```

### 2. 标准化启动
```bash
python3 main.py --config configs/yolo26_tennis_config.yaml --input "你的视频路径.mp4"
```

### 3. 项目清理
定期运行清理脚本以保持工作区整洁：
```bash
# 预览清理内容
./cleanup.sh --dry-run
# 执行正式清理
./cleanup.sh
```

---

## 🤝 故障排除

*   **视频打不开？** 请确保输入视频为 H.264/AVC 编码，或使用 `ffmpeg` 转换。
*   **速度还是慢？** 检查 `configs` 中 `skip_frames` 是否开启，或尝试调整 `compute_units` 为 `all`。
*   **模型找不到？** 确保 `models/` 目录下有相应的权重文件。

---

**最后更新**: 2026-02-03
**维护者**: Tennis Analyzer Team / Krumm