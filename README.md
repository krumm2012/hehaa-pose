# 网球分析系统 - Tennis Analyzer

这是一个基于计算机视觉的网球分析系统，能够跟踪球员姿势和网球轨迹，并进行挥拍动作分类。

## 功能特点

- 球员姿势估计（使用YOLOv8-pose），专注于身体关键点，不显示头部关键点
- 网球轨迹追踪（使用HSV颜色分割和形状检测）
- 挥拍动作分类（前场、后场、发球等）
- 球拍检测和状态分析（使用YOLO检测，追踪球拍与球员和球的交互）
- 球速估计和轨迹分析
- 视频输出，包含分析结果的可视化
- 批量处理功能，支持递归处理目录下的所有视频

## 安装步骤

1. 克隆仓库：
```
git clone https://github.com/yourusername/tennis_analyzer.git
cd tennis_analyzer
```

2. 安装依赖：
```
pip install -r requirements.txt
```

3. 下载预训练模型：
```
# YOLOv8-pose模型会在首次运行时自动下载
# 或者使用以下命令手动下载：
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"  # 用于球拍检测
```

## 使用方法

1. 准备输入视频：
   - 将网球比赛视频放在`data/`目录下
   - 在`configs/default_config.yaml`中设置视频路径

2. 运行单个视频分析：
```
python main.py
```

3. 批量处理多个视频：
```
python batch_process_videos.py
```

4. 查看结果：
   - 单个视频分析完成后，输出视频将保存在`data/output_video.mp4`
   - 批量处理时，输出视频将保存在每个源视频同级的`out`目录中

## 配置选项

可以通过修改`configs/default_config.yaml`文件来调整系统参数：

- 输入/输出视频路径
- 球追踪参数（如灵敏度、过滤阈值等）
- 姿势估计置信度阈值
- 球拍检测参数和状态判断阈值
- 可视化选项（现已默认不显示头部关键点）

## 系统架构

- `main.py`: 主程序入口点
- `pose_estimator.py`: 姿势估计和动作分类
- `ball_tracker.py`: 网球追踪和轨迹分析
- `racket_detector.py`: 球拍检测和状态分析
- `batch_process_videos.py`: 批量处理目录下的视频文件
- `utils.py`: 辅助函数（绘图、坐标变换等）
- `configs/`: 配置文件
- `models/`: 预训练模型目录
- `data/`: 输入输出数据目录

## 技术细节

- 球员检测和姿势估计使用YOLOv8-pose
- 姿势可视化专注于身体关键点（肩膀、手肘、手腕、臀部、膝盖、脚踝），不包括头部关键点
- 球拍检测使用YOLOv8，支持与球员手部关联和球拍状态分析
- 网球追踪使用HSV颜色空间分割和Hough圆检测
- 轨迹平滑使用Savitzky-Golay滤波器
- 视频处理使用OpenCV

## 最近更新

### 2023-05-27更新
1. **优化姿势估计显示**：
   - 移除了头部关键点（鼻子、眼睛、耳朵）的显示
   - 跳过所有连接到头部关键点的骨架连接线
   - 保留了身体关键点和人体框
   - 优化了人体标签的位置，移动到人体框的右下角

2. **新增球拍检测功能**：
   - 使用YOLOv8检测网球拍
   - 根据距离和位置将球拍关联到球员
   - 分析球拍与球的交互状态（接近球、碰撞区域、空闲）
   - 在视频中可视化球拍的状态信息

3. **批量处理功能增强**：
   - 更新batch_process_videos.py以支持新增的球拍检测功能
   - 优化输出信息面板，显示更多分析内容

### 2023-06-15更新
1. **新增完整挥拍分析功能**：
   - 集成FullSwingAnalyzer模块，提供全面的挥拍技术分析
   - 分析挥拍阶段（准备、前挥、击球点、随挥等）
   - 评估准备姿势（肩部转动、球拍引拍位置、非惯用臂的扩展）
   - 分析挥拍动作（击球点位置、击球高度、手臂伸展度）
   - 评估脚步工作（站姿宽度、站姿类型、膝盖角度）
   - 提供力量指标分析（腿部弯曲、髋肩分离度）
   - 扩展信息面板，显示分类整理的挥拍技术分析数据

2. **综合改进**：
   - 优化关键点显示，更清晰地专注于身体动作分析
   - 改进main.py和batch_process_videos.py保持功能一致性
   - 信息面板分类展示，更直观呈现多维度分析结果

### 2023-06-28更新
1. **信息面板UI重新设计**：
   - 采用更直观的分类展示方式，根据挥拍阶段动态显示分析内容
   - 引入笑脸+分数的评分格式，视觉效果更友好
   - 使用彩色标题和分类标签，增强可读性
   - 添加挥拍细节评价，如"表现完美"、"表现优秀"、"需要改进"等
   - 新增"获取建议"功能入口
   - 实现一键分享按钮，方便分享分析结果

2. **技术分析算法优化**：
   - 针对不同挥拍阶段(引拍准备、发力启动、击球阶段、随挥)优化技术评分算法
   - 完善四大类技术分析：引拍转体、发力启动、挥拍击球、分腿垫步
   - 综合多项技术指标计算得分，更全面反映技术水平
   - 增强关键技术要点识别，提供针对性评价和建议

### 2023-07-02更新
1. **中文字体显示支持**：
   - 解决OSD信息面板中文乱码问题
   - 集成PIL图像处理库以支持中文字体渲染
   - 自动识别并使用系统中可用的中文字体(PingFang、微软雅黑等)
   - 优化中文文本的大小、颜色和间距，提升可读性
   - 优化字间距和排版，使界面更加美观

2. **界面显示改进**：
   - 优化表情符号与文字的融合显示
   - 改进文本分隔符，使用"、"替代","，更符合中文阅读习惯
   - 调整按钮位置和尺寸，提高可点击性
   - 细化文本颜色对比度，增强内容可读性

### 2023-07-05更新
1. **中文字体自动下载功能**:
   - 添加字体检测与诊断日志，便于排查中文显示问题
   - 实现缺少中文字体时的自动下载机制
   - 支持多平台字体路径自动检测（Windows/macOS/Linux）
   - 创建本地字体目录用于存储和共享字体文件
   - 增加字体可用性验证，确保正常显示中文

2. **日志系统增强**:
   - 添加完整的日志记录功能，实时监控程序运行状态
   - 记录字体加载和渲染过程，便于调试中文显示问题
   - 优化错误处理，添加详细的异常信息记录
   - 使用分级日志级别（INFO/WARNING/ERROR）记录不同重要性的信息
   - 提供字体回退机制，确保在任何情况下都能显示文本

### 2023-07-10更新
1. **技术指标详细展示**:
   - 保留并展示所有技术指标的具体数值
   - 显示肩部转动角度、非执拍手伸展程度等准备阶段指标
   - 展示髋肩分离角度、膝盖弯曲角度等发力指标
   - 提供击球点位置、手臂伸展度、击球高度比率等击球技术数据
   - 详细呈现站姿宽度（像素和身高比）、站姿类型等脚步工作指标

2. **界面优化调整**:
   - 移除"一键分享给好友"按钮，精简界面
   - 增加当前帧号和总帧数信息，方便视频分析定位
   - 视频批处理时展示当前处理的文件名
   - 添加各技术指标原始数据，便于教练和专业人士分析
   - 使用浅灰色显示技术参数，与评价内容形成视觉层次

tennis_analyzer/
├── main.py                     # Main script to run the analysis
├── pose_estimator.py           # Handles yolo-pose detection and swing classification
├── ball_tracker.py             # Handles TrackNetV2 ball tracking
├── racket_detector.py          # Handles racket detection and state analysis
├── batch_process_videos.py     # Script for batch processing videos
├── utils.py                    # Helper functions (drawing, coordinate transformations)
├── configs/
│   └── default_config.yaml     # Configuration parameters
├── models/                     # To store downloaded model weights (e.g., .pt files)
│   ├── yolov8n-pose.pt         # For pose estimation
│   ├── yolov8s.pt              # For racket detection
│   └── tracknetv2_weights.pth  # (Or however TrackNetV2 weights are stored)
└── data/
    └── input_video.mp4         # Your input video
    └── output_video.mp4        # Annotated output




Racket Trajectory: Track the racket's center point over several frames.
Racket Velocity/Acceleration: Calculate this from the trajectory.
Ball Trajectory Analysis: Know if the ball is incoming or outgoing.
Player Pose during Swing: Certain body poses correlate with different swing phases (backswing, contact, follow-through). The swing_type from pose_estimator.py can be combined.
Temporal Logic:
Pre-hit (Backswing/Preparation): Racket moving away from the net, or player in a ready stance. Ball is approaching.
Pre-hit (Forward Swing): Racket moving towards the ball's predicted impact point.
Impact: Racket and ball in very close proximity, high racket speed (relative to previous frames). Ball trajectory changes direction. This is the hardest to pinpoint to a single frame.
Post-hit (Follow-through): Racket continues its motion after the impact zone. Ball is moving away.
This would involve a more complex state machine or even a sequence-based machine learning model trained on racket/ball/player data. For now, the provided code gives you the bounding box of the racket and a very basic state based on proximity to the ball. You can build upon this foundation.



FullSwingAnalyzer Class (full_swing_analyzer.py):
Takes player keypoints, associated racket information, ball position, and frame dimensions.
Dominant/Non-Dominant Side: Assigns keypoints based on the dominant_hand config.
Phase Estimation (Basic): Tries to guess the current phase (Backswing, Impact, etc.) based on the racket's state (from racket_detector) and its position relative to the player. This is very heuristic.
Preparation Metrics: Analyzes shoulder turn, racket take-back position relative to the dominant shoulder, and non-dominant arm extension.
Swing Motion Metrics: Infers contact point (in front, side, late) and contact height if the racket and ball are very close. Checks dominant arm extension.
Footwork Metrics: Calculates stance width, guesses stance type (open, closed - very simplified), and knee bend angles.
Power Indicators: Looks at leg bend (from knee angles) and estimates hip-shoulder separation as an indicator of body coil.
Returns a dictionary of these metrics.