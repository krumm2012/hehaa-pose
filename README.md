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