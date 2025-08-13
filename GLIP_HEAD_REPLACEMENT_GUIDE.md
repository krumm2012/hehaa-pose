# GLIP头像替换功能使用指南

本指南将帮助您在网球分析系统中集成GLIP模型进行头像替换功能。

## 功能概述

这个系统可以：
- 使用GLIP模型进行基于文本描述的人脸检测
- 如果GLIP不可用，自动回退到OpenCV人脸检测
- 将检测到的人脸替换为自定义的Judy头像
- 支持多种混合模式：直接替换、Alpha混合、无缝克隆
- 提供人脸跟踪功能减少闪烁
- 生成详细的处理统计信息

## 快速开始

### 1. 创建示例头像
```bash
python create_judy_sample.py
```
这将在 `assets/` 目录下创建示例Judy头像文件。

### 2. 基础运行（使用OpenCV检测）
```bash
python main.py
```
如果没有安装GLIP，系统会自动使用OpenCV进行人脸检测。

### 3. 完整安装（包含GLIP）

#### 3.1 安装基础依赖
```bash
pip install -r requirements.txt
```

#### 3.2 安装GLIP（可选，但推荐）
```bash
# 方法1: 直接从GitHub安装
pip install git+https://github.com/microsoft/GLIP.git

# 方法2: 如果上述方法失败，尝试克隆后安装
git clone https://github.com/microsoft/GLIP.git
cd GLIP
pip install -e .
```

#### 3.3 下载GLIP模型文件
创建 `models/` 目录并下载以下文件：

1. **模型配置文件**: `glip_Swin_T_O365_GoldG.yaml`
   - 从GLIP官方仓库下载或使用提供的配置

2. **模型权重文件**: `glip_tiny_model_o365_goldg_cc_sbu.pth`
   - 下载地址: [GLIP Model Zoo](https://github.com/microsoft/GLIP#model-zoo)

```bash
mkdir -p models
# 将下载的文件放置到models目录
```

## 配置说明

### 配置文件位置
`configs/default_config.yaml`

### GLIP模型配置
```yaml
glip_model:
  enabled: true                           # 启用GLIP模型
  model_config_path: "models/glip_Swin_T_O365_GoldG.yaml"
  model_checkpoint_path: "models/glip_tiny_model_o365_goldg_cc_sbu.pth"
  confidence_threshold: 0.7               # 检测置信度阈值
  text_prompt: "face . head"              # 检测文本提示词
  device: "cuda"                          # 运行设备 ("cuda" 或 "cpu")
```

### 头像替换配置
```yaml
face_replacement:
  enabled: true                           # 启用头像替换
  judy_head_image_path: "assets/judy_head.png"
  judy_head_scale_factor: 1.2             # 头像缩放因子 (0.5-2.0)
  blend_mode: "seamless"                  # 混合模式
  replacement_opacity: 0.95               # 不透明度 (0.0-1.0)
  face_margin_ratio: 0.1                  # 人脸边界扩展比例
  track_faces: true                       # 启用人脸跟踪
```

### 混合模式选项
- `"direct"`: 直接替换，速度最快
- `"alpha"`: Alpha混合，带边缘羽化
- `"seamless"`: 无缝克隆，效果最自然但速度较慢

### 调试配置
```yaml
debug_mode: false                         # 设为true显示检测框
```

## 使用不同的头像图片

### 替换默认头像
1. 准备您的头像图片（推荐PNG格式，支持透明背景）
2. 将图片保存为 `assets/judy_head.png`
3. 或者修改配置文件中的 `judy_head_image_path`

### 头像图片要求
- **格式**: PNG（推荐）、JPG
- **尺寸**: 200x200像素以上
- **背景**: 透明背景效果最佳
- **内容**: 正面头像，清晰可见

### 创建自定义头像
```python
# 使用create_judy_sample.py作为模板
python create_judy_sample.py
# 然后编辑生成的图片或替换为您的图片
```

## 性能优化

### GPU加速
```yaml
glip_model:
  device: "cuda"  # 使用GPU加速
```

### CPU运行
```yaml
glip_model:
  device: "cpu"   # 使用CPU运行
```

### 调整检测频率
如果性能不足，可以考虑：
1. 降低检测置信度阈值
2. 使用更简单的混合模式
3. 禁用人脸跟踪

## 故障排除

### GLIP安装问题
如果GLIP安装失败：
1. 系统会自动回退到OpenCV检测
2. 确保已安装PyTorch
3. 检查CUDA版本兼容性

### 检测效果不佳
1. 调整 `confidence_threshold`
2. 修改 `text_prompt`（如："person face", "human head"）
3. 确保视频中人脸清晰可见

### 替换效果不自然
1. 尝试不同的 `blend_mode`
2. 调整 `judy_head_scale_factor`
3. 修改 `face_margin_ratio`
4. 使用更高质量的头像图片

### 性能问题
1. 将设备设置为 `"cpu"` 如果GPU内存不足
2. 降低 `confidence_threshold`
3. 考虑使用更小的模型

## 输出信息解读

### 初始化信息
```
初始化头像替换模块...
头像替换状态: {'enabled': True, 'glip_loaded': True, ...}
检测方法: glip
混合模式: seamless
Judy头像已加载: True
```

### 处理统计
```
头像替换统计:
  检测到的人脸: 1250
  替换的人脸: 1250
  平均每帧人脸数: 1.04
  检测成功率: 87.5%
  检测方法分布: {'glip': 1100, 'opencv_front': 150, 'opencv_profile': 0}
```

## 高级功能

### 自定义文本提示词
```yaml
glip_model:
  text_prompt: "tennis player face . athlete head"  # 更具体的提示
```

### 多人脸处理
系统会自动检测和替换视频中的所有人脸。

### 实时调试
设置 `debug_mode: true` 查看检测框和置信度。

## API参考

### HeadReplacementProcessor类
```python
processor = HeadReplacementProcessor(config)
result_frame = processor.process_frame(frame)
status = processor.get_status()
performance = processor.get_performance_info()
```

### 主要方法
- `process_frame(frame)`: 处理单帧图像
- `get_status()`: 获取系统状态
- `get_performance_info()`: 获取性能统计
- `reset_stats()`: 重置统计信息

## 支持和贡献

如果遇到问题或有改进建议，请：
1. 检查配置文件设置
2. 查看控制台输出的错误信息
3. 尝试不同的配置参数
4. 确保所有依赖正确安装

## 许可证

本功能基于开源许可证，请遵循相关模型和库的许可证要求。 