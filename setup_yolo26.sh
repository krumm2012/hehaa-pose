#!/bin/bash
# setup_yolo26.sh
# YOLO26-pose 环境设置脚本

set -e  # 遇到错误立即退出

echo "============================================================"
echo "🎯 YOLO26-pose 环境设置"
echo "============================================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Python 版本
echo -e "\n${YELLOW}[1/5] 检查 Python 版本...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ 未找到 Python3${NC}"
    exit 1
fi

# 创建新的虚拟环境
echo -e "\n${YELLOW}[2/5] 创建虚拟环境...${NC}"
if [ -d "venv_yolo26" ]; then
    echo -e "${YELLOW}⚠️ venv_yolo26 已存在，跳过创建${NC}"
else
    python3 -m venv venv_yolo26
    echo -e "${GREEN}✅ 虚拟环境创建成功${NC}"
fi

# 激活虚拟环境
echo -e "\n${YELLOW}[3/5] 激活虚拟环境...${NC}"
source venv_yolo26/bin/activate
echo -e "${GREEN}✅ 虚拟环境已激活${NC}"

# 升级 pip
echo -e "\n${YELLOW}[4/5] 升级 pip...${NC}"
pip install --upgrade pip -q
echo -e "${GREEN}✅ pip 已升级${NC}"

# 安装依赖
echo -e "\n${YELLOW}[5/5] 安装依赖包...${NC}"
echo "   📦 安装基础依赖..."
pip install opencv-python numpy scipy pillow pyyaml -q

echo "   📦 安装深度学习框架..."
pip install torch torchvision ultralytics -q

echo "   📦 安装 coremltools (YOLO26 必需)..."
pip install coremltools -q

echo -e "${GREEN}✅ 所有依赖安装完成${NC}"

# 验证安装
echo -e "\n${YELLOW}验证安装...${NC}"
python3 -c "import cv2; print('  ✅ OpenCV:', cv2.__version__)"
python3 -c "import numpy; print('  ✅ NumPy:', numpy.__version__)"
python3 -c "import torch; print('  ✅ PyTorch:', torch.__version__)"
python3 -c "import coremltools; print('  ✅ coremltools:', coremltools.__version__)"

# 测试 YOLO26 模型
echo -e "\n${YELLOW}测试 YOLO26 模型...${NC}"
if [ -f "test_yolo26_model.py" ]; then
    python3 test_yolo26_model.py
else
    echo -e "${YELLOW}⚠️ 测试脚本不存在，跳过测试${NC}"
fi

echo -e "\n============================================================"
echo -e "${GREEN}✅ 环境设置完成！${NC}"
echo "============================================================"
echo -e "\n💡 下一步:"
echo "   1. 激活环境: source venv_yolo26/bin/activate"
echo "   2. 运行测试: python3 test_yolo26_model.py"
echo "   3. 处理视频: python3 main.py --config configs/yolo26_tennis_config.yaml --input 'video.mp4'"
echo ""
