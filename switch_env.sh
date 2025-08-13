#!/bin/bash
# switch_env.sh - 环境切换脚本

echo "🎾 网球分析系统 - 环境切换脚本"
echo "================================"
echo ""

# 检查可用环境
echo "📋 可用环境:"
echo "1. Python 3.11 + MTCNN (venv_311) - 最高精度"
echo "2. Python 3.13 + 增强OpenCV (venv) - 高兼容性"
echo ""

# 获取用户选择
read -p "请选择环境 (1/2): " choice

case $choice in
    1)
        echo "🔄 切换到Python 3.11 + MTCNN环境..."
        
        # 停用当前环境
        deactivate 2>/dev/null || true
        
        # 设置Python版本
        pyenv local 3.11.7
        
        # 激活虚拟环境
        source venv_311/bin/activate
        
        echo "✅ 环境切换完成！"
        echo "🧠 MTCNN深度学习检测已启用"
        echo "📊 当前Python版本: $(python --version)"
        echo "🎯 检测器: MTCNN + 增强OpenCV"
        ;;
    2)
        echo "🔄 切换到Python 3.13 + 增强OpenCV环境..."
        
        # 停用当前环境
        deactivate 2>/dev/null || true
        
        # 设置Python版本
        pyenv local 3.13.3
        
        # 激活虚拟环境
        source venv/bin/activate
        
        echo "✅ 环境切换完成！"
        echo "🔧 增强OpenCV检测已启用"
        echo "📊 当前Python版本: $(python --version)"
        echo "🎯 检测器: 增强OpenCV (高兼容性)"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "💡 提示:"
echo "  - 运行 'python test_system_final.py' 测试当前环境"
echo "  - 运行 'python main.py' 开始视频分析"
echo "  - 运行 './switch_env.sh' 切换到其他环境" 