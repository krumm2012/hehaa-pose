#!/bin/bash
# setup_mtcnn_env.sh
# MTCNN环境设置脚本 - 为网球分析系统创建兼容的Python环境

echo "🎾 网球分析系统 - MTCNN环境设置脚本"
echo "====================================="
echo ""

# 检查当前Python版本
CURRENT_PYTHON=$(python --version 2>&1)
echo "当前Python版本: $CURRENT_PYTHON"

if [[ $CURRENT_PYTHON == *"3.13"* ]]; then
    echo "⚠️  检测到Python 3.13，MTCNN不兼容此版本"
    echo "🔧 将为您创建Python 3.11兼容环境"
    NEED_NEW_ENV=true
elif [[ $CURRENT_PYTHON == *"3.12"* ]]; then
    echo "⚠️  Python 3.12可能存在兼容性问题"
    echo "🔧 建议创建Python 3.11环境以确保稳定性"
    read -p "是否创建新的Python 3.11环境？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        NEED_NEW_ENV=true
    else
        NEED_NEW_ENV=false
    fi
else
    echo "✅ 当前Python版本支持MTCNN"
    NEED_NEW_ENV=false
fi

# 检查pyenv是否安装
if command -v pyenv >/dev/null 2>&1; then
    echo "✅ pyenv已安装"
else
    echo "❌ pyenv未安装"
    echo "🔧 正在安装pyenv..."
    if command -v brew >/dev/null 2>&1; then
        brew install pyenv
        echo "✅ pyenv安装完成"
        echo "📝 请将以下内容添加到您的shell配置文件 (~/.zshrc 或 ~/.bash_profile):"
        echo "export PATH=\"\$HOME/.pyenv/bin:\$PATH\""
        echo "eval \"\$(pyenv init --path)\""
        echo "eval \"\$(pyenv init -)\""
        echo ""
        echo "然后运行: source ~/.zshrc (或重启终端)"
        echo "⚠️  配置完成后请重新运行此脚本"
        exit 1
    else
        echo "❌ 未检测到brew，请手动安装pyenv"
        echo "参考: https://github.com/pyenv/pyenv#installation"
        exit 1
    fi
fi

if [ "$NEED_NEW_ENV" = true ]; then
    echo ""
    echo "🏗️  创建Python 3.11环境..."
    
    # 安装Python 3.11.7
    echo "📦 安装Python 3.11.7..."
    pyenv install 3.11.7 2>/dev/null || echo "Python 3.11.7可能已安装"
    
    # 设置本地Python版本
    echo "🔧 设置项目Python版本..."
    pyenv local 3.11.7
    
    # 验证Python版本
    NEW_PYTHON=$(python --version 2>&1)
    echo "新Python版本: $NEW_PYTHON"
    
    # 创建虚拟环境
    echo "🏗️  创建虚拟环境 (venv_mtcnn)..."
    python -m venv venv_mtcnn
    
    echo "✅ Python 3.11环境创建完成！"
    echo ""
    echo "🚀 下一步操作："
    echo "1. 激活环境: source venv_mtcnn/bin/activate"
    echo "2. 安装MTCNN: pip install tensorflow mtcnn"
    echo "3. 安装其他依赖: pip install -r requirements.txt"
    echo "4. 在配置中启用MTCNN"
    echo ""
else
    echo ""
    echo "🚀 在当前环境中安装MTCNN..."
    
    # 检查TensorFlow
    if python -c "import tensorflow" 2>/dev/null; then
        echo "✅ TensorFlow已安装"
    else
        echo "📦 安装TensorFlow..."
        pip install tensorflow
    fi
    
    # 检查MTCNN
    if python -c "import mtcnn" 2>/dev/null; then
        echo "✅ MTCNN已安装"
    else
        echo "📦 安装MTCNN..."
        pip install mtcnn
    fi
    
    echo "✅ MTCNN安装完成！"
fi

# 提供配置更新指导
echo ""
echo "⚙️  配置更新指导:"
echo "在 configs/default_config.yaml 中设置:"
echo "advanced_face_detection:"
echo "  mtcnn:"
echo "    enabled: true"
echo ""

# 运行测试脚本
echo "🧪 运行MTCNN测试..."
if [ "$NEED_NEW_ENV" = true ]; then
    echo "请先激活新环境再运行测试："
    echo "source venv_mtcnn/bin/activate"
    echo "python test_mtcnn_installation.py"
else
    python test_mtcnn_installation.py
fi

echo ""
echo "📚 完整文档请参考: MTCNN_INSTALLATION_GUIDE.md"
echo "🎉 设置完成！" 