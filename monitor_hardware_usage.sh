#!/bin/bash
# monitor_hardware_usage.sh
# 监控 Core ML 推理时的硬件使用情况

echo "============================================================"
echo "🔍 Core ML 硬件使用监控"
echo "============================================================"
echo ""
echo "本脚本将监控以下指标:"
echo "  - CPU 使用率"
echo "  - GPU 使用率"
echo "  - ANE (Apple Neural Engine) 使用率"
echo "  - 内存使用"
echo "  - 功耗"
echo ""
echo "⚠️ 需要 sudo 权限来访问 powermetrics"
echo ""

# 检查是否有 sudo 权限
if ! sudo -n true 2>/dev/null; then
    echo "🔐 请输入密码以获取 sudo 权限..."
fi

echo "============================================================"
echo "📊 开始监控（按 Ctrl+C 停止）"
echo "============================================================"
echo ""

# 使用 powermetrics 监控
# -i 1000: 每1秒采样一次
# --samplers: 指定要监控的采样器
sudo powermetrics \
    --samplers cpu_power,gpu_power,ane_power,tasks \
    -i 1000 \
    --show-process-coalition \
    --show-process-gpu \
    --show-process-energy \
    | grep -E "(Python|ANE|GPU|CPU|Energy|Task)"

# 备选方案（如果 powermetrics 不可用）
# 使用 top 命令
# top -pid $(pgrep -f "python3 main.py") -stats pid,command,cpu,gpu,mem
