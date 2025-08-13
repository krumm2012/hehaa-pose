#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎾 网球分析器快速启动脚本
提供简单的菜单界面来选择不同的配置和功能
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def print_banner():
    """打印程序横幅"""
    print("🎾" * 20)
    print("   网球分析器 - 快速启动")
    print("🎾" * 20)
    print()

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查系统环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    
    # 检查必要文件
    required_files = [
        'main.py',
        'data/input_video.mp4',
        'configs/comprehensive_tennis_config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 环境检查通过")
    return True

def list_configs():
    """列出可用的配置文件"""
    config_dir = Path("configs")
    configs = []
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            configs.append(config_file.name)
    
    return sorted(configs)

def show_main_menu():
    """显示主菜单"""
    print("\n📋 请选择功能:")
    print("1. 🚀 性能优化模式 (推荐)")
    print("2. 📊 综合功能模式")
    print("3. 🎯 ROI兴趣区域演示")
    print("4. 🎾 球颜色标识演示")
    print("5. ⚡ 性能测试")
    print("6. 🔧 自定义配置")
    print("0. ❌ 退出")
    print()

def show_config_menu():
    """显示配置选择菜单"""
    configs = list_configs()
    
    print("\n📁 可用配置文件:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config}")
    print("0. 返回主菜单")
    print()

def run_config(config_name):
    """运行指定配置"""
    config_path = f"configs/{config_name}"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    print(f"\n🚀 启动配置: {config_name}")
    print("⏳ 正在启动程序...")
    
    try:
        # 启动程序
        process = subprocess.run([
            sys.executable, 'main.py', '--config', config_path
        ], check=True)
        
        print("✅ 程序执行完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 程序执行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  程序被用户中断")
        return False

def run_demo(demo_name):
    """运行演示程序"""
    demo_scripts = {
        "roi": "demo_roi_detection.py",
        "ball_color": "demo_ball_color_identification.py"
    }
    
    script_path = demo_scripts.get(demo_name)
    if not script_path or not os.path.exists(script_path):
        print(f"❌ 演示脚本不存在: {script_path}")
        return False
    
    print(f"\n🎬 启动演示: {script_path}")
    
    try:
        process = subprocess.run([
            sys.executable, script_path
        ], check=True)
        
        print("✅ 演示完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 演示执行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
        return False

def run_performance_test():
    """运行性能测试"""
    if not os.path.exists("test_performance_optimization.py"):
        print("❌ 性能测试脚本不存在")
        return False
    
    print("\n⚡ 启动性能测试...")
    
    try:
        process = subprocess.run([
            sys.executable, "test_performance_optimization.py", "compare"
        ], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 性能测试失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        return False

def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_dependencies():
        print("\n❌ 环境检查失败，请检查必要文件")
        return
    
    while True:
        show_main_menu()
        
        try:
            choice = input("请输入选择 (0-6): ").strip()
            
            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "1":
                # 性能优化模式
                success = run_config("performance_optimized_config.yaml")
                if not success:
                    print("💡 如果配置文件不存在，请先运行其他模式")
            elif choice == "2":
                # 综合功能模式
                run_config("comprehensive_tennis_config.yaml")
            elif choice == "3":
                # ROI演示
                run_demo("roi")
            elif choice == "4":
                # 球颜色标识演示
                run_demo("ball_color")
            elif choice == "5":
                # 性能测试
                run_performance_test()
            elif choice == "6":
                # 自定义配置
                show_config_menu()
                configs = list_configs()
                
                try:
                    config_choice = input("请选择配置文件编号: ").strip()
                    if config_choice == "0":
                        continue
                    
                    config_index = int(config_choice) - 1
                    if 0 <= config_index < len(configs):
                        run_config(configs[config_index])
                    else:
                        print("❌ 无效的选择")
                except ValueError:
                    print("❌ 请输入有效的数字")
            else:
                print("❌ 无效的选择，请输入0-6")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()


