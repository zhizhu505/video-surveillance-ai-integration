#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web界面功能演示
展示新的前端配置功能
"""

import time
import sys
import os

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🎥 视频监控系统 - Web界面配置功能演示")
    print("=" * 60)
    print()

def print_features():
    """打印功能特性"""
    print("✨ 新功能特性:")
    print("  1. 📊 实时停留时间阈值设置")
    print("  2. 🖱️  鼠标框选警戒区域")
    print("  3. 🔄 动态配置更新")
    print("  4. 📱 响应式Web界面")
    print("  5. ⚡ 实时告警监控")
    print()

def print_usage():
    """打印使用说明"""
    print("📖 使用说明:")
    print("  1. 启动系统:")
    print("     python src/all_in_one_system.py --web_interface --source 0")
    print("     或者运行: run_web_system.bat")
    print()
    print("  2. 打开浏览器访问: http://localhost:5000")
    print()
    print("  3. 配置功能:")
    print("     • 在右侧面板输入停留时间阈值（秒）")
    print("     • 点击'设置阈值'按钮")
    print("     • 点击'开始框选'按钮")
    print("     • 在视频上拖拽鼠标框选警戒区域")
    print("     • 点击'重置区域'清除警戒区域")
    print()

def print_api_info():
    """打印API信息"""
    print("🔧 API接口:")
    print("  POST /config/dwell_time_threshold - 设置停留时间阈值")
    print("  POST /config/alert_region - 设置警戒区域")
    print("  POST /config/reset_alert_region - 重置警戒区域")
    print()

def print_test_info():
    """打印测试信息"""
    print("🧪 测试功能:")
    print("  运行测试脚本: python test_web_config.py")
    print("  验证API接口是否正常工作")
    print()

def main():
    """主函数"""
    print_banner()
    print_features()
    print_usage()
    print_api_info()
    print_test_info()
    
    print("🚀 准备启动系统...")
    print("按 Enter 键继续，或按 Ctrl+C 退出")
    
    try:
        input()
        print("\n正在启动系统...")
        print("请在浏览器中访问: http://localhost:5000")
        print("按 Ctrl+C 停止系统")
        
        # 启动系统
        os.system("python src/all_in_one_system.py --web_interface --source 0 --width 640 --height 480 --max_fps 30")
        
    except KeyboardInterrupt:
        print("\n\n👋 系统已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("请检查:")
        print("  1. Python环境是否正确")
        print("  2. 依赖包是否已安装: pip install -r requirements.txt")
        print("  3. 摄像头是否可用")

if __name__ == "__main__":
    main() 