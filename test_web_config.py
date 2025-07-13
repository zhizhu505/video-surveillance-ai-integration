#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Web配置功能
验证前端的时间阈值设置和警戒区域框选功能
"""

import requests
import json
import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_web_config():
    """测试Web配置功能"""
    base_url = "http://localhost:5000"
    
    print("=== 测试Web配置功能 ===")
    
    # 测试1: 设置停留时间阈值
    print("\n1. 测试设置停留时间阈值...")
    try:
        response = requests.post(
            f"{base_url}/config/dwell_time_threshold",
            json={"threshold": 2.5},
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        print(f"响应: {result}")
        if result.get('success'):
            print("✓ 时间阈值设置成功")
        else:
            print("✗ 时间阈值设置失败")
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    
    # 测试2: 设置警戒区域
    print("\n2. 测试设置警戒区域...")
    try:
        # 模拟用户框选的矩形区域
        region = [
            [100, 100],  # 左上角
            [300, 100],  # 右上角
            [300, 300],  # 右下角
            [100, 300]   # 左下角
        ]
        response = requests.post(
            f"{base_url}/config/alert_region",
            json={"region": region},
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        print(f"响应: {result}")
        if result.get('success'):
            print("✓ 警戒区域设置成功")
        else:
            print("✗ 警戒区域设置失败")
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    
    # 测试3: 重置警戒区域
    print("\n3. 测试重置警戒区域...")
    try:
        response = requests.post(
            f"{base_url}/config/reset_alert_region",
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        print(f"响应: {result}")
        if result.get('success'):
            print("✓ 警戒区域重置成功")
        else:
            print("✗ 警戒区域重置失败")
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    
    # 测试4: 获取系统状态
    print("\n4. 测试获取系统状态...")
    try:
        response = requests.get(f"{base_url}/stats")
        result = response.json()
        print(f"系统状态: {result}")
        print("✓ 系统状态获取成功")
    except Exception as e:
        print(f"✗ 获取系统状态失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("\n使用说明:")
    print("1. 启动系统: python src/all_in_one_system.py --web_interface --source 0")
    print("2. 打开浏览器访问: http://localhost:5000")
    print("3. 在右侧配置面板中:")
    print("   - 输入停留时间阈值（秒）")
    print("   - 点击'开始框选'按钮")
    print("   - 在视频上拖拽鼠标框选警戒区域")
    print("   - 点击'重置区域'可以清除警戒区域")

if __name__ == "__main__":
    test_web_config() 