#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试危险等级告警系统
验证四种告警类型的危险等级是否正确设置和显示
"""

import sys
import os
import time
import json
import requests
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from danger_recognizer import DangerRecognizer
import cv2
import numpy as np

def test_danger_levels():
    """测试危险等级设置"""
    print("=== 测试危险等级告警系统 ===")
    
    # 初始化危险识别器
    config = {
        'feature_count_threshold': 30,
        'motion_area_threshold': 0.2,
        'alert_cooldown': 5,
        'save_alerts': False
    }
    
    recognizer = DangerRecognizer(config)
    
    # 测试危险等级映射
    print("\n1. 检查危险等级映射:")
    for danger_type, level in recognizer.DANGER_LEVELS.items():
        print(f"   {danger_type}: {level}")
    
    # 创建测试帧
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟不同类型的告警
    print("\n2. 测试告警生成:")
    
    # 测试Large Area Motion (低危险)
    print("\n   - 测试Large Area Motion (低危险):")
    class DummyFeature:
        def __init__(self, pos, end_pos, mag):
            self.position = pos
            self.end_position = end_pos
            self.magnitude = mag
    
    large_area_features = [DummyFeature((100, 100), (200, 200), 10) for _ in range(100)]
    alerts = recognizer.process_frame(test_frame, large_area_features)
    for alert in alerts:
        if 'large_area_motion' in alert.get('type', '').lower():
            print(f"     告警类型: {alert['type']}")
            print(f"     危险等级: {alert.get('danger_level', 'N/A')}")
            print(f"     置信度: {alert.get('confidence', 0):.2f}")
    
    # 测试Sudden Motion (低危险)
    print("\n   - 测试Sudden Motion (低危险):")
    recognizer.last_features_count = 10
    sudden_features = [DummyFeature((100, 100), (200, 200), 15) for _ in range(80)]
    alerts = recognizer.process_frame(test_frame, sudden_features)
    for alert in alerts:
        if 'sudden_motion' in alert.get('type', '').lower():
            print(f"     告警类型: {alert['type']}")
            print(f"     危险等级: {alert.get('danger_level', 'N/A')}")
            print(f"     置信度: {alert.get('confidence', 0):.2f}")
    
    # 测试Fall Detection (高危险)
    print("\n   - 测试Fall Detection (高危险):")
    # 模拟摔倒检测的历史数据
    recognizer.history = []
    for i in range(10):
        recognizer.history.append({
            'vertical_motion': 15 if i >= 7 else 5,
            'avg_magnitude': 8 if i >= 7 else 2,
            'feature_count': 50 if i >= 8 else 20
        })
    recognizer.current_frame = 100
    fall_features = [DummyFeature((100, 100), (200, 200), 20) for _ in range(60)]
    alerts = recognizer.process_frame(test_frame, fall_features)
    for alert in alerts:
        if 'fall' in alert.get('type', '').lower():
            print(f"     告警类型: {alert['type']}")
            print(f"     危险等级: {alert.get('danger_level', 'N/A')}")
            print(f"     置信度: {alert.get('confidence', 0):.2f}")
    
    # 测试Intrusion Alert (中危险)
    print("\n   - 测试Intrusion Alert (中危险):")
    # 添加告警区域
    region = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32)
    recognizer.add_alert_region(region, "测试区域")
    
    # 模拟入侵检测
    object_detections = [{
        'bbox': [150, 150, 200, 200],
        'class': 'person',
        'confidence': 0.8
    }]
    
    alerts = recognizer.process_frame(test_frame, [], object_detections)
    for alert in alerts:
        if 'intrusion' in alert.get('type', '').lower():
            print(f"     告警类型: {alert['type']}")
            print(f"     危险等级: {alert.get('danger_level', 'N/A')}")
            print(f"     置信度: {alert.get('confidence', 0):.2f}")
    
    print("\n3. 验证告警统计:")
    stats = recognizer.get_alert_stats()
    for alert_type, count in stats.items():
        print(f"   {alert_type}: {count} 次")
    
    print("\n=== 测试完成 ===")

def test_web_interface():
    """测试Web接口的危险等级显示"""
    print("\n=== 测试Web接口 ===")
    
    try:
        # 测试告警API
        response = requests.get('http://localhost:5000/alerts', timeout=5)
        if response.status_code == 200:
            alerts = response.json()
            print(f"获取到 {len(alerts)} 条告警")
            
            for alert in alerts:
                print(f"  告警: {alert.get('type', 'N/A')}")
                print(f"    危险等级: {alert.get('danger_level', 'N/A')}")
                print(f"    时间: {alert.get('time', 'N/A')}")
                print(f"    状态: {'已处理' if alert.get('handled', False) else '未处理'}")
        else:
            print(f"API请求失败: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("无法连接到Web服务器，请确保系统正在运行")
    except Exception as e:
        print(f"测试Web接口时出错: {e}")

if __name__ == "__main__":
    test_danger_levels()
    
    # 询问是否测试Web接口
    try:
        choice = input("\n是否测试Web接口? (y/n): ").lower().strip()
        if choice == 'y':
            test_web_interface()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    
    print("\n测试脚本结束") 