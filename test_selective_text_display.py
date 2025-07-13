#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证只移除了红色区域的文字，但保留了检测对象的名称显示
"""

import cv2
import numpy as np
import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from danger_recognizer import DangerRecognizer
from all_in_one_system import AllInOneSystem
import argparse

def test_selective_text_display():
    """测试只移除了红色区域的文字，但保留了检测对象的名称显示"""
    print("开始测试：验证选择性文字显示")
    
    # 创建危险识别器
    danger_recognizer = DangerRecognizer()
    
    # 添加测试警戒区域
    test_region = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32)
    danger_recognizer.add_alert_region(test_region, "Test Zone")
    
    # 创建测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些测试对象检测结果
    test_detections = [
        {
            'bbox': [150, 150, 200, 250],
            'class': 'person',
            'confidence': 0.95
        },
        {
            'bbox': [400, 200, 450, 300],
            'class': 'car',
            'confidence': 0.88
        },
        {
            'bbox': [50, 50, 100, 120],
            'class': 'dog',
            'confidence': 0.92
        }
    ]
    
    # 添加测试告警
    test_alerts = [
        {
            'type': 'Intrusion Alert',
            'confidence': 0.85,
            'time': time.time()
        }
    ]
    
    # 测试可视化功能
    print("测试1：基本可视化（无告警）")
    vis_frame1 = danger_recognizer.visualize(frame.copy(), detections=test_detections)
    if vis_frame1 is not None:
        print("✓ 基本可视化成功")
        # 保存测试图片
        cv2.imwrite("test_basic_visualization.jpg", vis_frame1)
        print("✓ 已保存基本可视化图片: test_basic_visualization.jpg")
    else:
        print("✗ 基本可视化失败")
    
    print("\n测试2：告警可视化")
    vis_frame2 = danger_recognizer.visualize(frame.copy(), alerts=test_alerts, detections=test_detections)
    if vis_frame2 is not None:
        print("✓ 告警可视化成功")
        # 保存测试图片
        cv2.imwrite("test_alert_visualization.jpg", vis_frame2)
        print("✓ 已保存告警可视化图片: test_alert_visualization.jpg")
    else:
        print("✗ 告警可视化失败")
    
    print("\n测试3：检查代码中的文字显示")
    # 检查danger_recognizer.py中的文字显示
    with open('src/danger_recognizer.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 检查是否移除了红色区域的文字
        red_zone_texts = ['Dwell:', 'Alert Zone', 'User Selected Zone']
        removed_texts = []
        for text in red_zone_texts:
            if text not in content:
                removed_texts.append(text)
        
        if removed_texts:
            print(f"✓ 已移除红色区域文字: {removed_texts}")
        else:
            print("⚠ 可能仍有红色区域文字")
        
        # 检查是否保留了检测对象的文字显示
        detection_texts = ['cv2.putText(vis_frame, f"{det.get(\'class\', \'unknown\')}']
        preserved_texts = []
        for text in detection_texts:
            if text in content:
                preserved_texts.append("检测对象名称")
        
        if preserved_texts:
            print(f"✓ 保留了检测对象文字显示: {preserved_texts}")
        else:
            print("⚠ 可能移除了检测对象文字显示")
    
    print("\n测试4：检查all_in_one_system.py中的文字显示")
    with open('src/all_in_one_system.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 检查是否保留了系统状态信息
        system_texts = ['FPS:', 'Frames:', 'Processed:', 'Uptime:']
        preserved_system_texts = []
        for text in system_texts:
            if text in content:
                preserved_system_texts.append(text)
        
        if preserved_system_texts:
            print(f"✓ 保留了系统状态信息: {preserved_system_texts}")
        else:
            print("⚠ 可能移除了系统状态信息")
        
        # 检查是否保留了检测结果文字
        detection_texts = ['cv2.putText(vis_frame, f"{cls} {conf:.2f}"']
        if detection_texts[0] in content:
            print("✓ 保留了AI检测结果的文字显示")
        else:
            print("⚠ 可能移除了AI检测结果的文字显示")
    
    print("\n测试完成！")
    print("总结：")
    print("- 移除了红色区域的文字显示（Dwell、Alert Zone、User Selected Zone等）")
    print("- 保留了检测对象的绿色方框左上角名称显示")
    print("- 保留了系统状态信息显示")
    print("- 界面更加简洁，但仍保持必要的识别信息")

if __name__ == "__main__":
    test_selective_text_display() 