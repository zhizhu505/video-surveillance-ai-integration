#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
摔倒检测清理测试 - 验证删除重复定义后的效果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cv2
import time
from danger_recognizer import DangerRecognizer

def create_test_frame(width=640, height=480):
    """创建测试帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # 添加一些背景纹理
    cv2.rectangle(frame, (0, 0), (width, height), (50, 50, 50), -1)
    return frame

def simulate_fall_motion(features_list, frame_count):
    """模拟摔倒运动特征"""
    features = []
    
    # 模拟摔倒的运动模式：先剧烈运动，然后突然停止
    if frame_count < 5:
        # 正常运动
        for i in range(10):
            feature = type('Feature', (), {
                'position': (100 + i*10, 100 + i*5),
                'end_position': (110 + i*10, 105 + i*5),
                'magnitude': 3 + np.random.random() * 2,
                'data': [2 + np.random.random() * 2, 1 + np.random.random()]
            })()
            features.append(feature)
    elif frame_count < 8:
        # 剧烈向下运动（摔倒）
        for i in range(15):
            feature = type('Feature', (), {
                'position': (100 + i*8, 100 + i*8),
                'end_position': (108 + i*8, 120 + i*8),  # 向下运动
                'magnitude': 8 + np.random.random() * 4,
                'data': [3 + np.random.random() * 3, 8 + np.random.random() * 4]  # 大的垂直运动
            })()
            features.append(feature)
    else:
        # 运动后静止
        for i in range(5):
            feature = type('Feature', (), {
                'position': (100 + i*5, 150 + i*2),
                'end_position': (105 + i*5, 152 + i*2),
                'magnitude': 1 + np.random.random(),
                'data': [0.5 + np.random.random(), 0.5 + np.random.random()]
            })()
            features.append(feature)
    
    return features

def simulate_camera_motion(features_list, frame_count):
    """模拟摄像头移动"""
    features = []
    
    # 模拟摄像头移动：大面积、方向一致的运动
    for i in range(50):  # 大量特征点
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 480)
        # 一致向右移动
        feature = type('Feature', (), {
            'position': (x, y),
            'end_position': (x + 10, y),  # 水平移动
            'magnitude': 5 + np.random.random() * 3,
            'data': [8 + np.random.random() * 4, 1 + np.random.random()]  # 大的水平运动
        })()
        features.append(feature)
    
    return features

def test_fall_detection_cleanup():
    """测试摔倒检测清理后的效果"""
    print("=== 摔倒检测清理测试 ===")
    
    # 创建危险行为识别器
    recognizer = DangerRecognizer()
    
    # 添加警戒区域
    recognizer.add_alert_region([(100, 100), (300, 100), (300, 300), (100, 300)], "测试区域")
    
    # 模拟人员检测结果
    person_detection = {
        'class': 'person',
        'confidence': 0.9,
        'bbox': [150, 150, 250, 350],
        'person_id': 1
    }
    
    print("1. 测试正常摔倒检测（无摄像头移动）")
    fall_detected = False
    for frame_idx in range(15):
        frame = create_test_frame()
        
        # 模拟摔倒运动
        features = simulate_fall_motion([], frame_idx)
        
        # 处理帧
        alerts = recognizer.process_frame(frame, features, [person_detection])
        
        # 检查摔倒检测
        for alert in alerts:
            if alert['type'] == 'Fall Detection':
                print(f"  帧 {frame_idx}: 检测到摔倒！置信度={alert['confidence']:.2f}")
                print(f"    描述: {alert['desc']}")
                fall_detected = True
                break
        
        if fall_detected:
            break
    
    if not fall_detected:
        print("  未检测到摔倒（可能需要调整参数）")
    
    print("\n2. 测试摄像头移动时的摔倒检测（应该被忽略）")
    recognizer.reset()  # 重置状态
    
    camera_motion_ignored = True
    for frame_idx in range(10):
        frame = create_test_frame()
        
        # 模拟摄像头移动
        features = simulate_camera_motion([], frame_idx)
        
        # 处理帧
        alerts = recognizer.process_frame(frame, features, [person_detection])
        
        # 检查是否错误地检测到摔倒
        for alert in alerts:
            if alert['type'] == 'Fall Detection':
                print(f"  帧 {frame_idx}: 错误检测到摔倒！置信度={alert['confidence']:.2f}")
                camera_motion_ignored = False
                break
    
    if camera_motion_ignored:
        print("  正确忽略摄像头移动，未产生摔倒误报")
    
    print("\n3. 测试其他告警类型是否正常工作")
    recognizer.reset()
    
    # 测试危险区域停留检测
    dwell_detected = False
    for frame_idx in range(30):  # 30帧，超过停留时间阈值
        frame = create_test_frame()
        features = []  # 无运动特征
        
        alerts = recognizer.process_frame(frame, features, [person_detection])
        
        for alert in alerts:
            if alert['type'] == 'Danger Zone Dwell':
                print(f"  帧 {frame_idx}: 检测到危险区域停留！")
                print(f"    描述: {alert['desc']}")
                dwell_detected = True
                break
        
        if dwell_detected:
            break
    
    if not dwell_detected:
        print("  未检测到危险区域停留（可能需要调整参数）")
    
    print("\n4. 测试打架检测是否正常工作")
    recognizer.reset()
    
    # 模拟两个人员
    person1 = {
        'class': 'person',
        'confidence': 0.9,
        'bbox': [150, 150, 250, 350],
        'person_id': 1
    }
    person2 = {
        'class': 'person',
        'confidence': 0.9,
        'bbox': [200, 150, 300, 350],  # 与person1重叠
        'person_id': 2
    }
    
    fighting_detected = False
    for frame_idx in range(20):
        frame = create_test_frame()
        
        # 模拟剧烈运动
        features = []
        for i in range(20):
            feature = type('Feature', (), {
                'position': (150 + i*5, 150 + i*5),
                'end_position': (155 + i*5, 155 + i*5),
                'magnitude': 6 + np.random.random() * 4,
                'data': [5 + np.random.random() * 3, 5 + np.random.random() * 3]
            })()
            features.append(feature)
        
        alerts = recognizer.process_frame(frame, features, [person1, person2])
        
        for alert in alerts:
            if alert['type'] == 'Fighting Detection':
                print(f"  帧 {frame_idx}: 检测到打架！置信度={alert['confidence']:.2f}")
                print(f"    描述: {alert['desc']}")
                fighting_detected = True
                break
        
        if fighting_detected:
            break
    
    if not fighting_detected:
        print("  未检测到打架（可能需要调整参数）")
    
    print("\n=== 测试总结 ===")
    print(f"摔倒检测: {'正常' if fall_detected else '需要调整'}")
    print(f"摄像头移动过滤: {'正常' if camera_motion_ignored else '有问题'}")
    print(f"危险区域停留: {'正常' if dwell_detected else '需要调整'}")
    print(f"打架检测: {'正常' if fighting_detected else '需要调整'}")
    
    # 显示统计信息
    behavior_stats = recognizer.get_behavior_stats()
    print(f"\n行为统计:")
    for key, value in behavior_stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_fall_detection_cleanup() 