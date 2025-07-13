#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重新组织后的行为检测测试 - 验证五种类型的行为检测
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

def simulate_sudden_motion(features_list, frame_count):
    """模拟突然运动"""
    features = []
    
    if frame_count == 5:  # 在第5帧突然运动
        for i in range(30):  # 大量特征点
            feature = type('Feature', (), {
                'position': (100 + i*5, 100 + i*3),
                'end_position': (110 + i*5, 105 + i*3),
                'magnitude': 8 + np.random.random() * 4,
                'data': [5 + np.random.random() * 3, 3 + np.random.random() * 2]
            })()
            features.append(feature)
    else:
        # 正常运动
        for i in range(5):
            feature = type('Feature', (), {
                'position': (100 + i*10, 100 + i*5),
                'end_position': (110 + i*10, 105 + i*5),
                'magnitude': 3 + np.random.random() * 2,
                'data': [2 + np.random.random() * 2, 1 + np.random.random()]
            })()
            features.append(feature)
    
    return features

def simulate_large_area_motion(features_list, frame_count):
    """模拟大范围运动"""
    features = []
    
    # 模拟大范围运动：大量分散的特征点
    for i in range(100):  # 大量特征点
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 480)
        feature = type('Feature', (), {
            'position': (x, y),
            'end_position': (x + 5, y + 5),
            'magnitude': 4 + np.random.random() * 3,
            'data': [3 + np.random.random() * 2, 2 + np.random.random() * 2]
        })()
        features.append(feature)
    
    return features

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

def simulate_fighting_motion(features_list, frame_count):
    """模拟打架运动"""
    features = []
    
    # 模拟剧烈的不规则运动
    for i in range(25):
        x = 150 + np.random.randint(-50, 50)
        y = 150 + np.random.randint(-50, 50)
        feature = type('Feature', (), {
            'position': (x, y),
            'end_position': (x + np.random.randint(-10, 10), y + np.random.randint(-10, 10)),
            'magnitude': 6 + np.random.random() * 4,
            'data': [5 + np.random.random() * 3, 5 + np.random.random() * 3]
        })()
        features.append(feature)
    
    return features

def test_reorganized_detection():
    """测试重新组织后的五种行为检测"""
    print("=== 重新组织后的行为检测测试 ===")
    
    # 创建危险行为识别器
    recognizer = DangerRecognizer()
    
    # 添加警戒区域
    recognizer.add_alert_region([(100, 100), (300, 100), (300, 300), (100, 300)], "测试区域")
    
    # 模拟人员检测结果
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
    
    print("1. 测试突然运动检测（不生成告警，只记录统计）")
    recognizer.reset()
    
    for frame_idx in range(10):
        frame = create_test_frame()
        features = simulate_sudden_motion([], frame_idx)
        alerts = recognizer.process_frame(frame, features, [person1])
        
        # 检查是否有告警（应该没有，因为突然运动不生成告警）
        if alerts:
            print(f"  帧 {frame_idx}: 意外生成告警: {[alert['type'] for alert in alerts]}")
    
    behavior_stats = recognizer.get_behavior_stats()
    print(f"  突然运动统计: {behavior_stats['sudden_motion_count']}")
    
    print("\n2. 测试大范围运动检测（不生成告警，只记录统计）")
    recognizer.reset()
    
    for frame_idx in range(5):
        frame = create_test_frame()
        features = simulate_large_area_motion([], frame_idx)
        alerts = recognizer.process_frame(frame, features, [person1])
        
        # 检查是否有告警（应该没有，因为大范围运动不生成告警）
        if alerts:
            print(f"  帧 {frame_idx}: 意外生成告警: {[alert['type'] for alert in alerts]}")
    
    behavior_stats = recognizer.get_behavior_stats()
    print(f"  大范围运动统计: {behavior_stats['large_area_motion_count']}")
    
    print("\n3. 测试危险区域停留检测")
    recognizer.reset()
    
    dwell_detected = False
    for frame_idx in range(30):  # 30帧，超过停留时间阈值
        frame = create_test_frame()
        features = []  # 无运动特征
        
        alerts = recognizer.process_frame(frame, features, [person1])
        
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
    
    print("\n4. 测试打架检测")
    recognizer.reset()
    
    fighting_detected = False
    for frame_idx in range(20):
        frame = create_test_frame()
        features = simulate_fighting_motion([], frame_idx)
        
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
    
    print("\n5. 测试摔倒检测")
    recognizer.reset()
    
    fall_detected = False
    for frame_idx in range(15):
        frame = create_test_frame()
        features = simulate_fall_motion([], frame_idx)
        
        alerts = recognizer.process_frame(frame, features, [person1])
        
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
    
    print("\n6. 测试冷却时间机制")
    recognizer.reset()
    
    # 先触发一个告警
    frame = create_test_frame()
    features = simulate_fall_motion([], 7)  # 使用摔倒特征
    alerts = recognizer.process_frame(frame, features, [person1])
    
    if alerts:
        print(f"  第一次检测: {alerts[0]['type']}")
        
        # 立即再次处理，应该被冷却时间阻止
        frame2 = create_test_frame()
        features2 = simulate_fall_motion([], 8)
        alerts2 = recognizer.process_frame(frame2, features2, [person1])
        
        if not alerts2:
            print("  ✅ 冷却时间机制正常工作")
        else:
            print(f"  ❌ 冷却时间机制失效，生成了告警: {[alert['type'] for alert in alerts2]}")
    else:
        print("  第一次检测未触发，无法测试冷却时间")
    
    print("\n=== 测试总结 ===")
    print(f"突然运动检测: {'正常' if behavior_stats['sudden_motion_count'] > 0 else '需要调整'}")
    print(f"大范围运动检测: {'正常' if behavior_stats['large_area_motion_count'] > 0 else '需要调整'}")
    print(f"危险区域停留: {'正常' if dwell_detected else '需要调整'}")
    print(f"打架检测: {'正常' if fighting_detected else '需要调整'}")
    print(f"摔倒检测: {'正常' if fall_detected else '需要调整'}")
    
    # 显示最终统计信息
    final_stats = recognizer.get_behavior_stats()
    print(f"\n最终行为统计:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_reorganized_detection() 