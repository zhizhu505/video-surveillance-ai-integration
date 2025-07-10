#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门测试摔倒检测的脚本
"""

import sys
import os
# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

import numpy as np
import cv2
from danger_recognizer import DangerRecognizer

def create_fall_only_simulation():
    """创建只包含摔倒的模拟数据，避免其他告警干扰"""
    features_list = []
    
    # 摔倒前：轻微运动（5帧）
    for i in range(5):
        features = {
            'flow_mean_magnitude': 2.0,
            'flow_max_magnitude': 4.0,
            'motion_vectors': [(1, 0), (0, 1), (-1, 0), (0, -1)] * 5  # 轻微运动
        }
        features_list.append(features)
    
    # 摔倒时：剧烈向下运动（5帧）
    for i in range(5):
        features = {
            'flow_mean_magnitude': 15.0 + i * 2.0,  # 大幅增加
            'flow_max_magnitude': 25.0 + i * 3.0,   # 大幅增加
            'motion_vectors': [(0, 20), (0, 22), (0, 25), (1, 20), (-1, 22)] * 25  # 剧烈向下运动
        }
        features_list.append(features)
    
    # 摔倒后：静止（5帧）
    for i in range(5):
        features = {
            'flow_mean_magnitude': 1.0,
            'flow_max_magnitude': 2.0,
            'motion_vectors': [(0, 0), (0, 0), (0, 0)] * 8  # 静止向量
        }
        features_list.append(features)
    
    return features_list

def test_fall_only():
    """专门测试摔倒检测"""
    print("=== 摔倒检测专项测试 ===")
    
    # 创建危险行为识别器
    recognizer = DangerRecognizer()
    
    # 创建测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 创建摔倒模拟数据
    features_list = create_fall_only_simulation()
    
    # 模拟处理每一帧
    for i, features in enumerate(features_list):
        print(f"\n--- 帧 {i+1} ---")
        
        # 处理帧
        alerts = recognizer.process_frame(frame, features)
        # 打印vertical_motion调试值
        print(f'vertical_motion: {recognizer.history[-1]["vertical_motion"]:.2f}')
        print(f'历史记录长度: {len(recognizer.history)}')
        
        # 如果历史记录足够，手动检查摔倒检测条件
        if len(recognizer.history) >= 10:
            recent_vertical_motions = [h['vertical_motion'] for h in recognizer.history[-8:]]
            max_vertical_motion = max(recent_vertical_motions)
            recent_magnitudes = [h['avg_magnitude'] for h in recognizer.history[-8:]]
            recent_avg = sum(recent_magnitudes[-3:]) / 3
            earlier_avg = sum(recent_magnitudes[:-3]) / (len(recent_magnitudes) - 3) if len(recent_magnitudes) > 3 else 0
            print(f'手动检查: max_vertical_motion={max_vertical_motion:.2f}, earlier_avg={earlier_avg:.2f}, recent_avg={recent_avg:.2f}')
        
        # 显示结果
        if alerts:
            for alert in alerts:
                print(f"检测到告警: {alert['type']} (置信度: {alert['confidence']:.2f})")
        else:
            print("无告警")
    
    # 显示统计信息
    print(f"\n=== 统计信息 ===")
    stats = recognizer.get_alert_stats()
    for alert_type, count in stats.items():
        if count > 0:
            print(f"{alert_type}: {count}")

if __name__ == "__main__":
    test_fall_only() 