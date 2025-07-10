#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试大面积运动检测逻辑
"""

import sys
import os
sys.path.append('src')

from danger_recognizer import DangerRecognizer

def test_motion_detection():
    """测试大面积运动检测"""
    print("测试大面积运动检测逻辑...")
    
    # 创建危险行为识别器
    config = {
        'motion_area_threshold': 0.12,  # 12%
        'min_confidence': 0.6,
        'alert_highlight_duration': 30,
    }
    recognizer = DangerRecognizer(config)
    
    # 模拟运动特征数据 - 使用字典格式（光流数据）
    motion_data = [
        {'motion_vectors': 50, 'flow_mean_magnitude': 0.3},   # 帧1 - 小运动
        {'motion_vectors': 80, 'flow_mean_magnitude': 0.4},   # 帧2 - 小运动
        {'motion_vectors': 60, 'flow_mean_magnitude': 0.2},   # 帧3 - 小运动
        {'motion_vectors': 70, 'flow_mean_magnitude': 0.3},   # 帧4 - 小运动
        {'motion_vectors': 90, 'flow_mean_magnitude': 0.5},   # 帧5 - 小运动
        {'motion_vectors': 200, 'flow_mean_magnitude': 0.8},  # 帧6 - 大运动，应该触发告警
    ]
    
    # 模拟检测到的对象
    detections = [
        {
            'bbox': [100, 100, 200, 300],
            'class': 'person',
            'confidence': 0.8
        }
    ]
    
    print("模拟连续帧处理...")
    for i, data in enumerate(motion_data):
        # 创建模拟光流特征
        features = {
            'motion_vectors': [(0, 0)] * data['motion_vectors'],  # 创建对应数量的运动向量
            'flow_mean_magnitude': data['flow_mean_magnitude'],
            'flow_max_magnitude': data['flow_mean_magnitude'] * 2
        }
        
        # 创建模拟帧
        frame = None
        
        # 使用process_frame方法
        alerts = recognizer.process_frame(frame, features, detections)
        
        # 获取当前运动面积
        current_motion_area = recognizer.history[-1]['motion_area'] if recognizer.history else 0
        
        print(f"帧 {i+1}: 运动向量={data['motion_vectors']}, 运动面积={current_motion_area:.4f}, 告警数={len(alerts)}")
        if alerts:
            for alert in alerts:
                print(f"  - {alert['type']}: 置信度={alert['confidence']:.2f}")
    
    print("\n测试完成！")
    print("预期结果：")
    print("- 前5帧：无告警（运动面积较小）")
    print("- 第6帧：应该触发大面积运动告警")

if __name__ == "__main__":
    test_motion_detection() 