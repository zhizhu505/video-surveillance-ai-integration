#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进后的检测测试 - 验证打架检测灵敏度和摔倒检测准确性
"""

import sys
import os
import cv2
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from danger_recognizer import DangerRecognizer

def test_fighting_sensitivity():
    """测试打架检测的灵敏度"""
    print("=== 打架检测灵敏度测试 ===")
    print("当前参数:")
    print("- 距离阈值: 80px")
    print("- 运动强度阈值: 4")
    print("- 持续时间: 8帧")
    print("- 置信度阈值: 0.5")
    print()
    
    # 创建识别器
    recognizer = DangerRecognizer()
    
    # 测试场景
    test_scenarios = [
        {
            'name': '轻微接触（不应该触发）',
            'person1_pos': [200, 200],
            'person2_pos': [300, 200],
            'motion_intensity': 3,
            'expected': False
        },
        {
            'name': '近距离接触（应该触发）',
            'person1_pos': [200, 200],
            'person2_pos': [220, 200],
            'motion_intensity': 5,
            'expected': True
        },
        {
            'name': '打架行为（应该触发）',
            'person1_pos': [200, 200],
            'person2_pos': [210, 200],
            'motion_intensity': 8,
            'expected': True
        }
    ]
    
    for scenario in test_scenarios:
        print(f"测试: {scenario['name']}")
        
        # 模拟多帧
        detected = False
        for frame_idx in range(15):  # 15帧测试
            # 创建测试帧
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # 模拟人物位置
            person1_pos = [
                scenario['person1_pos'][0] + int(3 * np.sin(frame_idx * 0.2)),
                scenario['person1_pos'][1] + int(2 * np.cos(frame_idx * 0.1))
            ]
            person2_pos = [
                scenario['person2_pos'][0] + int(3 * np.sin(frame_idx * 0.2 + np.pi)),
                scenario['person2_pos'][1] + int(2 * np.cos(frame_idx * 0.1 + np.pi))
            ]
            
            # 模拟检测结果
            detections = [
                {
                    'class': 'person',
                    'bbox': [person1_pos[0] - 20, person1_pos[1] - 30, person1_pos[0] + 20, person1_pos[1] + 30],
                    'confidence': 0.9,
                    'person_id': 1
                },
                {
                    'class': 'person',
                    'bbox': [person2_pos[0] - 20, person2_pos[1] - 30, person2_pos[0] + 20, person2_pos[1] + 30],
                    'confidence': 0.9,
                    'person_id': 2
                }
            ]
            
            # 模拟运动特征
            features = {
                'flow_mean_magnitude': scenario['motion_intensity'],
                'flow_max_magnitude': scenario['motion_intensity'] * 1.5,
                'motion_vectors': []
            }
            
            # 生成运动向量
            for i in range(15):
                x = np.random.randint(0, 640)
                y = np.random.randint(0, 480)
                dx = np.random.normal(0, max(0.1, scenario['motion_intensity'] * 0.3))
                dy = np.random.normal(0, max(0.1, scenario['motion_intensity'] * 0.3))
                features['motion_vectors'].append([x, y, dx, dy])
            
            # 处理帧
            alerts = recognizer.process_frame(frame, features, detections)
            
            # 检查是否检测到打架
            for alert in alerts:
                if alert['type'] == 'Fighting Detection':
                    detected = True
                    print(f"  ✅ 检测到打架行为！")
                    print(f"     置信度: {alert['confidence']:.2f}")
                    print(f"     条件: {alert['condition_details']}")
                    print(f"     像素距离: {alert.get('pixel_distance', 0):.1f}")
                    print(f"     持续时间: {alert.get('duration', 0)}帧")
                    break
            
            if detected:
                break
        
        # 输出结果
        if detected == scenario['expected']:
            print(f"  ✅ 结果正确")
        else:
            print(f"  ❌ 结果错误 - 期望: {scenario['expected']}, 实际: {detected}")
        print()

def test_fall_detection_with_camera_motion():
    """测试摔倒检测对摄像头移动的处理"""
    print("=== 摔倒检测摄像头移动测试 ===")
    
    # 创建识别器
    recognizer = DangerRecognizer()
    
    # 测试场景
    test_scenarios = [
        {
            'name': '正常摔倒（应该触发）',
            'has_person': True,
            'vertical_motion': 8,
            'motion_area': 0.1,  # 小范围运动
            'direction_consistency': 1.0,  # 低方向一致性
            'expected': True
        },
        {
            'name': '摄像头向下移动（不应该触发）',
            'has_person': True,
            'vertical_motion': 8,
            'motion_area': 0.6,  # 大范围运动
            'direction_consistency': 5.0,  # 高方向一致性
            'expected': False
        },
        {
            'name': '无人员（不应该触发）',
            'has_person': False,
            'vertical_motion': 8,
            'motion_area': 0.1,
            'direction_consistency': 1.0,
            'expected': False
        }
    ]
    
    for scenario in test_scenarios:
        print(f"测试: {scenario['name']}")
        
        # 模拟多帧
        detected = False
        for frame_idx in range(15):  # 15帧测试
            # 创建测试帧
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # 模拟检测结果
            detections = []
            if scenario['has_person']:
                detections = [
                    {
                        'class': 'person',
                        'bbox': [200, 200, 240, 260],
                        'confidence': 0.9,
                        'person_id': 1
                    }
                ]
            
            # 模拟运动特征
            features = {
                'flow_mean_magnitude': 5,
                'flow_max_magnitude': 8,
                'motion_vectors': []
            }
            
            # 生成运动向量（模拟摄像头移动或正常运动）
            for i in range(20):
                x = np.random.randint(0, 640)
                y = np.random.randint(0, 480)
                
                if scenario['direction_consistency'] > 3.0:
                    # 模拟摄像头移动：方向一致
                    dx = np.random.normal(scenario['direction_consistency'] * 0.5, 0.5)
                    dy = np.random.normal(scenario['direction_consistency'] * 0.3, 0.5)
                else:
                    # 模拟正常运动：方向随机
                    dx = np.random.normal(0, 2)
                    dy = np.random.normal(scenario['vertical_motion'] * 0.5, 1)
                
                features['motion_vectors'].append([x, y, dx, dy])
            
            # 处理帧
            alerts = recognizer.process_frame(frame, features, detections)
            
            # 检查是否检测到摔倒
            for alert in alerts:
                if alert['type'] == 'Fall Detection':
                    detected = True
                    print(f"  ✅ 检测到摔倒行为！")
                    print(f"     置信度: {alert['confidence']:.2f}")
                    print(f"     条件: {alert['condition_details']}")
                    break
            
            if detected:
                break
        
        # 输出结果
        if detected == scenario['expected']:
            print(f"  ✅ 结果正确")
        else:
            print(f"  ❌ 结果错误 - 期望: {scenario['expected']}, 实际: {detected}")
        print()

if __name__ == "__main__":
    print("改进后的检测测试")
    print("=" * 50)
    
    # 测试打架检测灵敏度
    test_fighting_sensitivity()
    
    # 测试摔倒检测对摄像头移动的处理
    test_fall_detection_with_camera_motion()
    
    print("测试完成！") 