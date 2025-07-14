#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
打架检测灵敏度测试脚本 - 验证调整后的参数
"""

import sys
import os
import cv2
import numpy as np
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from danger_recognizer import DangerRecognizer

def test_fighting_sensitivity():
    """测试打架检测的灵敏度"""
    print("=== 打架检测灵敏度测试 ===")
    
    # 测试不同的参数配置
    test_configs = [
        {
            'name': '保守配置（减少误报）',
            'config': {
                'fighting_distance_threshold': 120,
                'fighting_motion_threshold': 8,
                'fighting_duration_frames': 20,
                'fighting_confidence_threshold': 0.8,
                'save_alerts': False
            }
        },
        {
            'name': '中等配置',
            'config': {
                'fighting_distance_threshold': 150,
                'fighting_motion_threshold': 6,
                'fighting_duration_frames': 15,
                'fighting_confidence_threshold': 0.7,
                'save_alerts': False
            }
        },
        {
            'name': '灵敏配置（容易触发）',
            'config': {
                'fighting_distance_threshold': 200,
                'fighting_motion_threshold': 4,
                'fighting_duration_frames': 10,
                'fighting_confidence_threshold': 0.6,
                'save_alerts': False
            }
        }
    ]
    
    for test_config in test_configs:
        print(f"\n--- {test_config['name']} ---")
        
        # 创建识别器
        recognizer = DangerRecognizer(test_config['config'])
        
        # 模拟不同的场景
        scenarios = [
            {
                'name': '正常行走（应该不触发）',
                'person1_pos': [100, 200],
                'person2_pos': [300, 200],
                'motion_intensity': 3,
                'should_detect': False
            },
            {
                'name': '近距离接触（可能触发）',
                'person1_pos': [200, 200],
                'person2_pos': [220, 200],
                'motion_intensity': 5,
                'should_detect': False
            },
            {
                'name': '打架行为（应该触发）',
                'person1_pos': [200, 200],
                'person2_pos': [210, 200],
                'motion_intensity': 10,
                'should_detect': True
            }
        ]
        
        for scenario in scenarios:
            print(f"\n测试场景: {scenario['name']}")
            
            # 模拟多帧
            detected = False
            for frame_idx in range(30):  # 30帧测试
                # 创建测试帧
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
                
                # 模拟人物位置（添加一些随机运动）
                person1_pos = [
                    scenario['person1_pos'][0] + int(5 * np.sin(frame_idx * 0.3)),
                    scenario['person1_pos'][1] + int(3 * np.cos(frame_idx * 0.2))
                ]
                person2_pos = [
                    scenario['person2_pos'][0] + int(5 * np.sin(frame_idx * 0.3 + np.pi)),
                    scenario['person2_pos'][1] + int(3 * np.cos(frame_idx * 0.2 + np.pi))
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
                for i in range(20):
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
                        print(f"     估算实际距离: {alert.get('real_distance', 0):.1f}")
                        break
                
                if detected:
                    break
            
            # 输出结果
            if detected == scenario['should_detect']:
                print(f"  ✅ 结果正确")
            else:
                print(f"  ❌ 结果错误 - 期望: {scenario['should_detect']}, 实际: {detected}")
        
        # 输出行为统计
        behavior_stats = recognizer.get_behavior_stats()
        print(f"\n行为统计:")
        for behavior, count in behavior_stats.items():
            print(f"  {behavior}: {count}")

def test_distance_estimation_improved():
    """测试改进后的距离估算"""
    print("\n=== 改进后的距离估算测试 ===")
    
    recognizer = DangerRecognizer()
    
    # 测试不同场景的距离估算
    test_cases = [
        # 近距离，画面中心，相似大小
        ([300, 200, 320, 230], [340, 200, 360, 230], "近距离中心相似大小"),
        # 近距离，画面边缘，相似大小
        ([50, 50, 70, 80], [90, 50, 110, 80], "近距离边缘相似大小"),
        # 远距离，画面中心，相似大小
        ([200, 200, 220, 230], [400, 200, 420, 230], "远距离中心相似大小"),
        # 不同大小的人物（可能距离较远）
        ([300, 200, 340, 240], [360, 200, 380, 220], "不同大小人物"),
        # 画面边缘的小人物（可能很远）
        ([10, 10, 30, 40], [610, 10, 630, 40], "边缘小人物"),
    ]
    
    for bbox1, bbox2, description in test_cases:
        center1_x = (bbox1[0] + bbox1[2]) // 2
        center1_y = (bbox1[1] + bbox1[3]) // 2
        center2_x = (bbox2[0] + bbox2[2]) // 2
        center2_y = (bbox2[1] + bbox2[3]) // 2
        
        pixel_distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        real_distance = recognizer._estimate_real_distance(bbox1, bbox2, center1_x, center1_y, center2_x, center2_y)
        
        print(f"{description}:")
        print(f"  像素距离: {pixel_distance:.1f}")
        print(f"  估算实际距离: {real_distance:.1f}")
        print(f"  距离因子: {real_distance/pixel_distance:.2f}")
        
        # 判断是否可能误判
        if real_distance < 50 and pixel_distance > 100:
            print(f"  ⚠️  可能误判：实际距离较小但像素距离较大")
        elif real_distance > 200 and pixel_distance < 50:
            print(f"  ⚠️  可能误判：实际距离较大但像素距离较小")
        else:
            print(f"  ✅ 距离估算合理")
        print()

if __name__ == "__main__":
    print("打架检测灵敏度测试")
    print("=" * 50)
    
    # 测试距离估算
    test_distance_estimation_improved()
    
    # 测试打架检测灵敏度
    test_fighting_sensitivity()
    
    print("\n测试完成！") 