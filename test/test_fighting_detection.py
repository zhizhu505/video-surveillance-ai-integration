#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
打架检测测试脚本 - 测试改进后的打架检测功能
"""

import sys
import os
import cv2
import numpy as np
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from danger_recognizer import DangerRecognizer

def create_test_video_with_fighting():
    """创建包含打架行为的测试视频"""
    width, height = 640, 480
    fps = 30
    duration = 10  # 10秒视频
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_fighting.avi', fourcc, fps, (width, height))
    
    # 人物初始位置
    person1_pos = [100, 200]
    person2_pos = [150, 200]
    person1_vel = [2, 0]
    person2_vel = [-2, 0]
    
    for frame_idx in range(fps * duration):
        # 创建背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # 模拟打架行为（第3-7秒）
        if 3 * fps <= frame_idx <= 7 * fps:
            # 人物靠近并重叠
            person1_pos[0] = 200 + int(10 * np.sin(frame_idx * 0.5))
            person2_pos[0] = 250 + int(10 * np.sin(frame_idx * 0.5 + np.pi))
            person1_pos[1] = 200 + int(5 * np.cos(frame_idx * 0.3))
            person2_pos[1] = 200 + int(5 * np.cos(frame_idx * 0.3 + np.pi))
        else:
            # 正常行走
            person1_pos[0] += person1_vel[0]
            person2_pos[0] += person2_vel[0]
            
            # 边界检查
            if person1_pos[0] < 50 or person1_pos[0] > width - 50:
                person1_vel[0] *= -1
            if person2_pos[0] < 50 or person2_pos[0] > width - 50:
                person2_vel[0] *= -1
        
        # 绘制人物（模拟检测框）
        person_size = 40
        cv2.rectangle(frame, 
                     (person1_pos[0] - person_size//2, person1_pos[1] - person_size//2),
                     (person1_pos[0] + person_size//2, person1_pos[1] + person_size//2),
                     (0, 255, 0), 2)
        cv2.rectangle(frame, 
                     (person2_pos[0] - person_size//2, person2_pos[1] - person_size//2),
                     (person2_pos[0] + person_size//2, person2_pos[1] + person_size//2),
                     (0, 255, 0), 2)
        
        # 添加ID标签
        cv2.putText(frame, "ID:1", (person1_pos[0] - 20, person1_pos[1] - person_size//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ID:2", (person2_pos[0] - 20, person2_pos[1] - person_size//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加时间戳
        cv2.putText(frame, f"Time: {frame_idx/fps:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("测试视频已创建: test_fighting.avi")

def simulate_object_detections(frame_idx, person1_pos, person2_pos):
    """模拟对象检测结果"""
    person_size = 40
    
    detections = [
        {
            'class': 'person',
            'bbox': [
                person1_pos[0] - person_size//2,
                person1_pos[1] - person_size//2,
                person1_pos[0] + person_size//2,
                person1_pos[1] + person_size//2
            ],
            'confidence': 0.9,
            'person_id': 1
        },
        {
            'class': 'person',
            'bbox': [
                person2_pos[0] - person_size//2,
                person2_pos[1] - person_size//2,
                person2_pos[0] + person_size//2,
                person2_pos[1] + person_size//2
            ],
            'confidence': 0.9,
            'person_id': 2
        }
    ]
    
    return detections

def simulate_motion_features(frame_idx):
    """模拟运动特征"""
    # 模拟打架期间的运动强度
    if 3 * 30 <= frame_idx <= 7 * 30:  # 第3-7秒
        base_magnitude = 8 + 3 * np.sin(frame_idx * 0.5)
    else:
        base_magnitude = 2 + np.random.normal(0, 1)
    
    # 确保base_magnitude为正数
    base_magnitude = max(0.1, base_magnitude)
    
    features = {
        'flow_mean_magnitude': base_magnitude,
        'flow_max_magnitude': base_magnitude * 1.5,
        'motion_vectors': []
    }
    
    # 生成运动向量
    for i in range(20):
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 480)
        dx = np.random.normal(0, max(0.1, base_magnitude * 0.5))
        dy = np.random.normal(0, max(0.1, base_magnitude * 0.5))
        features['motion_vectors'].append([x, y, dx, dy])
    
    return features

def test_fighting_detection():
    """测试打架检测功能"""
    print("开始测试打架检测功能...")
    
    # 创建危险行为识别器 - 调整参数使其更容易触发
    config = {
        'fighting_distance_threshold': 200,  # 增加距离阈值
        'fighting_motion_threshold': 4,      # 降低运动强度阈值
        'fighting_duration_frames': 10,      # 减少持续时间阈值
        'fighting_confidence_threshold': 0.6, # 降低置信度阈值
        'save_alerts': False  # 不保存告警图片
    }
    
    recognizer = DangerRecognizer(config)
    
    # 模拟视频帧
    width, height = 640, 480
    fps = 30
    duration = 10  # 10秒
    
    fighting_detected = False
    detection_times = []
    
    for frame_idx in range(fps * duration):
        # 创建测试帧
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # 模拟人物位置
        if 3 * fps <= frame_idx <= 7 * fps:
            # 打架期间 - 让人物更靠近
            person1_pos = [200 + int(15 * np.sin(frame_idx * 0.5)), 
                          200 + int(8 * np.cos(frame_idx * 0.3))]
            person2_pos = [220 + int(15 * np.sin(frame_idx * 0.5 + np.pi)), 
                          200 + int(8 * np.cos(frame_idx * 0.3 + np.pi))]
        else:
            # 正常期间
            person1_pos = [100 + frame_idx % 200, 200]
            person2_pos = [400 - frame_idx % 200, 200]
        
        # 模拟检测结果
        detections = simulate_object_detections(frame_idx, person1_pos, person2_pos)
        
        # 模拟运动特征
        features = simulate_motion_features(frame_idx)
        
        # 添加调试信息
        if 3 * fps <= frame_idx <= 7 * fps:
            pixel_distance = np.sqrt((person1_pos[0] - person2_pos[0])**2 + (person1_pos[1] - person2_pos[1])**2)
            print(f"帧 {frame_idx}: 人物距离={pixel_distance:.1f}px, 运动强度={features['flow_mean_magnitude']:.1f}")
        
        # 处理帧
        alerts = recognizer.process_frame(frame, features, detections)
        
        # 检查打架检测
        for alert in alerts:
            if alert['type'] == 'Fighting Detection':
                if not fighting_detected:
                    fighting_detected = True
                    detection_times.append(frame_idx / fps)
                    print(f"检测到打架行为！时间: {frame_idx/fps:.1f}秒")
                    print(f"  置信度: {alert['confidence']:.2f}")
                    print(f"  条件: {alert['condition_details']}")
                    print(f"  持续时间: {alert.get('duration', 0)}帧")
                    print(f"  像素距离: {alert.get('pixel_distance', 0):.1f}")
                    print(f"  估算实际距离: {alert.get('real_distance', 0):.1f}")
                    print(f"  运动强度: {alert.get('motion_intensity', 0):.1f}")
                    print()
        
        # 显示进度
        if frame_idx % (fps * 2) == 0:  # 每2秒显示一次进度
            print(f"处理进度: {frame_idx/fps:.1f}/{duration}秒")
    
    # 输出测试结果
    print("\n=== 测试结果 ===")
    if fighting_detected:
        print(f"✅ 成功检测到打架行为")
        print(f"检测时间点: {detection_times}")
        print(f"检测次数: {len(detection_times)}")
    else:
        print("❌ 未检测到打架行为")
        print("可能的原因:")
        print("1. 人物距离不够近")
        print("2. 运动强度不够高")
        print("3. 持续时间不够长")
        print("4. 置信度阈值过高")
    
    # 输出行为统计
    behavior_stats = recognizer.get_behavior_stats()
    print(f"\n行为统计:")
    for behavior, count in behavior_stats.items():
        print(f"  {behavior}: {count}")
    
    return fighting_detected

def test_distance_estimation():
    """测试距离估算功能"""
    print("\n=== 测试距离估算功能 ===")
    
    recognizer = DangerRecognizer()
    
    # 测试不同距离和位置的情况
    test_cases = [
        # 近距离，画面中心
        ([300, 200, 320, 220], [340, 200, 360, 220], "近距离中心"),
        # 近距离，画面边缘
        ([50, 50, 70, 70], [90, 50, 110, 70], "近距离边缘"),
        # 远距离，画面中心
        ([200, 200, 220, 220], [400, 200, 420, 220], "远距离中心"),
        # 不同大小的人物
        ([300, 200, 340, 240], [360, 200, 380, 220], "不同大小"),
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
        print()

if __name__ == "__main__":
    print("打架检测功能测试")
    print("=" * 50)
    
    # 创建测试视频
    create_test_video_with_fighting()
    
    # 测试距离估算
    test_distance_estimation()
    
    # 测试打架检测
    success = test_fighting_detection()
    
    if success:
        print("\n🎉 测试通过！打架检测功能正常工作。")
    else:
        print("\n⚠️  测试未通过，可能需要调整参数。")
    
    print("\n测试完成！") 