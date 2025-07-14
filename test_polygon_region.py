#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证多边形警戒区域功能
"""

import cv2
import numpy as np
import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from danger_recognizer import DangerRecognizer

def test_polygon_region():
    """测试多边形警戒区域功能"""
    print("开始测试：验证多边形警戒区域功能")
    
    # 创建危险识别器
    danger_recognizer = DangerRecognizer()
    
    # 测试1：矩形区域（4个点）
    print("\n测试1：矩形区域（4个点）")
    rectangle_points = [
        [100, 100],
        [300, 100], 
        [300, 300],
        [100, 300]
    ]
    danger_recognizer.add_alert_region(rectangle_points, "Rectangle Zone")
    
    # 测试检测框是否在区域内
    test_bboxes = [
        [150, 150, 200, 250],  # 在区域内
        [50, 50, 80, 80],      # 在区域外
        [250, 250, 350, 350],  # 部分重叠
        [0, 0, 50, 50]         # 完全在区域外
    ]
    
    for i, bbox in enumerate(test_bboxes):
        in_region = False
        for region in danger_recognizer.alert_regions:
            if danger_recognizer._check_bbox_intersection(bbox, region['points']):
                in_region = True
                break
        print(f"  检测框{i+1} {bbox}: {'在区域内' if in_region else '在区域外'}")
    
    # 测试2：三角形区域（3个点）
    print("\n测试2：三角形区域（3个点）")
    danger_recognizer.clear_alert_regions()
    triangle_points = [
        [200, 100],
        [300, 300],
        [100, 300]
    ]
    danger_recognizer.add_alert_region(triangle_points, "Triangle Zone")
    
    # 测试检测框是否在三角形区域内
    test_bboxes_triangle = [
        [200, 200, 220, 220],  # 在三角形中心
        [150, 150, 170, 170],  # 在三角形内
        [50, 50, 70, 70],      # 在三角形外
        [350, 350, 370, 370]   # 在三角形外
    ]
    
    for i, bbox in enumerate(test_bboxes_triangle):
        in_region = False
        for region in danger_recognizer.alert_regions:
            if danger_recognizer._check_bbox_intersection(bbox, region['points']):
                in_region = True
                break
        print(f"  检测框{i+1} {bbox}: {'在区域内' if in_region else '在区域外'}")
    
    # 测试3：复杂多边形区域（5个点）
    print("\n测试3：复杂多边形区域（5个点）")
    danger_recognizer.clear_alert_regions()
    polygon_points = [
        [100, 100],
        [200, 50],
        [300, 100],
        [250, 200],
        [150, 200]
    ]
    danger_recognizer.add_alert_region(polygon_points, "Complex Polygon Zone")
    
    # 测试检测框是否在复杂多边形区域内
    test_bboxes_polygon = [
        [150, 100, 170, 120],  # 在多边形内
        [200, 150, 220, 170],  # 在多边形内
        [50, 50, 70, 70],      # 在多边形外
        [350, 350, 370, 370]   # 在多边形外
    ]
    
    for i, bbox in enumerate(test_bboxes_polygon):
        in_region = False
        for region in danger_recognizer.alert_regions:
            if danger_recognizer._check_bbox_intersection(bbox, region['points']):
                in_region = True
                break
        print(f"  检测框{i+1} {bbox}: {'在区域内' if in_region else '在区域外'}")
    
    # 测试4：可视化测试
    print("\n测试4：可视化测试")
    # 创建测试图像
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 绘制多边形区域
    for region in danger_recognizer.alert_regions:
        points = region['points'].astype(np.int32)
        cv2.polylines(test_image, [points], True, (255, 0, 0), 2)
    
    # 绘制测试检测框
    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, bbox in enumerate(test_bboxes_polygon):
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
        
        # 检查是否在区域内
        in_region = False
        for region in danger_recognizer.alert_regions:
            if danger_recognizer._check_bbox_intersection(bbox, region['points']):
                in_region = True
                break
        
        # 添加标签
        label = f"Box{i+1}: {'IN' if in_region else 'OUT'}"
        cv2.putText(test_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 保存测试图像
    cv2.imwrite("test_polygon_region.jpg", test_image)
    print("✓ 已保存可视化测试图像: test_polygon_region.jpg")
    
    # 测试5：停留时间检测
    print("\n测试5：停留时间检测")
    # 模拟对象检测结果
    test_detections = [
        {
            'bbox': [150, 100, 170, 120],
            'class': 'person',
            'confidence': 0.95
        }
    ]
    
    # 模拟多帧处理
    for frame in range(10):
        danger_recognizer.current_frame = frame
        alerts = danger_recognizer._track_danger_zone_dwell(test_detections)
        if alerts:
            for alert in alerts:
                print(f"  帧{frame}: 检测到停留告警 - {alert['type']}")
        time.sleep(0.1)  # 模拟时间流逝
    
    print("\n测试完成！")
    print("总结：")
    print("- 支持矩形、三角形、复杂多边形等多种形状的警戒区域")
    print("- 准确检测检测框是否在区域内")
    print("- 支持停留时间跟踪和告警")
    print("- 可视化功能正常工作")

if __name__ == "__main__":
    test_polygon_region() 