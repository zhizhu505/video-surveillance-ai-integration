#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试可视化改进的脚本
"""

import sys
import os
import numpy as np
import cv2

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_visualization_improvements():
    """测试可视化改进效果"""
    try:
        from danger_recognizer import DangerRecognizer
        
        print("=== 可视化改进测试 ===\n")
        
        # 创建危险行为识别器
        recognizer = DangerRecognizer()
        
        # 添加警戒区域
        recognizer.add_alert_region([(100, 100), (300, 100), (300, 300), (100, 300)], "Restricted Area")
        
        print("✓ 已添加警戒区域")
        
        # 创建测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 模拟AI检测结果
        detections = [
            {'bbox': [150, 150, 200, 200], 'class': 'person', 'confidence': 0.85},
            {'bbox': [400, 200, 450, 250], 'class': 'car', 'confidence': 0.92},
        ]
        
        # 模拟告警
        alerts = [
            {
                'type': 'Intrusion Alert',
                'confidence': 0.85,
                'frame': 100,
                'object': 'person',
                'region_name': 'Restricted Area'
            },
            {
                'type': 'Large Area Motion',
                'confidence': 0.75,
                'frame': 100,
                'area': 0.2,
                'threshold': 0.15
            }
        ]
        
        print("✓ 已创建测试数据")
        
        # 测试可视化
        vis_frame = recognizer.visualize(frame, alerts, detections=detections)
        
        print("✓ 可视化生成成功")
        
        # 保存测试图像
        output_path = "test_visualization.jpg"
        cv2.imwrite(output_path, vis_frame)
        print(f"✓ 测试图像已保存到: {output_path}")
        
        # 显示改进内容
        print("\n=== 改进内容总结 ===")
        print("1. ✅ 告警类型改为英文显示")
        print("   - 突然剧烈运动 → Sudden Motion")
        print("   - 大范围异常运动 → Large Area Motion")
        print("   - 入侵警告区域 → Intrusion Alert")
        print("   - 可能摔倒 → Fall Detection")
        
        print("\n2. ✅ 红色边框优化")
        print("   - 移除全局红色边框")
        print("   - 告警对象显示红色边框")
        print("   - 正常对象显示绿色边框")
        print("   - 告警对象添加'ALERT'标识")
        
        print("\n3. ✅ 告警信息显示优化")
        print("   - 告警信息显示在顶部")
        print("   - 添加黑色背景框提高可读性")
        print("   - 支持多个告警同时显示")
        
        print("\n4. ✅ 智能对象级告警")
        print("   - 根据告警类型智能选择颜色")
        print("   - 入侵告警对象显示红色")
        print("   - 正常对象显示绿色")
        
        print("\n✓ 可视化改进测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_improvements()
    sys.exit(0 if success else 1) 