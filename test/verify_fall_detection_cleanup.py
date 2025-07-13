#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证摔倒检测清理的完整性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def verify_fall_detection_cleanup():
    """验证摔倒检测清理的完整性"""
    print("=== 验证摔倒检测清理完整性 ===")
    
    # 读取源代码文件
    with open('../src/danger_recognizer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键指标
    checks = {
        "fall_alerts = []": content.count("fall_alerts = []"),
        "fall_alerts.append": content.count("fall_alerts.append"),
        "confidence >= 0.8 and cooldown_ok": content.count("confidence >= 0.8 and cooldown_ok"),
        "摔倒事件检测触发": content.count("摔倒事件检测触发"),
        "Fall Detection": content.count("Fall Detection"),
        "self.DANGER_TYPES['fall']": content.count("self.DANGER_TYPES['fall']"),
    }
    
    print("检查结果:")
    for check, count in checks.items():
        print(f"  {check}: {count} 次")
    
    # 验证清理效果
    print("\n验证结果:")
    
    # 应该只有一个 fall_alerts = []
    if checks["fall_alerts = []"] == 1:
        print("✅ fall_alerts 初始化: 正确（只有1次）")
    else:
        print(f"❌ fall_alerts 初始化: 错误（有{checks['fall_alerts = []']}次）")
    
    # 应该只有一个 fall_alerts.append
    if checks["fall_alerts.append"] == 1:
        print("✅ fall_alerts.append: 正确（只有1次）")
    else:
        print(f"❌ fall_alerts.append: 错误（有{checks['fall_alerts.append']}次）")
    
    # 应该只有一个置信度检查
    if checks["confidence >= 0.8 and cooldown_ok"] == 1:
        print("✅ 置信度检查: 正确（只有1次）")
    else:
        print(f"❌ 置信度检查: 错误（有{checks['confidence >= 0.8 and cooldown_ok']}次）")
    
    # 应该只有一个调试输出
    if checks["摔倒事件检测触发"] == 1:
        print("✅ 调试输出: 正确（只有1次）")
    else:
        print(f"❌ 调试输出: 错误（有{checks['摔倒事件检测触发']}次）")
    
    # 检查是否有重复的摔倒检测逻辑
    lines = content.split('\n')
    fall_detection_blocks = 0
    in_fall_block = False
    
    for line in lines:
        if '摔倒检测' in line and '#' in line:
            fall_detection_blocks += 1
        elif 'if len(self.history) >= 10:' in line:
            # 检查接下来的几行是否包含摔倒检测逻辑
            in_fall_block = True
        elif in_fall_block and ('recent_vertical_motions' in line or 'max_vertical_motion' in line):
            fall_detection_blocks += 1
            in_fall_block = False
    
    print(f"✅ 摔倒检测逻辑块: {fall_detection_blocks} 个")
    
    print("\n=== 清理验证完成 ===")
    
    # 总结
    if (checks["fall_alerts = []"] == 1 and 
        checks["fall_alerts.append"] == 1 and 
        checks["confidence >= 0.8 and cooldown_ok"] == 1 and 
        checks["摔倒事件检测触发"] == 1):
        print("🎉 摔倒检测清理成功！所有重复定义已删除。")
        return True
    else:
        print("⚠️ 摔倒检测清理可能不完整，请检查代码。")
        return False

if __name__ == "__main__":
    verify_fall_detection_cleanup() 