#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试告警处理功能
"""

import requests
import json
import time

def test_alert_handling():
    """测试告警处理功能"""
    base_url = "http://localhost:5000"
    
    print("测试告警处理功能...")
    
    try:
        # 1. 获取告警统计
        print("\n1. 获取告警统计:")
        response = requests.get(f"{base_url}/alerts/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"  总告警数: {stats['total_alerts']}")
            print(f"  已处理: {stats['handled_alerts']}")
            print(f"  未处理: {stats['unhandled_alerts']}")
        else:
            print(f"  获取统计失败: {response.status_code}")
        
        # 2. 获取告警列表
        print("\n2. 获取告警列表:")
        response = requests.get(f"{base_url}/alerts")
        if response.status_code == 200:
            alerts = response.json()
            print(f"  告警数量: {len(alerts)}")
            for i, alert in enumerate(alerts):
                status = "已处理" if alert.get('handled', False) else "未处理"
                print(f"  {i+1}. {alert.get('type', 'N/A')} - {status} (ID: {alert.get('id', 'N/A')})")
                
                # 3. 测试处理告警
                if not alert.get('handled', False):
                    print(f"    测试处理告警 {alert.get('id')}...")
                    response = requests.post(f"{base_url}/alerts/handle", 
                                           json={"alert_id": alert.get('id')})
                    if response.status_code == 200:
                        result = response.json()
                        print(f"    处理结果: {result['message']}")
                    else:
                        print(f"    处理失败: {response.status_code}")
                    break
        else:
            print(f"  获取告警列表失败: {response.status_code}")
        
        # 4. 再次获取统计，检查是否更新
        print("\n3. 再次获取告警统计:")
        time.sleep(1)  # 等待一下
        response = requests.get(f"{base_url}/alerts/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"  总告警数: {stats['total_alerts']}")
            print(f"  已处理: {stats['handled_alerts']}")
            print(f"  未处理: {stats['unhandled_alerts']}")
        
        print("\n测试完成！")
        
    except requests.exceptions.ConnectionError:
        print("连接失败，请确保系统正在运行并且Web界面已启用")
    except Exception as e:
        print(f"测试出错: {str(e)}")

if __name__ == "__main__":
    test_alert_handling() 