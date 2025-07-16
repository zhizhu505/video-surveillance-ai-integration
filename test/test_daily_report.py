#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日报生成功能测试脚本
测试CoT思维链 + Prompt Engineering的日报自动生成
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.daily_report_generator import DailyReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    config_path = "src/config/daily_report_config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return get_default_config()

def get_default_config():
    """获取默认配置"""
    return {
        'api_settings': {
            'base_url': 'http://localhost:5000',
            'timeout': 30
        },
        'llm_settings': {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            'temperature': 0.7,
            'api_key_env': 'OPENAI_API_KEY'
        },
        'multimodal_settings': {
            'enable': False,  # 测试时默认关闭多模态
            'model': 'Qwen/Qwen-VL-Chat',
            'device': 'cpu',
            'max_scene_descriptions': 5
        },
        'report_settings': {
            'output_dir': 'test/reports',
            'format': 'markdown',
            'include_images': False,
            'max_alerts_per_report': 100
        }
    }

def test_data_collection():
    """测试数据收集功能"""
    logger.info("=== 测试数据收集功能 ===")
    
    config = load_config()
    generator = DailyReportGenerator(config)
    
    # 测试数据收集
    test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    data = generator._collect_daily_data(test_date)
    
    logger.info(f"收集到 {data['total_alerts']} 条告警数据")
    logger.info(f"已处理: {data['handled_count']}, 未处理: {data['unhandled_count']}")
    logger.info(f"告警类型: {data['source_types']}")
    
    return data

def test_cot_analysis():
    """测试CoT思维链分析"""
    logger.info("=== 测试CoT思维链分析 ===")
    
    config = load_config()
    generator = DailyReportGenerator(config)
    
    # 模拟数据
    mock_data = {
        'date': '2024-01-15',
        'alerts': [
            {
                'id': 1,
                'source_type': 'fall_detection',
                'danger_level': 'high',
                'datetime': '2024-01-15 10:30:00',
                'message': '检测到人员摔倒',
                'acknowledged': False
            },
            {
                'id': 2,
                'source_type': 'danger_zone_dwell',
                'danger_level': 'medium',
                'datetime': '2024-01-15 14:20:00',
                'message': '人员在危险区域停留',
                'acknowledged': True
            }
        ],
        'statistics': {'total_alerts': 2, 'handled_alerts': 1},
        'source_types': ['fall_detection', 'danger_zone_dwell'],
        'total_alerts': 2,
        'handled_count': 1,
        'unhandled_count': 1
    }
    
    multimodal_content = {
        'scene_descriptions': [
            {
                'alert_id': 1,
                'description': '监控画面显示一名人员突然倒地，疑似摔倒',
                'level': 'high',
                'time': '2024-01-15 10:30:00'
            }
        ]
    }
    
    # 测试CoT分析
    analysis_results = generator._chain_of_thought_analysis(mock_data, multimodal_content)
    
    logger.info("CoT分析结果:")
    for step_name, result in analysis_results.items():
        logger.info(f"  {step_name}: {result.get('summary', '无结果')[:100]}...")
    
    return analysis_results

def test_prompt_engineering():
    """测试Prompt Engineering"""
    logger.info("=== 测试Prompt Engineering ===")
    
    config = load_config()
    generator = DailyReportGenerator(config)
    
    # 测试不同类型的Prompt
    test_prompts = [
        {
            'name': '统计分析',
            'data': {
                'total_alerts': 5,
                'handled_count': 3,
                'unhandled_count': 2,
                'source_types': ['fall_detection', 'fighting_detection']
            }
        },
        {
            'name': '风险分析',
            'data': {
                'high_risk_count': 2,
                'unhandled_high_risk': 1,
                'total_alerts': 5,
                'handling_rate': 60.0
            }
        }
    ]
    
    for test_case in test_prompts:
        logger.info(f"测试 {test_case['name']} Prompt:")
        # 这里可以测试具体的Prompt效果
        logger.info(f"  数据: {test_case['data']}")

def test_full_report_generation():
    """测试完整日报生成"""
    logger.info("=== 测试完整日报生成 ===")
    
    config = load_config()
    generator = DailyReportGenerator(config)
    
    # 生成昨天的日报
    try:
        report = generator.generate_daily_report()
        
        logger.info(f"日报生成成功:")
        logger.info(f"  日期: {report['date']}")
        logger.info(f"  生成时间: {report['generated_at']}")
        logger.info(f"  内容长度: {len(report['content'])} 字符")
        
        # 保存日报
        output_path = generator.save_report(report)
        if output_path:
            logger.info(f"  保存路径: {output_path}")
        
        # 显示内容预览
        logger.info("内容预览:")
        print("-" * 50)
        print(report['content'][:500] + "..." if len(report['content']) > 500 else report['content'])
        print("-" * 50)
        
        return report
        
    except Exception as e:
        logger.error(f"日报生成失败: {e}")
        return None

def test_multimodal_integration():
    """测试多模态集成"""
    logger.info("=== 测试多模态集成 ===")
    
    config = load_config()
    config['multimodal_settings']['enable'] = True
    generator = DailyReportGenerator(config)
    
    if generator.qwen_vl:
        logger.info("多模态模型已启用")
        # 这里可以添加具体的多模态测试
    else:
        logger.info("多模态模型未启用或初始化失败")

def test_configuration_loading():
    """测试配置加载"""
    logger.info("=== 测试配置加载 ===")
    
    config = load_config()
    
    logger.info("配置信息:")
    logger.info(f"  API基础URL: {config['api_settings']['base_url']}")
    logger.info(f"  LLM提供商: {config['llm_settings']['provider']}")
    logger.info(f"  LLM模型: {config['llm_settings']['model']}")
    logger.info(f"  多模态启用: {config['multimodal_settings']['enable']}")
    logger.info(f"  输出目录: {config['report_settings']['output_dir']}")

def main():
    """主测试函数"""
    logger.info("开始日报生成功能测试")
    
    try:
        # 1. 测试配置加载
        test_configuration_loading()
        
        # 2. 测试数据收集
        test_data_collection()
        
        # 3. 测试CoT分析
        test_cot_analysis()
        
        # 4. 测试Prompt Engineering
        test_prompt_engineering()
        
        # 5. 测试多模态集成
        test_multimodal_integration()
        
        # 6. 测试完整日报生成
        report = test_full_report_generation()
        
        if report:
            logger.info("✅ 所有测试完成，日报生成功能正常")
        else:
            logger.warning("⚠️ 日报生成测试失败，请检查配置和网络连接")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 