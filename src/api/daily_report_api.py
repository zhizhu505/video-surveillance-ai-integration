#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日报生成API接口
集成到Flask应用中，提供RESTful API接口
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app, Response

from daily_report_generator import DailyReportGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Blueprint
daily_report_bp = Blueprint('daily_report', __name__, url_prefix='/api/daily-report')

def load_config():
    """加载日报配置"""
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
            'enable': False,
            'model': 'Qwen/Qwen-VL-Chat',
            'device': 'cpu',
            'max_scene_descriptions': 5
        },
        'report_settings': {
            'output_dir': 'reports',
            'format': 'markdown',
            'include_images': False,
            'max_alerts_per_report': 100
        }
    }

@daily_report_bp.route('/generate', methods=['POST'])
def generate_daily_report():
    """生成日报API接口"""
    try:
        # 获取请求参数
        data = request.get_json() or {}
        target_date = data.get('date')
        output_format = data.get('format', 'markdown')
        include_multimodal = data.get('include_multimodal', False)
        
        # 加载配置
        config = load_config()
        
        # 更新配置
        config['multimodal_settings']['enable'] = include_multimodal
        config['report_settings']['format'] = output_format
        
        # 初始化生成器
        generator = DailyReportGenerator(config)
        
        # 生成日报
        report = generator.generate_daily_report(target_date)
        
        if not report:
            return jsonify({
                'success': False,
                'message': '日报生成失败'
            }), 500
        
        # 保存日报
        output_path = generator.save_report(report)
        
        return jsonify({
            'success': True,
            'message': '日报生成成功',
            'data': {
                'date': report['date'],
                'generated_at': report['generated_at'],
                'content': report['content'],
                'output_path': output_path,
                'analysis_results': report['analysis_results']
            }
        })
        
    except Exception as e:
        logger.error(f"生成日报失败: {e}")
        return jsonify({
            'success': False,
            'message': f'生成日报失败: {str(e)}'
        }), 500

@daily_report_bp.route('/generate', methods=['GET'])
def generate_daily_report_get():
    """GET方式生成日报（简化版）"""
    try:
        # 获取查询参数
        target_date = request.args.get('date')
        output_format = request.args.get('format', 'markdown')
        
        # 加载配置
        config = load_config()
        config['report_settings']['format'] = output_format
        
        # 初始化生成器
        generator = DailyReportGenerator(config)
        
        # 生成日报
        report = generator.generate_daily_report(target_date)
        
        if not report:
            return jsonify({
                'success': False,
                'message': '日报生成失败'
            }), 500
        
        # 保存日报
        output_path = generator.save_report(report)
        
        return jsonify({
            'success': True,
            'message': '日报生成成功',
            'data': {
                'date': report['date'],
                'generated_at': report['generated_at'],
                'content': report['content'],
                'output_path': output_path
            }
        })
        
    except Exception as e:
        logger.error(f"生成日报失败: {e}")
        return jsonify({
            'success': False,
            'message': f'生成日报失败: {str(e)}'
        }), 500

@daily_report_bp.route('/status', methods=['GET'])
def get_daily_report_status():
    """获取日报生成状态"""
    try:
        config = load_config()
        
        # 检查配置状态
        status = {
            'llm_configured': bool(os.getenv(config['llm_settings']['api_key_env'])),
            'multimodal_enabled': config['multimodal_settings']['enable'],
            'output_format': config['report_settings']['format'],
            'api_base_url': config['api_settings']['base_url']
        }
        
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取状态失败: {str(e)}'
        }), 500

@daily_report_bp.route('/config', methods=['GET'])
def get_daily_report_config():
    """获取日报配置"""
    try:
        config = load_config()
        
        # 隐藏敏感信息
        safe_config = config.copy()
        if 'llm_settings' in safe_config:
            safe_config['llm_settings'] = safe_config['llm_settings'].copy()
            safe_config['llm_settings']['api_key'] = '***' if os.getenv(safe_config['llm_settings']['api_key_env']) else '未设置'
        
        return jsonify({
            'success': True,
            'data': safe_config
        })
        
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取配置失败: {str(e)}'
        }), 500

@daily_report_bp.route('/test', methods=['POST'])
def test_daily_report():
    """测试日报生成功能"""
    try:
        # 获取请求参数
        data = request.get_json() or {}
        test_mode = data.get('test_mode', 'basic')
        
        config = load_config()
        generator = DailyReportGenerator(config)
        
        # 根据测试模式执行不同测试
        if test_mode == 'data_collection':
            test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            test_data = generator._collect_daily_data(test_date)
            return jsonify({
                'success': True,
                'message': '数据收集测试完成',
                'data': {
                    'total_alerts': test_data['total_alerts'],
                    'handled_count': test_data['handled_count'],
                    'unhandled_count': test_data['unhandled_count'],
                    'source_types': test_data['source_types']
                }
            })
        
        elif test_mode == 'cot_analysis':
            # 使用模拟数据测试CoT分析
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
                    }
                ],
                'total_alerts': 1,
                'handled_count': 0,
                'unhandled_count': 1,
                'source_types': ['fall_detection']
            }
            
            multimodal_content = {
                'scene_descriptions': [
                    {
                        'alert_id': 1,
                        'description': '监控画面显示一名人员突然倒地',
                        'level': 'high',
                        'time': '2024-01-15 10:30:00'
                    }
                ]
            }
            
            analysis_results = generator._chain_of_thought_analysis(mock_data, multimodal_content)
            
            return jsonify({
                'success': True,
                'message': 'CoT分析测试完成',
                'data': {
                    'steps': list(analysis_results.keys()),
                    'results': {k: v.get('summary', '')[:100] + '...' for k, v in analysis_results.items()}
                }
            })
        
        else:
            return jsonify({
                'success': False,
                'message': f'不支持的测试模式: {test_mode}'
            }), 400
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return jsonify({
            'success': False,
            'message': f'测试失败: {str(e)}'
        }), 500

@daily_report_bp.route('/list', methods=['GET'])
def list_daily_reports():
    """获取指定日期所有日报文件列表"""
    try:
        target_date = request.args.get('date')
        if not target_date:
            return jsonify({'success': False, 'message': '缺少日期参数'}), 400
        date_str = target_date.replace('/', '-')
        output_dir = 'reports'
        if not os.path.exists(output_dir):
            return jsonify({'success': True, 'data': []})
        files = []
        for fname in os.listdir(output_dir):
            if fname.startswith(f'daily_report_{date_str}_') and fname.endswith('.md'):
                fpath = os.path.join(output_dir, fname)
                files.append({
                    'filename': fname,
                    'created_at': os.path.getctime(fpath),
                    'filesize': os.path.getsize(fpath)
                })
        # 按创建时间倒序
        files.sort(key=lambda x: x['created_at'], reverse=True)
        return jsonify({'success': True, 'data': files})
    except Exception as e:
        logger.error(f"获取日报列表失败: {e}")
        return jsonify({'success': False, 'message': f'获取日报列表失败: {str(e)}'}), 500

@daily_report_bp.route('/raw_content', methods=['GET'])
def get_daily_report_raw_content():
    """直接返回日报Markdown原文（支持filename或date）"""
    try:
        filename = request.args.get('filename')
        output_dir = 'reports'
        if filename:
            file_path = os.path.join(output_dir, filename)
            if not os.path.exists(file_path):
                return jsonify({'success': False, 'message': f'未找到日报文件: {filename}'}), 404
        else:
            # 兼容老逻辑：date参数
            target_date = request.args.get('date')
            if not target_date:
                return jsonify({'success': False, 'message': '缺少日期参数'}), 400
            date_str = target_date.replace('/', '-')
            # 查找最新的日报
            candidates = [f for f in os.listdir(output_dir) if f.startswith(f'daily_report_{date_str}_') and f.endswith('.md')]
            if not candidates:
                return jsonify({'success': False, 'message': f'未找到日报文件: {date_str}'}), 404
            # 按创建时间倒序，取最新
            candidates.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
            file_path = os.path.join(output_dir, candidates[0])
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content, mimetype='text/markdown; charset=utf-8')
    except Exception as e:
        logger.error(f"读取日报原文失败: {e}")
        return jsonify({'success': False, 'message': f'读取日报原文失败: {str(e)}'}), 500

def init_daily_report_api(app):
    """初始化日报API"""
    app.register_blueprint(daily_report_bp)
    logger.info("日报生成API已注册") 