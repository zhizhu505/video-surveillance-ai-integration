#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全功能视频监控系统 - 整合所有已开发模块的综合系统
结合了运动特征提取、危险行为识别、Web界面等功能
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import threading
import json
from queue import Queue
from datetime import datetime, date
import traceback
import collections
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("all_in_one_system.log")
    ]
)
logger = logging.getLogger("AllInOneSystem")

# 导入运动特征提取模块
try:
    from models.motion.motion_manager import MotionFeatureManager

    logger.info("成功导入MotionFeatureManager")
except ImportError as e:
    logger.error(f"导入MotionFeatureManager失败: {str(e)}")
    sys.exit(1)

# 导入危险行为识别模块
try:
    from danger_recognizer import DangerRecognizer

    logger.info("成功导入DangerRecognizer")
except ImportError as e:
    logger.error(f"导入DangerRecognizer失败: {str(e)}")
    sys.exit(1)

# 导入告警数据库模块
try:
    from models.alert.alert_database import AlertDatabase
    from models.alert.mysql_database import MySQLAlertDatabase
    logger.info("成功导入AlertDatabase和MySQLAlertDatabase")
except ImportError as e:
    logger.error(f"导入AlertDatabase或MySQLAlertDatabase失败: {str(e)}")
    AlertDatabase = None
    MySQLAlertDatabase = None

# （可选）尝试导入Web界面模块,导入 Flask 相关组件（用于网页端流媒体服务、控制接口）
try:
    from flask import Flask, render_template, Response, jsonify, request
    import threading

    HAS_FLASK = True
    logger.info("成功导入Flask Web模块")
except ImportError:
    HAS_FLASK = False
    logger.warning("未找到Flask，Web界面功能将被禁用")

# （可选）尝试导入AI模块
try:
    import torch
    from ultralytics import YOLO

    HAS_AI = True
    logger.info("成功导入AI功能模块")
except ImportError:
    HAS_AI = False
    logger.warning("未找到必要的AI依赖，AI功能将被禁用")

try:
    import audio_monitor
    HAS_AUDIO_MONITOR = True
    logger.info("成功导入audio_monitor音频监控模块")
except ImportError as e:
    HAS_AUDIO_MONITOR = False
    logger.warning(f"未找到audio_monitor，声学检测功能将被禁用: {e}")


class AllInOneSystem:
    """全功能视频监控系统 - 整合所有模块"""

    def __init__(self, args):
        """初始化系统"""
        self.args = args

        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)

        # 初始化状态
        self.running = False  # 是否正在运行
        self.paused = False  # 是否暂停
        self.current_frame = None  # 原始摄像头当前帧
        self.processed_frame = None  # 处理（叠加可视化）后帧
        self.last_frame_time = 0  # 上一帧的时间戳
        self.fps = 0  # 平均帧率
        self.frame_count = 0  # 已读取的总帧数
        self.processed_count = 0  # 实际参与运动/AI 分析处理的帧数
        self.alerts = []  # 当前触发的告警列表
        self.alert_count = 0  # 累积告警次数
        self.start_time = time.time()  # 启动时间戳，用于计算运行时长
        self.recognized_behaviors = []  # 存储识别到的行为信息
        self.recognized_interactions = []  # 存储识别到的交互信息
        self.recent_alerts = collections.deque(maxlen=1000)  # 记录最近1000条告警
        # 新增：全历史告警列表
        self.all_alerts = []  # 保存所有历史告警
        
        # 新增：告警处理相关
        self.alert_handling_stats = {
            'total_alerts': 0,
            'handled_alerts': 0,
            'unhandled_alerts': 0
        }  # 告警处理统计
        self.alert_lock = threading.Lock()  # 告警数据锁
        
        # 初始化告警数据库（强制只用MySQL测试）
        from models.alert.mysql_database import MySQLAlertDatabase
        try:
            self.alert_database = MySQLAlertDatabase(
                host='localhost',
                port=3306,
                user='root',
                password='cangshu606',
                database='video_surveillance_alerts',
                charset='utf8mb4'
            )
            logger.info("MySQL告警数据库初始化成功")
        except Exception as e:
            logger.error(f"MySQL告警数据库初始化失败: {str(e)}")
            self.alert_database = None

        # 线程和队列设置
        self.frame_queue = Queue(maxsize=30)  # 存储「捕获线程 → 处理线程」的帧（缓冲队列）
        self.result_queue = Queue(maxsize=30)  # 暂未使用，可用于「处理线程 → 其他线程」
        self.threads = []  # 存储线程对象，后面方便一起管理 & join

        # 初始化运动特征管理器
        self.motion_manager = MotionFeatureManager(
            use_optical_flow=True,
            use_motion_history=args.use_motion_history,
            optical_flow_method='farneback',
            use_gpu=args.use_gpu
        )

        # 初始化危险行为识别器
        danger_config = {
            'feature_count_threshold': args.feature_threshold,
            'motion_area_threshold': args.area_threshold,
            'alert_cooldown': args.alert_cooldown,
            'save_alerts': args.save_alerts,
            'alert_dir': os.path.join(args.output, 'alerts'),
            'min_confidence': args.min_confidence,
            # 新增：危险区域停留检测配置
            'distance_threshold_m': getattr(args, 'distance_threshold', 50),
            'dwell_time_threshold_s': getattr(args, 'dwell_time_threshold', 1.0),
            'fps': args.max_fps
        }
        # 实例化危险检测器
        self.danger_recognizer = DangerRecognizer(danger_config)

        # 如果指定了警戒区域，添加它，也是从命令行传参
        if args.alert_region:
            try:
                regions = eval(args.alert_region)
                if isinstance(regions, list) and len(regions) >= 3:
                    self.danger_recognizer.add_alert_region(regions, "Alert Zone")
            except Exception as e:
                logger.error(f"解析警戒区域失败: {str(e)}")

        # 初始化AI模块（如果可用）
        self.ai_model = None
        if args.enable_ai and HAS_AI:
            try:
                self.ai_model = YOLO(args.vision_model + ".pt")
                logger.info(f"成功加载AI模型: {args.vision_model}")
            except Exception as e:
                logger.error(f"加载AI模型失败: {str(e)}")
                self.ai_model = None

        # 初始化Web服务器（如果可用且启用）
        self.app = None
        if args.web_interface and HAS_FLASK:
            self.init_web_server()

        # 初始化视频录制器，如果用户传了 --record 参数，就把最终处理后的画面同时录制到视频文件里保存
        self.video_writer = None
        if args.record:
            try:
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                output_path = os.path.join(args.output, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
                self.video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (args.width, args.height))
                logger.info(f"视频将录制到: {output_path}")
            except Exception as e:
                logger.error(f"初始化视频录制器失败: {str(e)}")
                self.video_writer = None

        # 自动启动音频监控线程（如可用）
        self.audio_thread = None
        if getattr(args, 'enable_audio_monitor', False) and HAS_AUDIO_MONITOR:
            self.audio_thread = threading.Thread(target=audio_monitor.audio_monitor_callback, args=(self.add_audio_alert,))
            self.audio_thread.daemon = True
            self.audio_thread.start()
            logger.info("音频监控线程已启动")
        else:
            logger.info("未启用音频监控")

        self.recent_audio_events = collections.deque(maxlen=20)  # 缓存最近音频事件（label, score, timestamp）

        logger.info("全功能视频监控系统初始化完成")

    def init_web_server(self):
        """初始化Web服务器"""
        self.app = Flask(__name__, template_folder='../templates')  # 指定 HTML 模板文件夹位置

        # 添加静态文件服务
        @self.app.route('/alerts_images/<path:relpath>')
        def serve_alert_image(relpath):
            """提供告警图片访问服务"""
            import os
            from flask import send_from_directory, abort
            
            # 告警图片目录
            abs_path = os.path.join(os.getcwd(), relpath)
            dir_name = os.path.dirname(abs_path)
            filename = os.path.basename(abs_path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                return send_from_directory(dir_name, filename)
            else:
                # 如果文件不存在，返回默认图片或404
                return abort(404, description="图片文件不存在")

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            """视频流路由"""
            return Response(self.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/stats')
        def stats():
            """统计信息API"""
            elapsed = time.time() - self.start_time
            return jsonify({
                'fps': self.fps,
                'frame_count': self.frame_count,
                'processed_count': self.processed_count,
                'alert_count': self.alert_count,
                'running_time': f"{elapsed:.1f}秒",
                'status': 'Running' if self.running else 'Stopped'
            })

        @self.app.route('/alerts')
        def alerts():
            # 过滤掉 Intrusion Alert 和 Large Area Motion，取全历史最新10条有效告警
            with self.alert_lock:
                filtered_alerts = [alert for alert in self.all_alerts[::-1]
                                   if alert.get('type', '') not in ['Intrusion Alert', 'Large Area Motion']]
                alerts_data = []
                for alert in filtered_alerts[:10]:
                    alert_data = {
                        'id': alert.get('id', ''),
                        'type': alert.get('type', ''),
                        'danger_level': alert.get('danger_level', 'medium'),
                        'time': alert.get('time', ''),
                        'confidence': alert.get('confidence', 0),
                        'frame': alert.get('frame', 0),
                        'desc': alert.get('desc', ''),
                        'handled': alert.get('handled', False),
                        'handled_time': alert.get('handled_time', None),
                        'person_id': alert.get('person_id', ''),
                        'person_class': alert.get('person_class', ''),
                        # 添加位置信息
                        'location': alert.get('location', {
                            'x': 0,
                            'y': 0,
                            'rel_x': 0,
                            'rel_y': 0,
                            'description': '未知位置',
                            'region_name': alert.get('region_name', '')
                        }),
                        # 新增：声学检测结果
                        'audio_labels': alert.get('audio_labels', None)
                    }
                    alerts_data.append(alert_data)
                return jsonify(alerts_data)

        @self.app.route('/alerts/stats')
        def alert_stats():
            # 统计时也过滤掉 Intrusion Alert 和 Large Area Motion，基于 all_alerts
            with self.alert_lock:
                filtered_alerts = [a for a in self.all_alerts if a.get('type', '') not in ['Intrusion Alert', 'Large Area Motion']]
                total = len(filtered_alerts)
                handled = sum(1 for a in filtered_alerts if a.get('handled', False))
                unhandled = total - handled
            return jsonify({
                'total_alerts': total,
                'handled_alerts': handled,
                'unhandled_alerts': unhandled
            })

        @self.app.route('/alerts/handle', methods=['POST'])
        def handle_alert():
            # 处理告警（标记为已处理，同时更新数据库）
            data = request.json
            alert_id = data.get('alert_id')

            # 先更新数据库
            db_success = False
            if self.alert_database:
                db_success = self.alert_database.acknowledge_alert(alert_id)

            with self.alert_lock:
                # 查找并更新告警状态
                for alert in self.all_alerts:
                    if alert.get('id') == alert_id:
                        if not alert.get('handled', False):
                            alert['handled'] = True
                            alert['handled_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.alert_handling_stats['handled_alerts'] += 1
                            self.alert_handling_stats['unhandled_alerts'] = max(0, self.alert_handling_stats['unhandled_alerts'] - 1)
                            if db_success:
                                return jsonify({'status': 'success', 'message': 'Alert marked as handled and database updated'})
                            else:
                                return jsonify({'status': 'warning', 'message': 'Alert marked as handled, but database update failed'})
                        else:
                            return jsonify({'status': 'info', 'message': 'Alert already handled'})

                return jsonify({'status': 'error', 'message': 'Alert not found'})

        @self.app.route('/alerts/unhandle', methods=['POST'])
        def unhandle_alert():
            # 取消处理告警（标记为未处理，同时更新数据库）
            data = request.json
            alert_id = data.get('alert_id')

            # 先更新数据库
            db_success = False
            if self.alert_database:
                db_success = self.alert_database.unacknowledge_alert(alert_id)

            with self.alert_lock:
                # 查找并更新告警状态
                for alert in self.all_alerts:
                    if alert.get('id') == alert_id:
                        if alert.get('handled', False):
                            alert['handled'] = False
                            alert.pop('handled_time', None)
                            self.alert_handling_stats['handled_alerts'] = max(0, self.alert_handling_stats['handled_alerts'] - 1)
                            self.alert_handling_stats['unhandled_alerts'] += 1
                            if db_success:
                                return jsonify({'status': 'success', 'message': 'Alert marked as unhandled and database updated'})
                            else:
                                return jsonify({'status': 'warning', 'message': 'Alert marked as unhandled, but database update failed'})
                        else:
                            return jsonify({'status': 'info', 'message': 'Alert already unhandled'})

                return jsonify({'status': 'error', 'message': 'Alert not found'})

        @self.app.route('/alerts/history')
        def alerts_history():
            # 告警历史页面
            return render_template('alerts_history.html')
        
        @self.app.route('/api/alerts/history')
        def api_alerts_history():
            # 告警历史API
            if not self.alert_database:
                return jsonify({'success': False, 'message': '数据库未初始化'})
            
            try:
                # 分页参数
                page = int(request.args.get('page', 1))
                limit = int(request.args.get('limit', 10))
                offset = (page - 1) * limit
                # 新增筛选参数
                danger_level = request.args.get('danger_level')
                source_type = request.args.get('source_type')
                acknowledged = request.args.get('acknowledged')
                start_time = request.args.get('start_time', type=float)
                end_time = request.args.get('end_time', type=float)
                # 时间戳转DATETIME字符串
                def ts2dtstr(ts):
                    if ts:
                        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    return None
                start_time_str = ts2dtstr(start_time) if start_time else None
                end_time_str = ts2dtstr(end_time) if end_time else None
                # 处理 acknowledged 字符串转布尔
                if acknowledged == 'true':
                    acknowledged = True
                elif acknowledged == 'false':
                    acknowledged = False
                else:
                    acknowledged = None
                alerts = self.alert_database.get_alert_events(
                    limit=limit,
                    offset=offset,
                    danger_level=danger_level,
                    source_type=source_type,
                    acknowledged=acknowledged,
                    start_time=start_time_str,
                    end_time=end_time_str
                )
                total = self.alert_database.get_alert_count(
                    danger_level=danger_level,
                    source_type=source_type,
                    acknowledged=acknowledged,
                    start_time=start_time_str,
                    end_time=end_time_str
                )
                pages = (total + limit - 1) // limit
                
                return jsonify({
                    'success': True,
                    'alerts': alerts,
                    'total': total,
                    'page': page,
                    'pages': pages
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/alerts/statistics')
        def api_alerts_statistics():
            if not self.alert_database:
                return jsonify({'success': False, 'message': '数据库未初始化'})
            try:
                days = int(request.args.get('days', 30))
                stats = self.alert_database.get_alert_statistics(days)
                # 今日告警数
                today = date.today()
                today_start_dt = datetime.combine(today, datetime.min.time())
                today_start_str = today_start_dt.strftime('%Y-%m-%d %H:%M:%S')
                today_alerts = self.alert_database.get_alert_count(start_time=today_start_str)
                # 计算未处理告警数
                unhandled_alerts = self.alert_database.get_alert_count(acknowledged=False)
                # 计算高级别告警数
                high_level_alerts = self.alert_database.get_alert_count(level='CRITICAL')
                stats.update({
                    'today_alerts': today_alerts,
                    'unhandled_alerts': unhandled_alerts,
                    'high_level_alerts': high_level_alerts
                })
                return jsonify({
                    'success': True,
                    'stats': stats
                })
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/alerts/acknowledge', methods=['POST'])
        def api_acknowledge_alert():
            # 确认告警API
            if not self.alert_database:
                return jsonify({'success': False, 'message': '数据库未初始化'})
            
            try:
                data = request.get_json()
                alert_id = data.get('alert_id')
                
                if not alert_id:
                    return jsonify({'success': False, 'message': '缺少告警ID'})
                
                success = self.alert_database.acknowledge_alert(alert_id)
                
                if success:
                    return jsonify({'success': True, 'message': '告警已确认'})
                else:
                    return jsonify({'success': False, 'message': '告警不存在'})
                    
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/alerts/unacknowledge', methods=['POST'])
        def api_unacknowledge_alert():
            # 取消确认告警API
            if not self.alert_database:
                return jsonify({'success': False, 'message': '数据库未初始化'})
            
            try:
                data = request.get_json()
                alert_id = data.get('alert_id')
                
                if not alert_id:
                    return jsonify({'success': False, 'message': '缺少告警ID'})
                
                success = self.alert_database.unacknowledge_alert(alert_id)
                
                if success:
                    return jsonify({'success': True, 'message': '告警已取消确认'})
                else:
                    return jsonify({'success': False, 'message': '告警不存在'})
                    
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        @self.app.route('/api/alerts/source_types')
        def api_alert_source_types():
            try:
                with self.alert_database._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT DISTINCT source_type FROM alert_events")
                        types = [row['source_type'] for row in cursor.fetchall() if row['source_type']]
                return jsonify({'success': True, 'types': types})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        @self.app.route('/config/dwell_time_threshold', methods=['POST'])
        def set_dwell_time_threshold():
            """设置停留时间阈值"""
            try:
                data = request.json
                if data is None:
                    return jsonify({'success': False, 'message': 'Invalid JSON data'})
                threshold = data.get('threshold')
                if threshold is None or threshold <= 0:
                    return jsonify({'success': False, 'message': '无效的时间阈值'})
                
                # 更新危险识别器的配置
                self.danger_recognizer.dwell_time_threshold_s = threshold
                logger.info(f"停留时间阈值已更新为: {threshold}秒")
                return jsonify({'success': True, 'message': '时间阈值设置成功'})
            except Exception as e:
                logger.error(f"设置时间阈值失败: {str(e)}")
                return jsonify({'success': False, 'message': f'设置失败: {str(e)}'})

        @self.app.route('/config/alert_region', methods=['POST'])
        def set_alert_region():
            """设置警戒区域"""
            try:
                data = request.json
                if data is None:
                    return jsonify({'success': False, 'message': 'Invalid JSON data'})
                region = data.get('region')
                if not region or not isinstance(region, list) or len(region) < 3:
                    return jsonify({'success': False, 'message': '无效的警戒区域格式'})
                
                # 清除现有警戒区域并添加新的
                self.danger_recognizer.clear_alert_regions()
                self.danger_recognizer.add_alert_region(region, "User Selected Zone")
                logger.info(f"警戒区域已更新: {region}")
                return jsonify({'success': True, 'message': '警戒区域设置成功'})
            except Exception as e:
                logger.error(f"设置警戒区域失败: {str(e)}")
                return jsonify({'success': False, 'message': f'设置失败: {str(e)}'})

        @self.app.route('/config/reset_alert_region', methods=['POST'])
        def reset_alert_region():
            """重置警戒区域"""
            try:
                self.danger_recognizer.clear_alert_regions()
                logger.info("警戒区域已重置")
                return jsonify({'success': True, 'message': '警戒区域已重置'})
            except Exception as e:
                logger.error(f"重置警戒区域失败: {str(e)}")
                return jsonify({'success': False, 'message': f'重置失败: {str(e)}'})

        @self.app.route('/config/alert_region')
        def get_alert_region():
            # 假设danger_recognizer.alert_regions为 [{'points': np.array([...]), ...}, ...]
            region_points = []
            if hasattr(self, 'danger_recognizer') and hasattr(self.danger_recognizer, 'alert_regions'):
                if self.danger_recognizer.alert_regions:
                    # 只取第一个区域（如有多个可扩展）
                    pts = self.danger_recognizer.alert_regions[0].get('points', None)
                    if pts is not None:
                        # np.array转list
                        region_points = pts.tolist()
            return jsonify(success=True, region=region_points)

        @self.app.route('/config/approach_distance_threshold', methods=['POST'])
        def set_approach_distance_threshold():
            try:
                data = request.get_json(force=True)
                threshold = int(data.get('threshold', 50))
                if hasattr(self, 'danger_recognizer'):
                    self.danger_recognizer.config['danger_zone_approach_distance'] = threshold
                return jsonify(success=True)
            except Exception as e:
                return jsonify(success=False, message=str(e))

        @self.app.route('/control', methods=['POST'])
        def control():
            data = request.get_json()
            action = data.get('action')
            if action == 'start':
                self.running = True
                return jsonify({'success': True, 'message': '系统已启动'})
            elif action == 'pause':
                self.paused = not self.paused
                return jsonify({'success': True, 'message': '系统已暂停' if self.paused else '系统已恢复', 'paused': self.paused})
            elif action == 'stop':
                self.running = False
                return jsonify({'success': True, 'message': '系统已停止'})
            else:
                return jsonify({'success': False, 'message': '未知操作'})

        def run_web_server():
            """在单独的线程中运行Web服务器"""
            if self.app is not None:
                self.app.run(host='0.0.0.0', port=self.args.web_port, debug=False, threaded=True)

        # 启动Web服务器线程
        web_thread = threading.Thread(target=run_web_server)
        web_thread.daemon = True
        web_thread.start()
        logger.info(f"Web服务器已启动，访问 http://localhost:{self.args.web_port}/")

    def generate_frames(self):
        """生成帧序列用于Web流"""
        while True:
            if self.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # 约30 FPS

    def start(self):
        """启动系统"""
        self.running = True

        # 启动捕获线程
        capture_thread = threading.Thread(target=self.capture_thread_func)
        capture_thread.daemon = True
        capture_thread.start()
        self.threads.append(capture_thread)

        # 启动处理线程
        process_thread = threading.Thread(target=self.process_thread_func)
        process_thread.daemon = True
        process_thread.start()
        self.threads.append(process_thread)

        logger.info("全功能视频监控系统已启动")

        # 如果没有Web界面，则使用显示循环
        if not (self.args.web_interface and HAS_FLASK):
            try:
                self.display_loop()
            except KeyboardInterrupt:
                logger.info("用户中断")
            finally:
                self.stop()
        else:
            # 使用Web界面时，主线程等待其他线程
            try:
                while self.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                logger.info("用户中断")
            finally:
                self.stop()

    def stop(self):
        """停止系统"""
        self.running = False

        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        if self.video_writer is not None:
            self.video_writer.release()

        # 生成报告
        self.generate_report()

        logger.info("全功能视频监控系统已停止")

    def capture_thread_func(self):
        """视频捕获线程"""
        logger.info("开始视频捕获线程")

        # 打开视频源
        if self.args.source.isdigit():
            source = int(self.args.source)
        else:
            source = self.args.source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            self.running = False
            return

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

        # 获取实际分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"视频分辨率: {actual_width}x{actual_height}")

        # 设置摄像头参数（仅用于本地摄像头）
        if source == 0 or (isinstance(source, int) and source >= 0):
            # 尝试设置MJPG格式（如果支持）
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # type: ignore
            except Exception as e:
                logger.warning(f"设置MJPG格式失败: {str(e)}")
            # 尝试设置缓冲区大小最小
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 计算最大帧率
        max_fps = self.args.max_fps
        min_frame_time = 1.0 / max_fps

        frame_count = 0
        last_time = time.time()

        try:
            while self.running:
                # 处理暂停状态
                if self.paused:
                    time.sleep(0.1)
                    continue

                # 限制帧率
                current_time = time.time()
                delta = current_time - last_time
                if delta < min_frame_time:
                    time.sleep(min_frame_time - delta)

                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and not source.isdigit():
                        # 视频文件结束
                        logger.info("视频文件播放完毕")
                        if self.args.loop_video:
                            # 重新开始视频
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            self.running = False
                            break
                    else:
                        # 摄像头出错，尝试重新连接
                        logger.warning("视频帧获取失败，尝试重新连接...")
                        time.sleep(0.5)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        if not cap.isOpened():
                            logger.error("重新连接失败")
                            self.running = False
                            break
                        continue

                # 更新时间和计数
                last_time = time.time()
                frame_count += 1
                self.frame_count = frame_count
                self.current_frame = frame

                # 降低分辨率（如果启用）
                if self.args.process_scale < 1.0:
                    h, w = frame.shape[:2]
                    new_width = int(w * self.args.process_scale)
                    new_height = int(h * self.args.process_scale)
                    process_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    process_frame = frame

                # 将帧放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put(
                        (frame_count, process_frame, frame.copy() if process_frame is not frame else None, last_time))
                else:
                    # 如果队列满了，移除最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame_count, process_frame,
                                              frame.copy() if process_frame is not frame else None, last_time))
                    except:
                        pass

        except Exception as e:
            logger.error(f"视频捕获线程出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            cap.release()
            logger.info("视频捕获线程结束")

    def process_thread_func(self):
        """视频处理线程"""
        logger.info("开始视频处理线程")

        prev_frame = None
        processed_count = 0
        process_every = self.args.process_every  # 每N帧处理一次
        last_ai_frame = 0

        try:
            while self.running:
                # 处理暂停状态
                if self.paused:
                    time.sleep(0.1)
                    continue

                # 从队列中获取帧
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                except:
                    continue

                frame_id, process_frame, original_frame, timestamp = frame_data

                # 仅处理每N帧
                if frame_id % process_every == 0:
                    processed_count += 1
                    self.processed_count = processed_count

                    # 提取运动特征
                    features = self.motion_manager.extract_features(process_frame, prev_frame)

                    # AI对象检测（如果启用）
                    object_detections = None
                    if self.ai_model is not None and (frame_id - last_ai_frame) >= self.args.ai_interval:
                        try:
                            results = self.ai_model(process_frame, verbose=False)
                            object_detections = self._parse_ai_results(results)
                            last_ai_frame = frame_id
                        except Exception as e:
                            logger.error(f"AI处理出错: {str(e)}")

                    # 检测危险行为
                    alerts = self.danger_recognizer.process_frame(process_frame, features, object_detections)

                    # 音视频联动：如有打架/摔倒告警，合并音频信息
                    now = time.time()
                    for alert in alerts:
                        if alert.get('type') in ['Fighting Detection', 'Fall Detection']:
                            # 查找1.5秒内的音频事件
                            audio_msgs = []
                            for label, score, ts in list(self.recent_audio_events):
                                if now - ts < 1.5:
                                    audio_msgs.append(f"声音: {label}({score:.2f})")
                            if audio_msgs:
                                alert['desc'] += '；' + '；'.join(audio_msgs)
                                alert['audio_labels'] = [label for label, score, ts in self.recent_audio_events if now - ts < 1.5]
                    # 更新all_alerts和recent_alerts
                    for alert in alerts:
                        alert_info = {
                            'id': str(uuid.uuid4()),  # 使用UUID生成唯一ID
                            'type': alert.get('type', ''),
                            'danger_level': alert.get('danger_level', 'medium'),  # 新增：危险等级
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'confidence': float(alert.get('confidence', 0)) if alert.get('confidence', '') != '' else '',
                            'frame': int(alert.get('frame', 0)) if alert.get('frame', '') != '' else '',
                            'desc': alert.get('desc', ''),
                            'handled': False,  # 默认未处理
                            'handled_time': None,  # 处理时间
                            'person_id': alert.get('person_id', ''),  # 新增：person id
                            'person_class': alert.get('person_class', ''),  # 新增：person类别
                            # 新增：位置信息
                            'location': alert.get('location', {
                                'x': 0,
                                'y': 0,
                                'rel_x': 0,
                                'rel_y': 0,
                                'description': '未知位置',
                                'region_name': alert.get('region_name', '')
                            }),
                            # 新增：声学检测结果
                            'audio_labels': alert.get('audio_labels', None)
                        }
                        
                        with self.alert_lock:
                            self.all_alerts.append(alert_info)
                            self.alert_handling_stats['total_alerts'] = len(self.all_alerts)
                            self.alert_handling_stats['handled_alerts'] = sum(1 for a in self.all_alerts if a.get('handled', False))
                            self.alert_handling_stats['unhandled_alerts'] = self.alert_handling_stats['total_alerts'] - self.alert_handling_stats['handled_alerts']
                            # recent_alerts只保留最新10条
                            self.recent_alerts.append(alert_info)
                        
                        # 追加行为信息
                        behavior_info = f"{alert.get('type', '未知')} (置信度: {alert.get('confidence', 0):.2f}, 帧号: {alert.get('frame', '-')})"
                        if behavior_info not in self.recognized_behaviors:
                            self.recognized_behaviors.append(behavior_info)

                        # 追加交互信息（如有）
                        if object_detections and len(object_detections) > 1:
                            interaction_info = f"多对象交互检测 (对象数: {len(object_detections)}, 帧号: {alert.get('frame', '-')})"
                            if interaction_info not in self.recognized_interactions:
                                self.recognized_interactions.append(interaction_info)
                        if alert.get('type') == '入侵警告区域':
                            region_name = alert.get('region_name', '未知区域')
                            interaction_info = f"区域入侵交互 ({region_name}, 帧号: {alert.get('frame', '-')})"
                            if interaction_info not in self.recognized_interactions:
                                self.recognized_interactions.append(interaction_info)

                else:
                    # 对于跳过处理的帧，仍然要可视化，但不进行特征提取
                    vis_frame = self.visualize_frame(original_frame or process_frame, None, None, None, None)

                # 可视化结果
                vis_frame = self.visualize_frame(original_frame or process_frame, process_frame, features, alerts, object_detections)
                self.processed_frame = vis_frame  # 确保前端能持续收到视频流
                prev_frame = process_frame.copy()

                # 只保存带标识的图片到数据库（只保存一次，用第一个alert的信息）
                if self.alert_database and self.args.save_alerts and vis_frame is not None and alerts:
                    try:
                        import os
                        # 类型简称映射
                        type_map = {
                            'fall_detection': 'fall',
                            'danger_zone_dwell': 'dwell',
                            'sudden_motion': 'motion',
                            'large_area_motion': 'area',
                        }
                        alert = alerts[0]
                        type_short = type_map.get(alert.get('type', '').lower(), alert.get('type', '').lower().split('_')[0])
                        # 统一命名：大写+空格+帧号
                        alert_type = alert.get('type', 'Alert').replace('_', ' ').title()  # Danger Zone Dwell
                        frame_id = alert.get('frame', self.frame_count)
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        alert_dir = os.path.join(self.args.output, 'alerts')
                        os.makedirs(alert_dir, exist_ok=True)
                        filename = f"{alert_type}_{frame_id}_{ts}.jpg"
                        vis_path = os.path.join(alert_dir, filename)
                        cv2.imwrite(vis_path, vis_frame)
                        rel_vis_path = os.path.relpath(vis_path, os.getcwd()).replace('\\', '/')
                        # 创建告警事件对象
                        from models.alert.alert_event import AlertEvent
                        from models.alert.alert_rule import AlertLevel
                        danger_level = alert.get('danger_level', 'medium')
                        if danger_level == 'high':
                            alert_level = AlertLevel.CRITICAL
                        elif danger_level == 'medium':
                            alert_level = AlertLevel.ALERT
                        else:
                            alert_level = AlertLevel.WARNING
                        alert_event = AlertEvent.create(
                            rule_id=f"rule_{alert.get('type', 'unknown')}",
                            level=alert_level,
                            danger_level=danger_level,
                            source_type=alert.get('type', 'unknown'),
                            message=alert.get('desc', ''),
                            details={
                                'person_id': alert.get('person_id', ''),
                                'person_class': alert.get('person_class', ''),
                                'confidence': alert.get('confidence', 0),
                                'frame': alert.get('frame', 0),
                                'location': alert.get('location', {}),
                                'region_name': alert.get('region_name', '')
                            },
                            frame_idx=alert.get('frame', 0),
                            frame=None
                        )
                        # 只写入一条frame类型图片
                        rel_image_paths = {'frame': rel_vis_path}
                        logger.info(f"保存可视化告警图片: {rel_image_paths}")
                        new_id = self.alert_database.save_alert_event(alert_event, rel_image_paths)
                        logger.info(f"写入数据库返回id: {new_id}")
                        # 更新alert_info的id
                        with self.alert_lock:
                            for info in self.all_alerts:
                                if alerts and info['time'] == alert_event.to_dict().get('datetime') and info['type'] == alert_event.source_type:
                                    info['id'] = new_id
                    except Exception as e:
                        logger.error(f"保存可视化告警图片到数据库失败: {str(e)}")

                # 更新recent_alerts
                for alert in alerts:
                    alert_info = {
                        'id': str(uuid.uuid4()),  # 使用UUID生成唯一ID
                        'type': alert.get('type', ''),
                        'danger_level': alert.get('danger_level', 'medium'),  # 新增：危险等级
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'confidence': float(alert.get('confidence', 0)) if alert.get('confidence', '') != '' else '',
                        'frame': int(alert.get('frame', 0)) if alert.get('frame', '') != '' else '',
                        'desc': alert.get('desc', ''),
                        'handled': False,  # 默认未处理
                        'handled_time': None,  # 处理时间
                        'person_id': alert.get('person_id', ''),  # 新增：person id
                        'person_class': alert.get('person_class', '')  # 新增：person类别
                    }
                    
                    with self.alert_lock:
                        self.recent_alerts.append(alert_info)
                        self.alert_handling_stats['total_alerts'] += 1
                        self.alert_handling_stats['unhandled_alerts'] += 1
                        
                        if len(self.recent_alerts) > 10:
                            removed_alert = self.recent_alerts.popleft()
                            # 如果移除的告警未处理，减少未处理计数
                            if not removed_alert.get('handled', False):
                                self.alert_handling_stats['unhandled_alerts'] = max(0, self.alert_handling_stats['unhandled_alerts'] - 1)

                # 追加行为信息
                if alerts:
                    for alert in alerts:
                        behavior_info = f"{alert.get('type', '未知')} (置信度: {alert.get('confidence', 0):.2f}, 帧号: {alert.get('frame', '-')})"
                        if behavior_info not in self.recognized_behaviors:
                            self.recognized_behaviors.append(behavior_info)

                        # 追加交互信息（如有）
                        if object_detections and len(object_detections) > 1:
                            interaction_info = f"多对象交互检测 (对象数: {len(object_detections)}, 帧号: {alert.get('frame', '-')})"
                            if interaction_info not in self.recognized_interactions:
                                self.recognized_interactions.append(interaction_info)
                        if alert.get('type') == '入侵警告区域':
                            region_name = alert.get('region_name', '未知区域')
                            interaction_info = f"区域入侵交互 ({region_name}, 帧号: {alert.get('frame', '-')})"
                            if interaction_info not in self.recognized_interactions:
                                self.recognized_interactions.append(interaction_info)

                # 计算FPS
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0

                # 录制视频
                if self.video_writer is not None and vis_frame is not None:
                    # 确保尺寸匹配
                    if vis_frame.shape[1] != self.args.width or vis_frame.shape[0] != self.args.height:
                        vis_frame_resized = cv2.resize(vis_frame, (self.args.width, self.args.height))
                        self.video_writer.write(vis_frame_resized)
                    else:
                        self.video_writer.write(vis_frame)

                # 清理帧队列以保持低延迟
                while self.frame_queue.qsize() > 5:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break

        except Exception as e:
            logger.error(f"视频处理线程出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("视频处理线程结束")

    def _parse_ai_results(self, results):
        """解析AI检测结果"""
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.item() > self.args.ai_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': r.names[cls],
                        'confidence': conf
                    })
        return detections

    def visualize_frame(self, original_frame, process_frame=None, features=None, alerts=None, detections=None):
        """可视化处理结果"""
        if original_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        vis_frame = original_frame.copy()

        # 如果启用简洁模式，则跳过复杂的可视化
        if self.args.minimal_ui:
            # 仅显示简单的状态信息
            self._add_minimal_info(vis_frame)
            return vis_frame

        # 可视化特征（如果有）
        if features and process_frame is not None:
            try:
                # 绘制特征
                vis_frame = self.motion_manager.visualize_features(vis_frame, features)
            except Exception as e:
                logger.error(f"可视化特征出错: {str(e)}")

        # 可视化危险行为告警（如果有）
        if alerts:
            try:
                # 使用危险识别器的可视化功能，传递AI检测结果
                vis_frame = self.danger_recognizer.visualize(vis_frame, alerts, features, detections=detections)
            except Exception as e:
                logger.error(f"可视化告警出错: {str(e)}")
        # 如果没有告警但有AI检测结果，仍然显示检测框
        elif detections:
            for det in detections:
                try:
                    x1, y1, x2, y2 = det['bbox']
                    cls = det['class']
                    conf = det['confidence']

                    color = (0, 255, 0)  # 绿色 - 正常对象
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    # 保留AI检测结果的文字显示
                    cv2.putText(vis_frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    logger.error(f"可视化检测结果出错: {str(e)}")

        # 不再添加系统状态信息到帧
        # self._add_system_info(vis_frame)

        return vis_frame

    def _add_system_info(self, frame):
        """添加系统状态信息到帧"""
        h, w = frame.shape[:2]

        # 绘制右上角系统状态文字
        info_items = [
            f"FPS: {self.fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Processed: {self.processed_count}",
            f"Uptime: {int(time.time() - self.start_time)} s"
        ]
        for i, info in enumerate(info_items):
            color = (255, 255, 255)
            cv2.putText(frame, info, (w - 230, 25 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _add_minimal_info(self, frame):
        """添加最小化的系统信息到帧"""
        h, w = frame.shape[:2]

        # 显示FPS和告警数
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 如果有告警，显示红色警告
        if self.alert_count > 0:
            cv2.putText(frame, f"Alerts: {self.alert_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    def display_loop(self):
        """显示循环"""
        logger.info("开始显示循环")

        try:
            while self.running:
                # 显示处理后的帧
                if self.processed_frame is not None:
                    cv2.imshow("全功能视频监控系统", self.processed_frame)

                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc键退出
                    logger.info("用户按下Esc键，退出")
                    break
                elif key == ord('p'):  # 'p'键暂停/继续
                    self.paused = not self.paused
                    logger.info(f"系统{'暂停' if self.paused else '继续'}")
                elif key == ord('s'):  # 's'键保存当前帧
                    if self.processed_frame is not None:
                        save_path = os.path.join(self.args.output,
                                                 f"frame_{self.frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(save_path, self.processed_frame)
                        logger.info(f"已保存当前帧: {save_path}")
                elif key == ord('r'):  # 'r'键重置统计
                    self.danger_recognizer.reset_stats()
                    self.alert_count = 0
                    logger.info("已重置统计信息")

                # 避免CPU占用过高
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"显示循环出错: {str(e)}")
        finally:
            cv2.destroyAllWindows()
            logger.info("显示循环结束")

    def generate_report(self):
        """生成报告"""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        processed_ratio = self.processed_count / max(1, self.frame_count) * 100

        # 获取识别到的行为和交互信息
        behavior_info = self.get_recognized_behavior_info()
        
        # 打印识别到的行为和交互信息
        print("识别到的行为:")
        for behavior in behavior_info['behaviors']:
                print(f"  - {behavior}")
        print("识别到的交互:")
        for interaction in behavior_info['interactions']:
                print(f"  - {interaction}")

        report = "\n==== 系统报告 ====\n"
        report += f"运行时间: {elapsed:.2f} 秒\n"
        report += f"总帧数: {self.frame_count}\n"
        report += f"处理帧数: {self.processed_count} ({processed_ratio:.1f}%)\n"
        report += f"平均帧率: {avg_fps:.2f} FPS\n"
        report += f"告警总数: {self.alert_count}\n"

        # 添加识别到的行为信息到报告
        report += "\n识别到的行为:\n"
        # 直接用行为统计，输出所有类型的计数
        behavior_stats = self.danger_recognizer.get_behavior_stats()
        for k, v in [
            ("Sudden Motion", behavior_stats.get('sudden_motion_count', 0)),
            ("Large Area Motion", behavior_stats.get('large_area_motion_count', 0)),
            ("Fall Detection", behavior_stats.get('fall_count', 0)),
            ("Danger Zone Dwell", behavior_stats.get('danger_zone_dwell_count', 0)),
            ("Fighting Detection", behavior_stats.get('fighting_count', 0)),
            ("Abnormal Audio Event", behavior_stats.get('audio_event_count', 0)),  # 新增
        ]:
            report += f"  - {k}: {v}\n"
            
        # 添加识别到的交互信息到报告
        report += "\n识别到的交互:\n"
        if behavior_info['interactions']:
            for interaction in behavior_info['interactions']:
                report += f"  - {interaction}\n"
        else:
            report += "  - 无\n"

        # 添加告警分类统计
        alert_stats = self.danger_recognizer.get_alert_stats()
        report += "\n告警分类统计:\n"
        for alert_type, count in alert_stats.items():
            if count > 0:
                report += f"  - {alert_type}: {count}\n"

        # 新增：行为检测统计（包括不生成告警的行为）
        report += "\n行为检测统计:\n"
        for k, v in [
            ("Sudden Motion", behavior_stats.get('sudden_motion_count', 0)),
            ("Large Area Motion", behavior_stats.get('large_area_motion_count', 0)),
            ("Fall Detection", behavior_stats.get('fall_count', 0)),
            ("Danger Zone Dwell", behavior_stats.get('danger_zone_dwell_count', 0)),
            ("Fighting Detection", behavior_stats.get('fighting_count', 0)),
            ("Abnormal Audio Event", behavior_stats.get('audio_event_count', 0)),
        ]:
            report += f"  - {k}: {v}\n"

        # 新增：告警处理统计
        report += "\n告警处理统计:\n"
        report += f"  - 总告警数: {self.alert_handling_stats['total_alerts']}\n"
        report += f"  - 已处理: {self.alert_handling_stats['handled_alerts']}\n"
        report += f"  - 未处理: {self.alert_handling_stats['unhandled_alerts']}\n"
        report += f"  - 处理率: {(self.alert_handling_stats['handled_alerts'] / max(1, self.alert_handling_stats['total_alerts']) * 100):.1f}%\n"

        # 新增：详细告警处理记录
        report += "\n详细告警处理记录:\n"
        with self.alert_lock:
            for alert in self.all_alerts:
                status = "已处理" if alert.get('handled', False) else "未处理"
                handled_time = alert.get('handled_time', 'N/A')
                report += f"  - {alert.get('time', 'N/A')} | {alert.get('type', 'N/A')} | {status}"
                if alert.get('handled', False):
                    report += f" | 处理时间: {handled_time}"
                report += "\n"

        # 添加系统配置信息
        report += "\n系统配置:\n"
        for arg, value in vars(self.args).items():
            report += f"  - {arg}: {value}\n"

        logger.info(report)

        # 保存报告到文件
        report_path = os.path.join(self.args.output, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"报告已保存到: {report_path}")

    def get_recognized_behavior_info(self):
        """返回已识别的行为和交互信息"""
        return {
            'behaviors': getattr(self, 'recognized_behaviors', []),
            'interactions': getattr(self, 'recognized_interactions', [])
        }

    def add_audio_alert(self, labels, scores):
        """供音频监控模块调用，推送声学异常告警，支持一人/多人喧哗"""
        now = time.time()
        # 记录音频事件到队列
        if not labels or len(labels) == 0:
            # 无声音，不生成声学告警，仅在详情中可体现
            return
        self.recent_audio_events.append((labels, scores, now))
        # 检查最近2秒内是否有打架/摔倒类行为告警
        recent_behavior = False
        with self.alert_lock:
            for alert in list(self.all_alerts)[-10:]:
                if alert.get('type') in ['Fighting Detection', 'Fall Detection']:
                    alert_time = alert.get('time')
                    try:
                        alert_ts = time.mktime(time.strptime(alert_time, '%Y-%m-%d %H:%M:%S'))
                    except:
                        continue
                    if now - alert_ts < 2.0:
                        recent_behavior = True
                        break
        if not recent_behavior:
            # 只检测到声音，区分一人/多人喧哗
            if len(labels) == 1:
                alert_type = '一人喧哗'
                desc = f"检测到一人喧哗：{labels[0]}"
            else:
                alert_type = '多人喧哗'
                desc = f"检测到多人喧哗：{', '.join(labels)}"
            alert_info = {
                'id': str(uuid.uuid4()),
                'type': alert_type,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'confidence': float(max(scores)) if scores else 0,
                'frame': '',
                'desc': desc,
                'handled': False,
                'handled_time': None,
                'person_id': '',
                'person_class': '',
                'audio_labels': labels
            }
            with self.alert_lock:
                self.all_alerts.append(alert_info)
                self.alert_handling_stats['total_alerts'] = len(self.all_alerts)
                self.alert_handling_stats['handled_alerts'] = sum(1 for a in self.all_alerts if a.get('handled', False))
                self.alert_handling_stats['unhandled_alerts'] = self.alert_handling_stats['total_alerts'] - self.alert_handling_stats['handled_alerts']
                self.recent_alerts.append(alert_info)
        # 新增：统计声学异常（不再生成“声学异常”告警）
        if hasattr(self, 'danger_recognizer') and hasattr(self.danger_recognizer, 'behavior_stats'):
            stats = self.danger_recognizer.behavior_stats
            stats['audio_event_count'] = stats.get('audio_event_count', 0) + 1


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='全功能视频监控系统')

    # 输入参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    parser.add_argument('--loop_video', action='store_true', help='循环播放视频文件')

    # 处理参数
    parser.add_argument('--process_every', type=int, default=3, help='每N帧处理一次')
    parser.add_argument('--process_scale', type=float, default=1.0, help='处理分辨率缩放比例 (0.5=半分辨率)')
    parser.add_argument('--max_fps', type=int, default=30, help='最大帧率')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速')
    parser.add_argument('--use_motion_history', action='store_true', help='使用运动历史')
    parser.add_argument('--minimal_ui', action='store_true', help='使用最小化界面')

    # 危险行为检测参数
    parser.add_argument('--feature_threshold', type=int, default=80, help='特征点数量阈值')
    parser.add_argument('--area_threshold', type=float, default=0.05, help='运动区域阈值')
    parser.add_argument('--alert_cooldown', type=int, default=10, help='告警冷却帧数')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='最小置信度')
    parser.add_argument('--alert_region', type=str,
                        help='警戒区域, 格式为坐标点列表, 例如: "[(100,100), (300,100), (300,300), (100,300)]"')
    # 新增：危险区域停留检测参数
    parser.add_argument('--distance_threshold', type=int, default=50, help='距离危险区域边界的阈值（像素）')
    parser.add_argument('--dwell_time_threshold', type=float, default=1.0, help='危险区域停留时间阈值（秒）')

    # AI参数
    parser.add_argument('--enable_ai', action='store_true', help='启用AI功能')
    parser.add_argument('--vision_model', type=str, default='yolov8n', help='使用的视觉模型')
    parser.add_argument('--ai_interval', type=int, default=20, help='AI处理间隔帧数')
    parser.add_argument('--ai_confidence', type=float, default=0.4, help='AI检测置信度阈值')

    # Web界面参数
    parser.add_argument('--web_interface', action='store_true', help='启用Web界面')
    parser.add_argument('--web_port', type=int, default=5000, help='Web服务器端口')

    # 输出参数
    parser.add_argument('--output', type=str, default='system_output', help='输出目录')
    parser.add_argument('--record', action='store_true', help='记录视频')
    parser.add_argument('--save_alerts', action='store_true', help='保存告警帧')
    parser.add_argument('--enable_audio_monitor', action='store_true', help='启用音频监控（声学异常检测）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    system = AllInOneSystem(args)
    system.start()


if __name__ == "__main__":
    main() 