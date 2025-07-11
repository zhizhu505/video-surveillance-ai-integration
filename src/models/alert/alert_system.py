import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# 正确的路径 (在models和模块名之间多加一层 .alert)
from models.alert.alert_rule import AlertRule, AlertRuleConfig, AlertLevel
from models.alert.alert_event import AlertEvent, AlertEventStore
# 同样，在models和模块名之间多加一层 .alert
from models.alert.alert_processor import AlertProcessor
from models.alert.alert_plugins import NotificationManager


class AlertSystem:
    """
    主告警系统，集成所有告警组件并协调它们的运行。
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化告警系统。
        
        Args:
            config_path: 配置文件路径
        """
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AlertSystem")
        
        # 初始化配置
        self.config = self._load_config(config_path)
        
        # 创建输出目录
        self.output_dir = self.config.get("output_dir", "alert_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化规则配置
        rules_file = self.config.get("rules_file", os.path.join(self.output_dir, "alert_rules.json"))
        self.rules_config = self._init_rules(rules_file)
        
        # 初始化告警处理器
        self.alert_processor = AlertProcessor(self.rules_config)
        
        # 初始化事件存储
        event_store_config = self.config.get("event_store", {})
        max_events = event_store_config.get("max_events", 1000)
        events_dir = os.path.join(self.output_dir, "events")
        self.event_store = AlertEventStore(max_events=max_events, output_dir=events_dir)
        
        # 初始化通知管理器
        notification_config = self.config.get("notifications", {})
        self.notification_manager = NotificationManager(notification_config)
        
        # 初始化状态
        self.is_initialized = True
        self.is_running = False
        self.frame_queue = []
        self.queue_lock = threading.Lock()
        self.processing_thread = None
        
        self.logger.info("Alert system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置或使用默认配置。"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
        
        # 默认配置
        return {
            "output_dir": "alert_output",
            "rules_file": "alert_rules.json",
            "event_store": {
                "max_events": 1000
            },
            "notifications": {
                "file": {
                    "type": "file",
                    "enabled": True,
                    "min_level": "info",
                    "output_file": "alerts.log",
                    "append": True
                }
            },
            "processing": {
                "max_queue_size": 30,
                "process_every_n_frames": 5
            }
        }
    
    def _init_rules(self, rules_file: str) -> AlertRuleConfig:
        """从文件初始化告警规则或创建默认规则。"""
        rules_config = None
        
        # 首先尝试从文件加载
        if os.path.exists(rules_file):
            rules_config = AlertRuleConfig.load_from_file(rules_file)
        
            # 如果未加载规则，创建默认规则
        if not rules_config or not rules_config.rules:
            rules_config = AlertRuleConfig()
            rules_config.create_default_rules()
            
            # 保存默认规则
            rules_config.save_to_file(rules_file)
        
        self.logger.info(f"Loaded {len(rules_config.rules)} alert rules")
        return rules_config
    
    def start(self) -> bool:
        """启动告警系统处理线程。"""
        if not self.is_initialized:
            self.logger.error("Alert system not initialized")
            return False
        
        if self.is_running:
            self.logger.warning("Alert system already running")
            return True
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Alert system started")
        return True
    
    def stop(self) -> bool:
        """停止告警系统处理线程。"""
        if not self.is_running:
            return True
        
        # 通知线程停止
        self.is_running = False
        
        # 等待线程终止
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # 关闭通知管理器
        self.notification_manager.shutdown()
        
        # 保存规则
        rules_file = self.config.get("rules_file", os.path.join(self.output_dir, "alert_rules.json"))
        self.rules_config.save_to_file(rules_file)
        
        # 保存事件
        events_file = os.path.join(self.output_dir, "alert_events.json")
        self.event_store.save_to_file(events_file)
        
        self.logger.info("Alert system stopped")
        return True
    
    def process_frame(self, frame_idx: int, frame: Any,
                     behavior_results: List[Any] = None,
                     tracks: List[Dict[str, Any]] = None,
                     motion_features: List[Any] = None,
                     scene_data: Dict[str, Any] = None,
                     process_now: bool = False) -> Optional[List[AlertEvent]]:
        """
        处理帧和相关数据以生成告警。
        
        Args:
            frame_idx: 帧索引
            frame: 当前帧
            behavior_results: 行为分析结果
            tracks: 对象跟踪结果
            motion_features: 运动特征
            scene_data: 场景分析数据
            process_now: 立即处理而不是排队
            
        Returns:
            如果process_now为True，则返回生成的告警列表，否则返回None
        """
        if not self.is_initialized:
            self.logger.error("Alert system not initialized")
            return None
        
        # 创建帧数据包
        frame_data = {
            "frame_idx": frame_idx,
            "frame": frame.copy() if frame is not None else None,
            "behavior_results": behavior_results,
            "tracks": tracks,
            "motion_features": motion_features,
            "scene_data": scene_data,
            "timestamp": time.time()
        }
        
        # 如果请求立即处理，则处理
        if process_now:
            return self._process_frame_data(frame_data)
        
        # 否则排队进行后台处理
        with self.queue_lock:
            # 添加到队列
            self.frame_queue.append(frame_data)
            
            # 如果队列太大，则修剪
            max_queue_size = self.config.get("processing", {}).get("max_queue_size", 30)
            if len(self.frame_queue) > max_queue_size:
                # 保留最近的帧
                self.frame_queue = self.frame_queue[-max_queue_size:]
        
        return None
    
    def _process_frame_data(self, frame_data: Dict[str, Any]) -> List[AlertEvent]:
        """处理帧数据包并生成告警。"""
        try:
            # 从包中提取数据
            frame_idx = frame_data["frame_idx"]
            frame = frame_data["frame"]
            behavior_results = frame_data["behavior_results"]
            tracks = frame_data["tracks"]
            motion_features = frame_data["motion_features"]
            scene_data = frame_data["scene_data"]
            
            # 使用告警处理器处理
            alerts = self.alert_processor.process_frame_data(
                frame_idx=frame_idx,
                frame=frame,
                behavior_results=behavior_results,
                tracks=tracks,
                motion_features=motion_features,
                scene_data=scene_data
            )
            
            # 存储和通知每个告警
            for alert in alerts:
                # 存储在事件存储中
                self.event_store.add_event(alert)
                
                # 发送通知
                self.notification_manager.notify(alert)
                
                # 记录告警
                self.logger.info(f"Alert generated: {alert.level.name} - {alert.message}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error processing frame data: {e}")
            return []
    
    def _processing_worker(self):
        """后台处理帧队列。"""
        process_every_n = self.config.get("processing", {}).get("process_every_n_frames", 5)
        frame_counter = 0
        
        while self.is_running:
            frame_data_to_process = None
            
            # 从队列中获取帧
            with self.queue_lock:
                if self.frame_queue:
                    frame_counter += 1
                    
                    # 只处理每N帧以减少负载
                    if frame_counter >= process_every_n:
                        frame_data_to_process = self.frame_queue.pop(0)
                        frame_counter = 0
            
            # 如果一个帧被出队，则处理它
            if frame_data_to_process:
                self._process_frame_data(frame_data_to_process)
            else:
                # Sleep if queue is empty
                time.sleep(0.1)
    
    def get_recent_alerts(self, count: int = 10, 
                         level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts as dictionaries.
        
        Args:
            count: Maximum number of alerts to return
            level: Filter by alert level
            
        Returns:
            List of alert dictionaries
        """
        events = self.event_store.get_events(count=count, level=level)
        return [event.to_dict() for event in events]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get statistics about alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        return self.event_store.get_stats()
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        return self.event_store.acknowledge_event(alert_id)
    
    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """
        Enable or disable an alert rule.
        
        Args:
            rule_id: ID of the rule to update
            enabled: Whether the rule should be enabled
            
        Returns:
            True if updated, False if not found
        """
        rule = self.rules_config.get_rule(rule_id)
        if rule:
            rule.enabled = enabled
            return True
        return False
    
    def update_rule(self, rule: AlertRule) -> bool:
        """
        Update an alert rule.
        
        Args:
            rule: Updated rule
            
        Returns:
            True if updated
        """
        self.rules_config.add_rule(rule)
        return True
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Get all alert rules as dictionaries.
        
        Returns:
            List of rule dictionaries
        """
        return [rule.to_dict() for rule in self.rules_config.rules] 