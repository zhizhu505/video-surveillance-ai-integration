import time
import uuid
import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from models.alert_rule import AlertLevel


@dataclass
class AlertEvent:
    """
    告警事件数据结构，包含一次告警的所有信息。
    """
    id: str                      # 唯一事件标识符
    rule_id: str                 # 触发该告警的规则ID
    level: AlertLevel            # 告警级别
    source_type: str             # 触发告警的来源类型
    timestamp: float             # 告警生成时的Unix时间戳
    message: str                 # 可读的告警信息
    details: Dict[str, Any]      # 告警的详细信息
    frame_idx: int               # 触发告警时的视频帧编号
    frame: Optional[np.ndarray] = None  # 触发告警时的图像帧
    thumbnail: Optional[np.ndarray] = None  # 图像帧的小缩略图

    # 附加元数据
    acknowledged: bool = False   # 告警是否已被确认
    related_events: List[str] = field(default_factory=list)  # 相关事件的ID列表
    
    # 创建告警事件
    @classmethod
    def create(cls, rule_id: str, level: AlertLevel, source_type: str, 
               message: str, details: Dict[str, Any], frame_idx: int,
               frame: Optional[np.ndarray] = None) -> 'AlertEvent':
        """
        Create a new alert event.
        
        Args:
            rule_id: ID of the rule that triggered this alert
            level: Alert level
            source_type: Type of source that triggered the alert
            message: Human-readable message
            details: Detailed information about the alert
            frame_idx: Frame index when the alert was triggered
            frame: Frame image when the alert was triggered
            
        Returns:
            New AlertEvent instance
        """
        event_id = f"evt_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        timestamp = time.time()
        
        # Create thumbnail if frame is provided
        thumbnail = None
        if frame is not None:
            # Resize to small thumbnail for storage efficiency
            h, w = frame.shape[:2]
            thumbnail_size = (min(320, w), min(240, h * 320 // w))
            thumbnail = cv2.resize(frame, thumbnail_size)
        
        return cls(
            id=event_id,
            rule_id=rule_id,
            level=level,
            source_type=source_type,
            timestamp=timestamp,
            message=message,
            details=details,
            frame_idx=frame_idx,
            frame=frame.copy() if frame is not None else None,
            thumbnail=thumbnail
        )
    
    # 将告警事件转换为字典，用于序列化
    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_images: Whether to include encoded images
            
        Returns:
            Dictionary representation of the event
        """
        result = {
            'id': self.id,
            'rule_id': self.rule_id,
            'level': self.level.name,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'message': self.message,
            'details': self.details,
            'frame_idx': self.frame_idx,
            'acknowledged': self.acknowledged,
            'related_events': self.related_events,
            'datetime': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果需要包含图像，将缩略图编码为base64
        if include_images and self.thumbnail is not None:
            # Encode thumbnail as base64 if requested
            import base64
            _, buffer = cv2.imencode('.jpg', self.thumbnail)
            result['thumbnail_base64'] = base64.b64encode(buffer).decode('utf-8')
        
        return result
    # 将告警事件的图像保存到磁盘
    def save_images(self, output_dir: str) -> Dict[str, str]:
        """
        Save images associated with the event to disk.
        
        Args:
            output_dir: Directory to save images
            
        Returns:
            Dictionary with paths to saved images
        """
        import os
        paths = {} # paths 用于存储保存后的图像路径
        
        # 创建指定的输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 时间戳用于唯一文件名
        ts = datetime.fromtimestamp(self.timestamp).strftime('%Y%m%d_%H%M%S')
        
        # 如果图像帧存在，保存为jpg格式
        if self.frame is not None:
            frame_path = os.path.join(output_dir, f"{self.id}_{ts}_frame.jpg")
            cv2.imwrite(frame_path, self.frame)
            paths['frame'] = frame_path
        
        # 如果缩略图存在，保存为jpg格式
        if self.thumbnail is not None:
            thumb_path = os.path.join(output_dir, f"{self.id}_{ts}_thumb.jpg")
            cv2.imwrite(thumb_path, self.thumbnail)
            paths['thumbnail'] = thumb_path
        
        return paths


# 存储、管理和操作多个告警事件
class AlertEventStore:
    """
    Store for alert events with persistence capabilities.
    """
    
    def __init__(self, max_events: int = 1000, output_dir: str = "alert_events"):
        """
        Initialize the alert event store.
        
        Args:
            max_events: Maximum number of events to keep in memory
            output_dir: Directory to save event images
        """
        self.events = [] # 存储告警事件的列表
        self.max_events = max_events # 最大存储事件数
        self.output_dir = output_dir # 保存事件图像的目录
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # 添加一个新的告警事件到事件存储中
    def add_event(self, event: AlertEvent, save_images: bool = True) -> str:
        """
        添加一个新的告警事件到存储中。

        参数:
            event: 要添加的告警事件
            save_images: 是否将事件相关图片保存到磁盘

        返回:
            添加事件的ID
        """
        # Save images if requested
        if save_images:
            event.save_images(self.output_dir)
        
        # 将事件添加到存储中
        self.events.append(event)
        
        # 如果事件数超过最大存储数，删除最早的事件
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # 返回添加事件的ID
        return event.id
    # 根据事件ID查找并返回对应的告警事件对象
    def get_event(self, event_id: str) -> Optional[AlertEvent]:
        """
        根据事件ID获取告警事件。

        参数:
            event_id: 要获取的事件ID

        返回:
            如果找到则返回事件对象，否则返回None
        """
        for event in self.events:
            if event.id == event_id:
                return event
        return None
    # 批量获取（筛选）告警事件
    def get_events(self, count: int = None, level: Optional[AlertLevel] = None, 
                  source_type: Optional[str] = None, 
                  acknowledged: Optional[bool] = None) -> List[AlertEvent]:
        """
        获取筛选后的告警事件列表。

        参数:
            count: 返回的最大事件数量（按最新优先）
            level: 按告警级别筛选
            source_type: 按来源类型筛选
            acknowledged: 按是否已确认筛选

        返回:
            匹配条件的事件列表
        """
        filtered = self.events
        
        # 应用筛选条件
        if level is not None:
            filtered = [e for e in filtered if e.level == level]
        
        if source_type is not None:
            filtered = [e for e in filtered if e.source_type == source_type]
        
        if acknowledged is not None:
            filtered = [e for e in filtered if e.acknowledged == acknowledged]
        
        # 按时间戳排序（最新优先）
        filtered.sort(key=lambda e: e.timestamp, reverse=True)
        
        # 如果指定数量，限制返回数量
        if count is not None:
            filtered = filtered[:count]
        
        return filtered
    
    # 标记事件为已确认
    def acknowledge_event(self, event_id: str) -> bool:
        """
        标记事件为已确认状态。
        
        Args:
            event_id: 需要确认的事件ID
            
        Returns:
            若成功确认则返回True，若未找到对应事件则返回False
        """
        event = self.get_event(event_id)
        if event:
            event.acknowledged = True
            return True
        return False
    
    # 将事件数据保存到 JSON 文件中
    def save_to_file(self, filepath: str) -> bool:
        """
        将事件保存到JSON文件。
        
        Args:
            filepath: 保存事件的文件路径
            
        Returns:
            成功保存返回True，否则返回False
        """
        try:
            # 将事件对象转换为字典
            events_data = [event.to_dict(include_images=False) for event in self.events]
            
            with open(filepath, 'w') as f:
                json.dump(events_data, f, indent=2)
            
            return True
        except Exception as e:
            import logging
            logging.error(f"Failed to save events to {filepath}: {e}")
            return False
    
    # 统计事件数据并返回结构化的统计信息
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored events.
        
        Returns:
            Dictionary with event statistics
        """
        if not self.events:
            return {
                'total': 0,
                'by_level': {},
                'by_source': {},
                'acknowledged': 0
            }
        
        # Count events by level and source
        levels = {}
        sources = {}
        acknowledged = 0
        
        for event in self.events:
            # Count by level
            level_name = event.level.name
            levels[level_name] = levels.get(level_name, 0) + 1
            
            # Count by source
            sources[event.source_type] = sources.get(event.source_type, 0) + 1
            
            # Count acknowledged
            if event.acknowledged:
                acknowledged += 1
        
        return {
            'total': len(self.events),
            'by_level': levels,
            'by_source': sources,
            'acknowledged': acknowledged
        } 