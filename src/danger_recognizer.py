#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
危险行为识别模块 - 提供用于生产环境的危险行为检测功能
"""

import logging
import numpy as np
import cv2
import time
import os
from datetime import datetime
from threading import Lock

logger = logging.getLogger("DangerRecognizer")

class DangerRecognizer:
    """危险行为识别器 - 用于检测和识别视频中的危险行为"""
    
    DANGER_TYPES = {
        'sudden_motion': 'Sudden Motion',
        'large_area_motion': 'Large Area Motion',
        'fall': 'Fall Detection',
        'abnormal_pattern': 'Abnormal Pattern',
        'intrusion': 'Intrusion Alert',
        'loitering': 'Loitering',
        'danger_zone_dwell': 'Danger Zone Dwell',  # 新增：危险区域停留告警
        'fighting': 'Fighting Detection',  # 新增：打架检测
        'approaching_danger_zone': 'Approaching Danger Zone',  # 新增：接近危险区域
    }
    
    # 危险等级映射
    DANGER_LEVELS = {
        'sudden_motion': 'low',
        'large_area_motion': 'low',
        'fall': 'medium',  # 修改：摔倒检测改为中危险
        'abnormal_pattern': 'medium',
        'intrusion': 'medium',
        'loitering': 'medium',
        'danger_zone_dwell': 'medium',  # 修改：危险区域停留改为中风险
        'fighting': 'high',  # 打架检测保持高危险
        'approaching_danger_zone': 'low',  # 新增
    }
    
    def __init__(self, config=None):
        """初始化危险行为识别器
        
        Args:
            config: 配置字典，包含检测参数和阈值
        """
        # 默认配置
        self.config = {
                    'feature_count_threshold': 50,      # 提高特征点数量阈值，减少误报
        'feature_change_ratio': 1.5,        # 提高特征变化率阈值，减少误报
        'motion_magnitude_threshold': 5,    # 提高运动幅度阈值，减少误报
        'motion_area_threshold': 0.25,      # 25%画面有大幅运动才告警
        'fall_motion_threshold': 5,         # 降低运动强度阈值
            'alert_cooldown': 15,               # 增加告警冷却时间，减少频繁告警
            'history_length': 30,
            'save_alerts': True,
            'alert_dir': 'alerts',
            'min_confidence': 0.6,              # 置信度提高到0.6
            'alert_highlight_duration': 30,     # 红框显示持续时间（帧数）
            # 新增：危险区域停留检测配置
            'distance_threshold_m': 50,         # 距离区域边界的阈值（像素）
            'dwell_time_threshold_s': 1.0,      # 停留时间阈值（秒）
            'fps': 30,                          # 帧率，用于计算时间
            # 新增：打架检测配置（大幅提升灵敏度）
            'fighting_distance_threshold': 100,   # 放宽距离
            'fighting_motion_threshold': 3.0,    # 降低运动强度要求
            'fighting_duration_frames': 3,       # 降低持续帧数
            'fighting_overlap_threshold': 0.02,  # 放宽重叠面积
            'fighting_confidence_threshold': 0.38, # 降低置信度
            'danger_zone_approach_distance': 50,  # 新增：接近危险区域距离阈值
        }
        
        # 更新用户配置
        if config:
            self.config.update(config)
        
        # 创建告警保存目录
        if self.config['save_alerts']:
            os.makedirs(self.config['alert_dir'], exist_ok=True)
        
        # 初始化状态
        self.history = []
        self.current_frame = 0
        self.last_alert_frame = 0
        self.last_fall_frame = 0  # 摔倒检测的冷却时间
        self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
        self.last_features_count = 0
        self.debug_info = {}
        self.alert_lock = Lock()
        
        # 添加用于ROI区域的属性
        self.alert_regions = []  # 告警区域列表
        
        # 新增：告警对象跟踪
        self.alerted_objects = {}  # 格式: {object_id: {'frame': frame_num, 'alert_type': type, 'bbox': bbox}}
        self.next_object_id = 0
        
        # 新增：危险区域停留检测相关
        self.danger_zone_trackers = {}  # 格式: {object_id: {'start_time': time, 'start_frame': frame, 'region_id': id, 'bbox': bbox}}
        self.dwell_alert_cooldown = {}  # 格式: {object_id: last_alert_frame} 防止重复告警
        
        # 新增：多目标跟踪
        self.tracked_persons = {}  # {id: {'bbox': [x1, y1, x2, y2], 'last_seen': frame_idx}}
        self.next_person_id = 1
        self.tracking_max_distance = 50  # 最大中心点距离，判定为同一人
        self.tracking_max_missing = 30   # 最大丢失帧数
        
        # 新增：跟踪稳定性参数
        self.tracking_iou_threshold = 0.3  # IOU匹配阈值
        self.tracking_distance_threshold = 75  # 距离匹配阈值（增加）
        self.tracking_recent_frames = 10  # 最近帧数限制
        self.tracking_min_consecutive = 3  # 最小连续跟踪帧数
        self.tracking_max_consecutive = 100  # 最大连续跟踪帧数（防止ID溢出）
        
        # 新增：帧尺寸信息
        self.frame_width = 640
        self.frame_height = 480
        
        # 新增：行为统计（用于系统报告）
        self.behavior_stats = {
            'sudden_motion_count': 0,
            'large_area_motion_count': 0,
            'fall_count': 0,
            'danger_zone_dwell_count': 0,
            'fighting_count': 0,
            'audio_event_count': 0  # 新增：声学异常统计
        }
        
        # 新增：打架检测相关
        self.fighting_history = {}  # 格式: {pair_key: {'start_frame': frame, 'duration': frames, 'motion_history': []}}
        self.last_fighting_frame = 0  # 打架检测冷却时间
        
        # 添加属性访问器，方便动态更新配置
        self._dwell_time_threshold_s = self.config['dwell_time_threshold_s']
        
        self.approach_alert_cooldown = {}  # 新增：接近危险区域告警冷却
        
        # 新增：每人历史最大高度缓存
        self.person_max_heights = {}  # person_id: max_height
        
        self.person_fall_state = {}  # person_id: bool，唯一摔倒状态
        
        logger.info(f"危险行为识别器已初始化，特征点阈值:{self.config['feature_count_threshold']}, " + 
                   f"变化率阈值:{self.config['feature_change_ratio']}, " +
                   f"危险区域停留阈值:{self.config['dwell_time_threshold_s']}秒")
    
    @property
    def dwell_time_threshold_s(self):
        """获取停留时间阈值"""
        return self._dwell_time_threshold_s
    
    @dwell_time_threshold_s.setter
    def dwell_time_threshold_s(self, value):
        """设置停留时间阈值"""
        self._dwell_time_threshold_s = value
        self.config['dwell_time_threshold_s'] = value
        logger.info(f"停留时间阈值已更新为: {value}秒")
    
    def add_alert_region(self, region, name="Alert Zone"):
        """添加告警区域
        
        Args:
            region: [(x1,y1), (x2,y2), ...] 形式的多边形区域
            name: 区域名称
        """
        self.alert_regions.append({
            'points': np.array(region, dtype=np.int32),
            'name': name,
            'color': (255, 0, 0),  # 蓝色
            'thickness': 2
        })
        logger.info(f"Added alert region: {name}")
        return len(self.alert_regions) - 1  # 返回区域ID
    
    def clear_alert_regions(self):
        """清除所有告警区域"""
        self.alert_regions.clear()
        logger.info("Cleared all alert regions")
    
    def _calculate_distance_to_region(self, bbox, region_points):
        """计算边界框到危险区域的距离
        
        Args:
            bbox: [x1, y1, x2, y2] 边界框坐标
            region_points: 危险区域的多边形点
            
        Returns:
            distance: 最小距离（像素）
        """
        x1, y1, x2, y2 = bbox
        
        # 计算边界框的四个角点和中心点
        corners = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2),  # 四个角点
            ((x1 + x2) // 2, (y1 + y2) // 2)  # 中心点
        ]
        
        min_distance = float('inf')
        
        for corner in corners:
            # 计算点到多边形的最小距离
            distance = cv2.pointPolygonTest(region_points, corner, True)
            if abs(distance) < abs(min_distance):
                min_distance = distance
        
        return abs(min_distance)
    
    def _check_bbox_intersection(self, bbox, region_points):
        """检查边界框是否与危险区域有重合
        
        Args:
            bbox: [x1, y1, x2, y2] 边界框坐标
            region_points: 危险区域的多边形点
            
        Returns:
            bool: 是否有重合
        """
        x1, y1, x2, y2 = bbox
        
        # 检查边界框的四个角点是否在区域内
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        for corner in corners:
            if cv2.pointPolygonTest(region_points, corner, False) >= 0:
                return True
        
        # 检查边界框的中心点是否在区域内
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if cv2.pointPolygonTest(region_points, (center_x, center_y), False) >= 0:
            return True
        
        # 检查边界框的边是否与区域相交（简化检查）
        # 这里使用边界框的四个边与区域的重叠检查
        bbox_rect = [x1, y1, x2, y2]
        region_rect = cv2.boundingRect(region_points)
        
        # 检查两个矩形是否重叠
        if (bbox_rect[0] < region_rect[0] + region_rect[2] and
            bbox_rect[0] + bbox_rect[2] > region_rect[0] and
            bbox_rect[1] < region_rect[1] + region_rect[3] and
            bbox_rect[1] + bbox_rect[3] > region_rect[1]):
            return True
        
        return False
    
    def _track_danger_zone_dwell(self, object_detections):
        """跟踪危险区域停留时间
        
        Args:
            object_detections: 对象检测结果列表
            
        Returns:
            alerts: 停留时间告警列表
        """
        alerts = []
        current_time = time.time()
        current_frame = self.current_frame
        
        # 首先更新人员跟踪，分配唯一ID
        self.update_person_tracking(object_detections)
        
        # # print("[调试] 当前危险区域设置:")
        # # for i, region in enumerate(self.alert_regions):
        # #     print(f"  区域{i}: {region['points'].tolist()}")
        
        # 检查每个检测到的对象
        for obj in object_detections:
            if 'bbox' not in obj or str(obj.get('class', '')).lower() != 'person':
                continue
            
            bbox = obj['bbox']
            # # print(f"[调试] 检测到person方框: {bbox}")
            
            # 使用分配的唯一person_id，如果没有则使用哈希ID作为备用
            person_id = obj.get('person_id', None)
            if person_id is not None:
                object_id = f"person_{person_id}"  # 使用唯一ID
            else:
                object_id = f"{obj.get('class', 'person')}_{hash(tuple(bbox))}"  # 备用哈希ID
            
            # 检查是否在危险区域内
            in_danger_zone = False
            region_id = None
            
            for i, region in enumerate(self.alert_regions):
                # 检查边界框是否与危险区域有重合
                is_overlap = self._check_bbox_intersection(bbox, region['points'])
                # print(f"[调试] 检查person方框 {bbox} 与区域{i} 是否重合: {is_overlap}")
                if is_overlap:
                    in_danger_zone = True
                    region_id = i
                    # 只要有重合就输出一次告警
                    # print(f"[告警] person方框 {bbox} 已进入危险区域{i}！")
                    break
            
            if in_danger_zone:
                # 对象在危险区域内，开始或继续计时
                if object_id not in self.danger_zone_trackers:
                    # 新进入危险区域，开始计时
                    self.danger_zone_trackers[object_id] = {
                        'start_time': current_time,
                        'start_frame': current_frame,
                        'region_id': region_id,
                        'bbox': bbox.copy(),
                        'person_id': person_id  # 保存person_id
                    }
                    logger.info(f"对象 {object_id} 进入危险区域 {region_id}")
                else:
                    # 已在危险区域内，检查停留时间
                    dwell_time = current_time - self.danger_zone_trackers[object_id]['start_time']
                    
                    # 检查是否超过停留时间阈值
                    if (dwell_time >= self.dwell_time_threshold_s and 
                        object_id not in self.dwell_alert_cooldown):
                        
                        # 触发告警
                        # 修复：region_id 作为索引时类型安全
                        if isinstance(region_id, int) and 0 <= region_id < len(self.alert_regions):
                            region_name = self.alert_regions[region_id]['name']
                        else:
                            region_name = str(region_id)
                        
                        # 生成告警描述
                        if person_id is not None:
                            desc = f"检测到人员（ID: {person_id}）在{region_name}内停留超过{self.config['dwell_time_threshold_s']}秒（实际{dwell_time:.1f}秒）"
                        else:
                            desc = f"检测到人员在{region_name}内停留超过{self.config['dwell_time_threshold_s']}秒（实际{dwell_time:.1f}秒）"
                        
                        alert = {
                            'type': self.DANGER_TYPES['danger_zone_dwell'],
                            'danger_level': self.DANGER_LEVELS['danger_zone_dwell'],
                            'confidence': 0.9,  # 高置信度
                            'frame': current_frame,
                            'object_id': object_id,
                            'person_id': person_id,  # 添加person_id
                            'region_id': region_id,
                            'region_name': region_name,
                            'dwell_time': dwell_time,
                            'threshold': self.config['dwell_time_threshold_s'],
                            'bbox': bbox,
                            'desc': desc  # 添加描述
                        }
                        alerts.append(alert)
                        
                        # 设置冷却时间，防止重复告警
                        self.dwell_alert_cooldown[object_id] = current_frame
                        
                        logger.info(f"危险区域停留告警: 对象 {object_id} 在区域 {region_id} 停留 {dwell_time:.2f}秒")
                        # 增加行为统计
                        self.behavior_stats['danger_zone_dwell_count'] += 1
            else:
                # 对象不在危险区域内，清除跟踪
                if object_id in self.danger_zone_trackers:
                    dwell_time = current_time - self.danger_zone_trackers[object_id]['start_time']
                    logger.info(f"对象 {object_id} 离开危险区域，停留时间: {dwell_time:.2f}秒")
                    del self.danger_zone_trackers[object_id]
                
                # 清除冷却时间
                if object_id in self.dwell_alert_cooldown:
                    del self.dwell_alert_cooldown[object_id]
        
        # 清理过期的冷却时间记录
        cooldown_duration = 30  # 30帧的冷却时间
        expired_objects = []
        for obj_id, last_frame in self.dwell_alert_cooldown.items():
            if current_frame - last_frame > cooldown_duration:
                expired_objects.append(obj_id)
        
        for obj_id in expired_objects:
            del self.dwell_alert_cooldown[obj_id]
        
        return alerts
    
    def _track_danger_zone_approach(self, object_detections):
        """检测人员接近危险区域但未进入区域的告警"""
        alerts = []
        current_frame = self.current_frame
        cooldown_frames = int(self.config['fps'])  # 1秒冷却
        approach_distance = self.config.get('danger_zone_approach_distance', 50)
        for obj in object_detections:
            if 'bbox' not in obj or str(obj.get('class', '')).lower() != 'person':
                continue
            bbox = obj['bbox']
            person_id = obj.get('person_id', None)
            object_id = f"person_{person_id}" if person_id is not None else f"person_{hash(tuple(bbox))}"  # 与停留一致
            in_any_zone = False
            for region in self.alert_regions:
                if self._check_bbox_intersection(bbox, region['points']):
                    in_any_zone = True
                    break
            if in_any_zone:
                # 进入区域后，移除冷却，后续离开可再次触发
                self.approach_alert_cooldown.pop(object_id, None)
                continue
            # 未进入区域，检测距离
            for i, region in enumerate(self.alert_regions):
                dist = self._calculate_distance_to_region(bbox, region['points'])
                if dist < approach_distance:
                    last_frame = self.approach_alert_cooldown.get(object_id, -10000)
                    if current_frame - last_frame > cooldown_frames:
                        region_name = region.get('name', str(i))
                        desc = f"检测到人员（ID: {person_id}）距离危险区域 '{region_name}' 过近（{dist:.1f} 像素），请注意安全"
                        alert = {
                            'type': self.DANGER_TYPES['approaching_danger_zone'],
                            'danger_level': self.DANGER_LEVELS['approaching_danger_zone'],
                            'confidence': 0.8,
                            'frame': current_frame,
                            'object_id': object_id,
                            'person_id': person_id,
                            'region_id': i,
                            'region_name': region_name,
                            'distance': dist,
                            'threshold': approach_distance,
                            'bbox': bbox,
                            'desc': desc
                        }
                        alerts.append(alert)
                        self.approach_alert_cooldown[object_id] = current_frame
                    break  # 只报一次
        return alerts
    
    def process_frame(self, frame, features, object_detections=None):
        """处理视频帧，分析是否存在危险行为
        
        Args:
            frame: 当前视频帧
            features: 从运动特征管理器获取的特征列表
            object_detections: 可选的物体检测结果列表
            
        Returns:
            alerts: 检测到的告警列表
        """
        self.current_frame += 1
        frame_shape = frame.shape if frame is not None else (480, 640, 3)
        
        # 更新帧尺寸信息
        self.frame_height, self.frame_width = frame_shape[:2]
        
        # 提取当前帧的运动统计
        motion_stats = self._extract_motion_stats(features, (frame_shape[1], frame_shape[0]))
        
        # 更新特征计数
        feature_count = len(features) if features else 0
        self.debug_info = {
            'frame': self.current_frame,
            'feature_count': feature_count,
            'avg_magnitude': motion_stats['avg_magnitude'],
            'max_magnitude': motion_stats['max_magnitude'],
            'motion_area': motion_stats['motion_area'],
        }
        
        # 更新历史记录
        self.history.append(motion_stats)
        if len(self.history) > self.config['history_length']:
            self.history.pop(0)
        
        # 分析危险行为
        alerts = self._analyze_danger(frame, features, object_detections)
        
        # 新增：更新告警对象跟踪
        if alerts and object_detections:
            self._update_alerted_objects(alerts, object_detections)
        
        # 清理过期的告警对象
        self._cleanup_expired_alerts()
        
        # 更新统计信息
        with self.alert_lock:
            for alert in alerts:
                if alert['type'] in self.alerts_count:
                    self.alerts_count[alert['type']] += 1
                
                # 保存告警帧
                if self.config['save_alerts'] and frame is not None:
                    self._save_alert_frame(frame, alert)
        
        # 更新上一帧的特征计数
        self.last_features_count = feature_count
        
        return alerts
    
    def _extract_motion_stats(self, features, frame_size):
        """从特征中提取运动统计
        
        Args:
            features: 特征列表
            frame_size: 帧大小 (width, height)
            
        Returns:
            stats: 统计信息字典
        """
        frame_area = frame_size[0] * frame_size[1]
        stats = {
            'timestamp': time.time(),
            'avg_magnitude': 0,
            'max_magnitude': 0,
            'motion_directions': {},
            'motion_area': 0,
            'vertical_motion': 0,
            'feature_count': len(features) if features else 0,
        }
        
        if not features:
            return stats

        motion_area = 0  # 修复：初始化motion_area
        vertical_motion = 0  # 初始化垂直运动

        # 优先用光流幅度
        if isinstance(features, dict) and 'flow_mean_magnitude' in features:
            stats['avg_magnitude'] = features['flow_mean_magnitude']
            stats['max_magnitude'] = features.get('flow_max_magnitude', features['flow_mean_magnitude'])
            # 用motion_vectors数量估算运动面积
            if 'motion_vectors' in features:
                # 16为采样步长，motion_vectors数量*采样面积
                motion_area = len(features['motion_vectors']) * 16 * 16
                
                # 计算垂直运动分量
                if 'motion_vectors' in features and len(features['motion_vectors']) > 0:
                    vertical_components = []
                    for vector in features['motion_vectors']:
                        if len(vector) >= 2:  # 确保有y分量
                            vertical_components.append(vector[1])  # y分量
                    if vertical_components:
                        vertical_motion = np.mean(vertical_components)
        else:
            # 兼容原有特征点方式
            magnitudes = []
            vertical_components = []
            if features:
                for feature in features:
                    if hasattr(feature, 'magnitude'):
                        magnitudes.append(feature.magnitude)
                        motion_area += feature.magnitude * 10  # 原有面积估算
                    
                    # 计算垂直运动分量
                    if hasattr(feature, 'position') and hasattr(feature, 'end_position'):
                        start_y = feature.position[1] if len(feature.position) > 1 else 0
                        end_y = feature.end_position[1] if len(feature.end_position) > 1 else 0
                        vertical_components.append(end_y - start_y)
            
            if magnitudes:
                stats['avg_magnitude'] = np.mean(magnitudes)
                stats['max_magnitude'] = np.max(magnitudes)
            
            if vertical_components:
                vertical_motion = np.mean(vertical_components)

        stats['motion_area'] = min(1.0, motion_area / frame_area)
        stats['vertical_motion'] = vertical_motion
        return stats
    
    def _analyze_danger(self, frame, features, object_detections=None):
        """分析危险行为"""
        if len(self.history) < 3:
            return []
        alerts = []
        if object_detections is None:
            object_detections = []
        frame_height, frame_width = frame.shape[:2]
        
        # 检查冷却时间（摔倒检测除外）
        in_cooldown = self.current_frame - self.last_alert_frame <= self.config['alert_cooldown']

        # 五种类型的告警处理
        # 1. 检测突然运动（仅统计，不告警）
        current_features = self.history[-1]['feature_count']
        prev_features = self.history[-2]['feature_count'] if len(self.history) > 1 else 0
        feature_change_ratio = current_features / prev_features if prev_features > 0 else 1.0

        if (feature_change_ratio > self.config['feature_change_ratio'] and 
            current_features > self.config['feature_count_threshold']):
            # 仅更新统计，不生成告警
            with self.alert_lock:
                self.behavior_stats['sudden_motion_count'] += 1

        # 2. 检测大面积运动（仅统计，不告警）
        if features:
            motion_area_ratio = len([f for f in features if hasattr(f, 'magnitude') and 
                                   f.magnitude > self.config['motion_magnitude_threshold']]) / len(features)
            
            if motion_area_ratio > self.config['motion_area_threshold']:
                # 仅更新统计，不生成告警
                with self.alert_lock:
                    self.behavior_stats['large_area_motion_count'] += 1

        # 3. 检测警戒区域入侵和停留时间
        if object_detections and self.alert_regions:
            # 新增：检测接近危险区域
            approach_alerts = self._track_danger_zone_approach(object_detections)
            # 3.1 检测危险区域停留时间
            dwell_alerts = self._track_danger_zone_dwell(object_detections)
            # 合并两类告警
            for alert in approach_alerts:
                alerts.append(alert)
            for alert in dwell_alerts:
                alerts.append(alert)

        # 4. 打架检测
        if object_detections:
            # 移除重复统计，只在_detect_fighting中统计
            fighting_alerts = self._detect_fighting(object_detections, features)
            alerts.extend(fighting_alerts)

        # 5. 摔倒检测唯一状态机
        for obj in object_detections:
            if str(obj.get('class', '')).lower() == 'person':
                person_id = obj.get('person_id')
                if person_id is None:
                    continue  # 必须有person_id才能唯一判定
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                wh_ratio = height / width if width > 0 else 0
                wh_fall = wh_ratio < 0.7
                bottom_contact = y2 > frame_height * 0.9
                # 垂直运动
                vertical_motion = 0
                if isinstance(features, dict) and 'motion_vectors' in features:
                    verticals = [v[3] for v in features['motion_vectors'] if len(v) >= 4]
                    if verticals:
                        vertical_motion = np.mean(verticals)
                vertical_fall = vertical_motion > 2.5
                # 综合判据
                is_fall = (wh_fall and bottom_contact) or (wh_fall and vertical_fall)
                # 站起判据
                stand_up = (wh_ratio > 1.0) or (y2 < frame_height * 0.8)
                prev_state = self.person_fall_state.get(person_id, False)
                # 日志
                if is_fall or prev_state:
                    msg = f"[摔倒唯一状态机] frame={self.current_frame}, person_id={person_id}, bbox={bbox}, wh_ratio={wh_ratio:.2f}, wh_fall={wh_fall}, bottom_contact={bottom_contact}, vertical_motion={vertical_motion:.2f}, vertical_fall={vertical_fall}, is_fall={is_fall}, stand_up={stand_up}, prev_state={prev_state}"
                    try:
                        with open("fall_debug.log", "a", encoding="utf-8") as f:
                            f.write(msg + "\n")
                    except Exception:
                        pass
                if is_fall and not prev_state:
                    alert = {
                        'type': self.DANGER_TYPES['fall'],
                        'danger_level': self.DANGER_LEVELS['fall'],
                        'confidence': 0.9,
                        'frame': self.current_frame,
                        'object_id': person_id,
                        'bbox': bbox,
                        'desc': f"检测到人员（ID: {person_id}）发生摔倒！"
                    }
                    alerts.append(alert)
                    self.person_fall_state[person_id] = True
                elif not is_fall and prev_state and stand_up:
                    self.person_fall_state[person_id] = False
        # 如果在冷却时间内，只返回摔倒检测结果
        if in_cooldown:
            return alerts
        # 将摔倒告警添加到总告警列表中
        alerts.extend(alerts)
        # 如果有告警，更新最后告警帧
        if alerts:
            self.last_alert_frame = self.current_frame
        return alerts
    
    def _detect_fighting(self, object_detections, features):
        """检测打架行为 - 改进版本
        
        改进点：
        1. 实际距离估算：基于人物在画面中的位置和大小
        2. 持续性检测：要求持续一定时间的斗殴行为
        3. 运动模式分析：分析人物的运动模式
        4. 更灵敏的检测：优化阈值和检测条件
        
        Args:
            object_detections: 对象检测结果列表
            features: 运动特征
            
        Returns:
            alerts: 打架检测告警列表
        """
        alerts = []
        
        # 获取所有人员检测结果
        persons = [det for det in object_detections if str(det.get('class', '')).lower() == 'person']
        
        if len(persons) < 2:
            return alerts  # 至少需要2个人才能打架
        
        # 检查冷却时间
        if self.current_frame - self.last_fighting_frame <= 15:  # 原45帧，缩短冷却
            return alerts
        
        # 分析运动特征
        motion_intensity = 0
        if isinstance(features, dict) and 'flow_mean_magnitude' in features:
            motion_intensity = features['flow_mean_magnitude']
        elif features:
            magnitudes = [f.magnitude for f in features if hasattr(f, 'magnitude')]
            if magnitudes:
                motion_intensity = np.mean(magnitudes)
        
        # 计算人员之间的距离和运动特征
        person_pairs = []
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                person1 = persons[i]
                person2 = persons[j]
                
                # 计算两个人员边界框中心点之间的距离
                bbox1 = person1['bbox']
                bbox2 = person2['bbox']
                center1_x = (bbox1[0] + bbox1[2]) // 2
                center1_y = (bbox1[1] + bbox1[3]) // 2
                center2_x = (bbox2[0] + bbox2[2]) // 2
                center2_y = (bbox2[1] + bbox2[3]) // 2
                
                # 计算像素距离
                pixel_distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
                
                # 估算实际距离（基于人物在画面中的位置和大小）
                estimated_real_distance = self._estimate_real_distance(bbox1, bbox2, center1_x, center1_y, center2_x, center2_y)
                
                # 计算边界框的重叠面积
                overlap_area = self._calculate_overlap_area(bbox1, bbox2)
                
                # 计算人物大小（用于距离估算）
                person1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                person2_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                
                person_pairs.append({
                    'person1': person1,
                    'person2': person2,
                    'pixel_distance': pixel_distance,
                    'real_distance': estimated_real_distance,
                    'overlap_area': overlap_area,
                    'center1': (center1_x, center1_y),
                    'center2': (center2_x, center2_y),
                    'person1_size': person1_size,
                    'person2_size': person2_size,
                    'motion_intensity': motion_intensity
                })
        
        # 持续性打架检测
        current_time = time.time()
        fighting_detected = False
        confidence = 0.0
        condition_details = []
        
        for pair in person_pairs:
            # 新增：必须有重叠面积>60 或 距离<80px，否则直接跳过（更宽松）
            if pair['overlap_area'] < 60 and pair['pixel_distance'] > 80:
                continue
            # 必须有剧烈运动
            motion_threshold = self.config['fighting_motion_threshold']
            if pair['motion_intensity'] <= motion_threshold:
                continue
            # 生成配对键（用于跟踪持续性）
            person1_id = pair['person1'].get('person_id', 'unknown')
            person2_id = pair['person2'].get('person_id', 'unknown')
            pair_key = f"{min(person1_id, person2_id)}_{max(person1_id, person2_id)}"
            
            # 检查是否满足打架条件
            pair_confidence = 0.0
            pair_conditions = []
            
            # 条件1：实际距离很近（考虑人物大小和位置）
            distance_threshold = self.config['fighting_distance_threshold']
            if pair['pixel_distance'] < distance_threshold:
                # 根据人物大小调整距离权重
                avg_size = (pair['person1_size'] + pair['person2_size']) / 2
                size_factor = min(avg_size / 10000, 2.0)  # 人物越大，距离权重越高
                adjusted_distance_score = max(0, 1 - pair['pixel_distance'] / (distance_threshold * size_factor))
                pair_confidence += adjusted_distance_score * 0.3
                pair_conditions.append(f"距离很近({pair['pixel_distance']:.1f}px)")
            
            # 条件2：边界框有重叠（人员接触）
            if pair['overlap_area'] > 0:
                overlap_score = min(pair['overlap_area'] / 1000, 1.0)  # 重叠面积越大，分数越高
                pair_confidence += overlap_score * 0.4
                pair_conditions.append(f"人员接触(重叠{pair['overlap_area']:.1f})")
            
            # 条件3：剧烈运动（运动强度大于阈值）
            motion_threshold = self.config['fighting_motion_threshold']
            if pair['motion_intensity'] > motion_threshold:
                motion_score = min((pair['motion_intensity'] - motion_threshold) / motion_threshold, 1.0)
                pair_confidence += motion_score * 0.3
                pair_conditions.append(f"剧烈运动({pair['motion_intensity']:.1f})")
            
            # 条件4：多个人员同时运动（轻微排除摔倒场景）
            if len(persons) >= 2:
                # 检查是否为摔倒场景：如果运动主要是垂直向下或水平运动，可能是摔倒
                is_fall_scenario = False
                if isinstance(features, dict) and 'motion_vectors' in features:
                    vertical_motions = []
                    horizontal_motions = []
                    for v in features['motion_vectors']:
                        if len(v) >= 4:
                            vertical_motions.append(v[3])  # dy
                            horizontal_motions.append(v[2])  # dx
                    
                    if vertical_motions and horizontal_motions:
                        avg_vertical = np.mean(vertical_motions)
                        avg_horizontal = np.mean(horizontal_motions)
                        # 提高阈值，减少误判
                        if abs(avg_vertical) > 6 or abs(avg_horizontal) > 6:  # 从5提高到6
                            is_fall_scenario = True
                
                # 只有在不是摔倒场景时才加分
                if not is_fall_scenario:
                    pair_confidence += 0.2
                    pair_conditions.append("多人同时运动")
                else:
                    pair_conditions.append("检测到可能的摔倒场景，显著降低打架置信度")
                    pair_confidence -= 0.2  # 原-0.05，扣分更重
            
            # 新增条件5：检查人物大小是否合理（更宽松，原2000→800）
            min_reasonable_size = 800
            if person1_size > min_reasonable_size and person2_size > min_reasonable_size:
                pair_confidence += 0.1
                pair_conditions.append("人物大小合理")
            
            # 新增条件6：检查运动模式（避免静态或缓慢移动被误判）
            if pair['motion_intensity'] > motion_threshold * 2.0:  # 从1.5提高到2.0，要求更高的运动强度
                pair_confidence += 0.2
                pair_conditions.append("高强度运动")
            
            # 新增条件7：排除摔倒场景（检查运动特征，更宽松）
            fall_indicators = 0
            if isinstance(features, dict) and 'motion_vectors' in features:
                # 检查是否有大量垂直向下运动（提高阈值）
                vertical_down_motions = 0
                for v in features['motion_vectors']:
                    if len(v) >= 4 and v[3] > 3:  # dy > 3 表示明显的向下运动
                        vertical_down_motions += 1
                
                if vertical_down_motions > len(features['motion_vectors']) * 0.4:  # 40%以上明显向下运动
                    fall_indicators += 1
                
                # 检查运动后是否静止（更严格的条件）
                if len(self.history) >= 5:
                    recent_magnitudes = [h['avg_magnitude'] for h in self.history[-3:]]
                    earlier_magnitudes = [h['avg_magnitude'] for h in self.history[-5:-3]]
                    if earlier_magnitudes and recent_magnitudes:
                        if np.mean(earlier_magnitudes) > 5 and np.mean(recent_magnitudes) < 1.5:
                            fall_indicators += 1
            
            # 如果有摔倒指标，适度降低打架置信度
            if fall_indicators >= 2:
                pair_confidence -= 0.3
                pair_conditions.append(f"检测到{fall_indicators}个摔倒指标，适度降低打架置信度")
            elif fall_indicators == 1:
                pair_confidence -= 0.1
                pair_conditions.append("检测到1个摔倒指标，轻微降低打架置信度")
            
            # 持续性检测
            if pair_key not in self.fighting_history:
                # 新配对，初始化历史记录
                if pair_confidence >= self.config['fighting_confidence_threshold'] * 0.7:  # 原0.9，放宽初始阈值
                    self.fighting_history[pair_key] = {
                        'start_frame': self.current_frame,
                        'duration': 1,
                        'motion_history': [pair['motion_intensity']],
                        'confidence_history': [pair_confidence],
                        'last_update': current_time
                    }
            else:
                # 更新现有配对的历史记录
                history = self.fighting_history[pair_key]
                
                # 检查时间连续性（允许短暂中断）
                time_gap = current_time - history['last_update']
                frame_gap = self.current_frame - history['start_frame'] - history['duration']
                
                if time_gap < 2.0 and frame_gap < 10:  # 允许2秒或10帧的短暂中断
                    # 更新历史记录
                    history['duration'] += 1
                    history['motion_history'].append(pair['motion_intensity'])
                    history['confidence_history'].append(pair_confidence)
                    history['last_update'] = current_time
                    
                    # 保持历史记录长度
                    if len(history['motion_history']) > 30:
                        history['motion_history'] = history['motion_history'][-30:]
                        history['confidence_history'] = history['confidence_history'][-30:]
                    
                    # 检查是否满足持续性要求
                    duration_threshold = self.config['fighting_duration_frames']
                    if history['duration'] >= duration_threshold:
                        # 计算平均置信度
                        avg_confidence = np.mean(history['confidence_history'][-duration_threshold:])
                        if avg_confidence >= self.config['fighting_confidence_threshold']:
                            fighting_detected = True
                            confidence = avg_confidence
                            condition_details = pair_conditions
                            condition_details.append(f"持续{history['duration']}帧")
                            
                            # 生成告警信息
                            center_x = (pair['center1'][0] + pair['center2'][0]) // 2
                            center_y = (pair['center1'][1] + pair['center2'][1]) // 2
                            
                            # 使用类的帧尺寸属性
                            rel_x = round(center_x / self.frame_width * 100, 2)
                            rel_y = round(center_y / self.frame_height * 100, 2)
                            x_desc = "左侧" if rel_x < 33.33 else "右侧" if rel_x > 66.67 else "中间"
                            y_desc = "上方" if rel_y < 33.33 else "下方" if rel_y > 66.67 else "中间"
                            
                            location = {
                                'x': center_x,
                                'y': center_y,
                                'rel_x': rel_x,
                                'rel_y': rel_y,
                                'description': f"画面{x_desc}{y_desc}"
                            }
                            
                            desc = f"检测到人员（ID: {person1_id}）和人员（ID: {person2_id}）在{location['description']}发生持续性打架行为"
                            
                            alert = {
                                'type': self.DANGER_TYPES['fighting'],
                                'danger_level': self.DANGER_LEVELS['fighting'],
                                'confidence': confidence,
                                'frame': self.current_frame,
                                'location': location,
                                'desc': desc,
                                'person1_id': person1_id,
                                'person2_id': person2_id,
                                'pixel_distance': pair['pixel_distance'],
                                'real_distance': pair['real_distance'],
                                'motion_intensity': pair['motion_intensity'],
                                'duration': history['duration'],
                                'condition_details': condition_details
                            }
                            alerts.append(alert)
                            # 增加行为统计
                            self.behavior_stats['fighting_count'] += 1
                            self.last_fighting_frame = self.current_frame
                else:
                    # 时间间隔太长，重置历史记录
                    del self.fighting_history[pair_key]
        
        # 清理过期的打架历史记录
        self._cleanup_fighting_history()
        
        return alerts
    
    def _estimate_real_distance(self, bbox1, bbox2, center1_x, center1_y, center2_x, center2_y):
        """估算两个人物之间的实际距离
        
        改进的距离估算方法：
        1. 考虑人物在画面中的相对位置
        2. 考虑人物大小与距离的关系
        3. 考虑画面透视效果
        4. 更保守的距离估算，减少误判
        
        Args:
            bbox1, bbox2: 两个边界框
            center1_x, center1_y, center2_x, center2_y: 两个中心点坐标
            
        Returns:
            estimated_distance: 估算的实际距离（相对单位）
        """
        # 计算人物大小
        size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 计算人物在画面中的位置（距离画面中心的距离）
        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2
        
        distance1_to_center = np.sqrt((center1_x - frame_center_x)**2 + (center1_y - frame_center_y)**2)
        distance2_to_center = np.sqrt((center2_x - frame_center_x)**2 + (center2_y - frame_center_y)**2)
        
        # 计算像素距离
        pixel_distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # 改进的距离因子计算
        # 1. 人物大小因子：人物越大，距离因子越小（更近）
        avg_size = (size1 + size2) / 2
        size_factor = min(avg_size / 8000, 1.5)  # 限制大小因子的影响
        
        # 2. 位置因子：距离画面中心越远，距离因子越大（更远）
        avg_distance_to_center = (distance1_to_center + distance2_to_center) / 2
        position_factor = 1 + (avg_distance_to_center / 200)  # 位置影响
        
        # 3. 大小差异因子：如果两个人物大小差异很大，可能距离较远
        size_ratio = max(size1, size2) / min(size1, size2) if min(size1, size2) > 0 else 1
        size_diff_factor = min(size_ratio / 2, 1.5)  # 限制大小差异的影响
        
        # 4. 综合距离因子（更保守的估算）
        combined_factor = (size_factor + position_factor + size_diff_factor) / 3
        
        # 估算实际距离（更保守）
        estimated_distance = pixel_distance * combined_factor * 1.2  # 增加1.2倍保守系数
        
        return estimated_distance
    
    def _cleanup_fighting_history(self):
        """清理过期的打架历史记录"""
        current_time = time.time()
        to_delete = []
        
        for pair_key, history in self.fighting_history.items():
            # 如果超过5秒没有更新，删除记录
            if current_time - history['last_update'] > 5.0:
                to_delete.append(pair_key)
        
        for pair_key in to_delete:
            del self.fighting_history[pair_key]
    
    def _calculate_overlap_area(self, bbox1, bbox2):
        """计算两个边界框的重叠面积
        
        Args:
            bbox1: [x1, y1, x2, y2] 第一个边界框
            bbox2: [x1, y1, x2, y2] 第二个边界框
            
        Returns:
            overlap_area: 重叠面积
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算重叠区域
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0  # 没有重叠
        
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        return overlap_area
    
    def _save_alert_frame(self, frame, alert):
        """保存告警帧
        
        Args:
            frame: 当前视频帧
            alert: 告警信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{alert['type']}_{self.current_frame}_{timestamp}.jpg"
        filepath = os.path.join(self.config['alert_dir'], filename)
        
        # 在图像上添加告警信息
        vis_frame = frame.copy()
        
        # 绘制告警信息 - 已移除文字显示
        # alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
        # cv2.putText(vis_frame, alert_text, (10, vis_frame.shape[0] - 20), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加告警标识（不再使用全局红色边框）- 已移除
        # cv2.putText(vis_frame, "ALERT DETECTED", (10, 30), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 添加更多的调试信息 - 已移除
        # y_offset = 30
        # for key, value in self.debug_info.items():
        #     text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        #     cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     y_offset += 20
        
        # 绘制警戒区域
        for region in self.alert_regions:
            cv2.polylines(vis_frame, [region['points']], True, region['color'], region['thickness'])
        
        # 保存图像
        cv2.imwrite(filepath, vis_frame)
        logger.info(f"已保存告警帧: {filepath}")
    
    def update_person_tracking(self, detections):
        """改进的人员跟踪方法
        
        使用更稳定的跟踪算法：
        1. IOU匹配（优先）
        2. 中心点距离匹配（备用）
        3. 外观特征匹配（可选）
        4. 状态管理（处理遮挡和重新出现）
        """
        # 只处理person
        persons = [det for det in detections if str(det.get('class', '')).lower() == 'person']
        updated_ids = set()
        
        # 计算当前帧所有检测框的IOU矩阵
        current_bboxes = [det['bbox'] for det in persons]
        if not current_bboxes:
            return
        
        # 与历史跟踪的IOU匹配
        for det in persons:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            
            best_match_id = None
            best_iou = self.tracking_iou_threshold  # 使用配置的IOU阈值
            best_distance = float('inf')
            
            # 1. 优先使用IOU匹配
            for pid, info in self.tracked_persons.items():
                prev_bbox = info['bbox']
                iou = self._calculate_iou(bbox, prev_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = pid
            
            # 2. 如果IOU匹配失败，使用距离匹配
            if best_match_id is None:
                for pid, info in self.tracked_persons.items():
                    prev_bbox = info['bbox']
                    pcx = (prev_bbox[0] + prev_bbox[2]) // 2
                    pcy = (prev_bbox[1] + prev_bbox[3]) // 2
                    dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                    
                    # 使用配置的距离阈值和帧数限制
                    if (dist < self.tracking_distance_threshold and 
                        dist < best_distance and
                        self.current_frame - info['last_seen'] < self.tracking_recent_frames):
                        best_distance = dist
                        best_match_id = pid
            
            # 3. 分配ID
            if best_match_id is not None:
                # 更新已有ID
                self.tracked_persons[best_match_id] = {
                    'bbox': bbox, 
                    'last_seen': self.current_frame,
                    'consecutive_frames': self.tracked_persons[best_match_id].get('consecutive_frames', 0) + 1
                }
                det['person_id'] = best_match_id
                updated_ids.add(best_match_id)
            else:
                # 分配新ID
                pid = self.next_person_id
                self.next_person_id += 1
                self.tracked_persons[pid] = {
                    'bbox': bbox, 
                    'last_seen': self.current_frame,
                    'consecutive_frames': 1
                }
                det['person_id'] = pid
                updated_ids.add(pid)
        
        # 4. 清理长时间未出现的ID（更宽松的清理策略）
        to_del = []
        for pid, info in self.tracked_persons.items():
            frames_missing = self.current_frame - info['last_seen']
            consecutive_frames = info.get('consecutive_frames', 0)
            
            # 根据连续跟踪帧数调整清理策略
            if consecutive_frames < self.tracking_min_consecutive:
                max_missing = 15  # 15帧
            else:
                max_missing = self.tracking_max_missing  # 30帧
            
            # 防止连续帧数过多导致ID溢出
            if consecutive_frames > self.tracking_max_consecutive:
                to_del.append(pid)
            elif frames_missing > max_missing:
                to_del.append(pid)
        
        for pid in to_del:
            del self.tracked_persons[pid]
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IOU
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            iou: IOU值 (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize(self, frame, alerts=None, features=None, show_debug=True, detections=None):
        """可视化危险行为检测结果
        
        Args:
            frame: 当前视频帧
            alerts: 告警列表
            features: 特征列表
            show_debug: 是否显示调试信息
            detections: AI检测结果，用于对象级告警显示
        Returns:
            vis_frame: 可视化后的视频帧
        """
        if frame is None:
            return None
        
        vis_frame = frame.copy()
        
        # 绘制警戒区域 - 始终显示为蓝色，不显示名称
        for region in self.alert_regions:
            region_color = (255, 0, 0)  # 蓝色
            cv2.polylines(vis_frame, [region['points']], True, region_color, region['thickness'])
            # 移除区域名称显示
            # x, y = region['points'].mean(axis=0).astype(int)
            # cv2.putText(vis_frame, region['name'], (x, y), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)
        
        # 显示危险区域停留时间信息 - 已移除文字显示
        # if self.danger_zone_trackers:
        #     y_offset = 60  # 从顶部开始显示
        #     for obj_id, tracker in self.danger_zone_trackers.items():
        #         dwell_time = time.time() - tracker['start_time']
        #         region_id = tracker['region_id']
        #         # 修正：region_id 可能为 None 或超出范围，需检查
        #         if isinstance(region_id, int) and 0 <= region_id < len(self.alert_regions):
        #             region_name = self.alert_regions[region_id]['name']
        #         else:
        #             region_name = str(region_id)
        #         # 使用英文避免中文显示问题
        #         text = f"Dwell: {dwell_time:.1f}s - {region_name}"
        #         cv2.putText(vis_frame, text, (10, y_offset), 
        #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #         y_offset += 20
        
        # 智能告警可视化 - 移除告警文本显示
        # if alerts:
        #     # 显示告警文本（顶部）
        #     for i, alert in enumerate(alerts):
        #         alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
        #         text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        #         cv2.rectangle(vis_frame, (10, 10 + i*35), (10 + text_size[0] + 10, 10 + i*35 + text_size[1] + 10), 
        #                     (0, 0, 0), -1)
        #         cv2.putText(vis_frame, alert_text, (15, 10 + i*35 + text_size[1]), 
        #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 优化：使用告警对象跟踪来精确显示红框
        if detections:
            self.update_person_tracking(detections)
            for det in detections:
                if str(det.get('class', '')).lower() == 'person':
                    x1, y1, x2, y2 = det['bbox']
                    pid = det.get('person_id', -1)
                    thickness = 3
                    # 添加告警标识 - 已移除ID显示
                    # if pid != -1: # 只有当有ID时才显示
                    #     cv2.putText(vis_frame, f"ID:{pid}", (x1, y1 - 25),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # 在危险区域内或处于告警状态的person显示红框
                    if pid != -1: # 只有当有ID时才检查
                        is_alerted = self.is_object_alerted(det)
                        in_danger_zone = False
                        if self.alert_regions:
                            for region in self.alert_regions:
                                if self._check_bbox_intersection(det['bbox'], region['points']):
                                    in_danger_zone = True
                                    break
                        if is_alerted or in_danger_zone:
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                    else:
                        # 没有ID的person，显示绿色框
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                else:
                    color = (0, 255, 0)  # 绿色 - 非person对象
                    thickness = 2
                    cv2.rectangle(vis_frame, (det['bbox'][0], det['bbox'][1]), (det['bbox'][2], det['bbox'][3]), color, thickness)
                    # 保留非person对象的文字显示
                    cv2.putText(vis_frame, f"{det.get('class', 'unknown')} {det.get('confidence', 0.8):.2f}", (det['bbox'][0], det['bbox'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示调试信息 - 已移除
        # if show_debug:
        #     y_offset = 30
        #     for key, value in self.debug_info.items():
        #         text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        #         cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #         y_offset += 20
        #     
        #     # 显示告警对象统计
        #     alerted_count = len([obj for obj in self.alerted_objects.values() if obj['class'] == 'person'])
        #     cv2.putText(vis_frame, f"Alerted Objects: {alerted_count}", (10, y_offset), 
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        #     y_offset += 20
        #     # 新增：显示跟踪统计信息
        #     tracked_count = len(self.tracked_persons)
        #     cv2.putText(vis_frame, f"Tracked Persons: {tracked_count}", (10, y_offset), 
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #     y_offset += 20
        #     # 显示跟踪质量信息
        #     if self.tracked_persons:
        #         avg_consecutive = np.mean([info.get('consecutive_frames', 0) for info in self.tracked_persons.values()])
        #         cv2.putText(vis_frame, f"Avg Consecutive: {avg_consecutive:.1f}", (10, y_offset), 
        #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #         y_offset += 20
        #     for i, (alert_type, count) in enumerate(self.alerts_count.items()):
        #         if count > 0:
        #             cv2.putText(vis_frame, f"{alert_type}: {count}", (vis_frame.shape[1] - 180, y_offset), 
        #                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        #             y_offset += 20
        return vis_frame
    
    def get_alert_stats(self):
        """获取告警统计
        
        Returns:
            stats: 告警统计信息
        """
        with self.alert_lock:
            return self.alerts_count.copy()
    
    def get_behavior_stats(self):
        """获取行为统计（包括不生成告警的行为）
        
        Returns:
            stats: 行为统计信息
        """
        # 确保所有行为类型都在统计中
        stats = self.behavior_stats.copy()
        # 补全所有类型，防止外部遗漏
        for k in ['sudden_motion_count', 'large_area_motion_count', 'fall_count', 'danger_zone_dwell_count', 'fighting_count']:
            if k not in stats:
                stats[k] = 0
        return stats
    
    def reset_stats(self):
        """重置告警统计"""
        with self.alert_lock:
            self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
            # 重置行为统计
            self.behavior_stats = {
                'sudden_motion_count': 0,
                'large_area_motion_count': 0,
                'fall_count': 0,
                'danger_zone_dwell_count': 0,
                'fighting_count': 0,
                'audio_event_count': 0  # 新增：声学异常统计
            }
    
    def reset(self):
        """重置危险行为识别器"""
        with self.alert_lock:
            self.history = []
            self.current_frame = 0
            self.last_alert_frame = 0
            self.last_fall_frame = 0  # 摔倒检测的冷却时间
            self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
            self.last_features_count = 0
            # 清理告警对象跟踪
            self.alerted_objects = {}
            self.next_object_id = 0
            # 清理危险区域停留跟踪
            self.danger_zone_trackers = {}
            self.dwell_alert_cooldown = {}
            # 清理多目标跟踪
            self.tracked_persons = {}
            self.next_person_id = 1
            # 清理打架检测历史
            self.fighting_history = {}
            self.last_fighting_frame = 0
            logger.info("危险行为识别器已重置")

    def _cleanup_expired_alerts(self):
        """清理过期的告警对象"""
        highlight_duration = self.config['alert_highlight_duration']
        expired_objects = []
        
        for obj_id, obj_info in self.alerted_objects.items():
            if self.current_frame - obj_info['frame'] > highlight_duration:
                expired_objects.append(obj_id)
        
        for obj_id in expired_objects:
            del self.alerted_objects[obj_id]
    
    def _update_alerted_objects(self, alerts, object_detections):
        """更新告警对象跟踪
        
        Args:
            alerts: 当前帧的告警列表
            object_detections: 当前帧的对象检测结果
        """
        for alert in alerts:
            alert_type = alert['type']
            
            # 根据告警类型确定哪些对象应该被标记
            if alert_type == self.DANGER_TYPES['fall']:
                # 摔倒检测：标记所有person（因为摔倒检测是基于整体运动）
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person':
                        self._add_alerted_object(det, alert_type)
            
            elif alert_type == self.DANGER_TYPES['danger_zone_dwell']:
                # 危险区域停留告警：标记所有person
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person':
                        self._add_alerted_object(det, alert_type)
            
            elif alert_type == self.DANGER_TYPES['fighting']:
                # 打架检测：标记参与打架的人员
                person1_id = alert.get('person1_id', None)
                person2_id = alert.get('person2_id', None)
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person':
                        det_person_id = det.get('person_id', None)
                        if det_person_id in [person1_id, person2_id]:
                            self._add_alerted_object(det, alert_type)
            
            else:
                # 其他告警类型：标记所有person
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person':
                        self._add_alerted_object(det, alert_type)
    
    def _add_alerted_object(self, detection, alert_type):
        """添加告警对象
        
        Args:
            detection: 检测到的对象
            alert_type: 告警类型
        """
        # 使用bbox作为对象标识（简化版本）
        bbox = detection.get('bbox', [0, 0, 0, 0])
        obj_key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        
        # 检查是否已经存在相同的告警对象
        for obj_id, obj_info in self.alerted_objects.items():
            if obj_info['bbox'] == bbox and obj_info['alert_type'] == alert_type:
                # 更新现有对象的帧号
                obj_info['frame'] = self.current_frame
                return
        
        # 添加新的告警对象
        self.alerted_objects[self.next_object_id] = {
            'frame': self.current_frame,
            'alert_type': alert_type,
            'bbox': bbox,
            'class': detection.get('class', 'unknown')
        }
        self.next_object_id += 1
    
    def is_object_alerted(self, detection):
        """检查对象是否处于告警状态
        
        Args:
            detection: 检测到的对象
            
        Returns:
            bool: 是否处于告警状态
        """
        if not detection or 'bbox' not in detection:
            return False
        
        bbox = detection['bbox']
        
        # 检查是否有匹配的告警对象
        for obj_info in self.alerted_objects.values():
            if obj_info['bbox'] == bbox:
                return True
        
        return False

# 简单测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建危险行为识别器
    recognizer = DangerRecognizer()
    
    # 添加警戒区域
    recognizer.add_alert_region([(100, 100), (300, 100), (300, 300), (100, 300)], "禁区1")
    
    # 模拟特征
    class DummyFeature:
        def __init__(self, pos, end_pos, mag):
            self.position = pos
            self.end_position = end_pos
            self.magnitude = mag
    
    features = [
        DummyFeature((150, 150), (160, 160), 15),
        DummyFeature((200, 200), (210, 220), 20),
        DummyFeature((50, 50), (60, 70), 25),
    ]
    
    # 创建测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 处理帧
    alerts = recognizer.process_frame(frame, features)
    
    # 可视化
    vis_frame = recognizer.visualize(frame, alerts, features)
    
    # 显示结果
    if vis_frame is not None:
        cv2.imshow("Test", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()