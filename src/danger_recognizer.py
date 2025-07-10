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
            'fall_motion_threshold': 5,         # 降低摔倒检测阈值，增强灵敏度
            'alert_cooldown': 15,               # 增加告警冷却时间，减少频繁告警
            'history_length': 30,
            'save_alerts': True,
            'alert_dir': 'alerts',
            'min_confidence': 0.6,              # 置信度提高到0.6
            'alert_highlight_duration': 30,     # 红框显示持续时间（帧数）
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
        
        logger.info(f"危险行为识别器已初始化，特征点阈值:{self.config['feature_count_threshold']}, " + 
                   f"变化率阈值:{self.config['feature_change_ratio']}")
    
    def add_alert_region(self, region, name="警戒区"):
        """添加告警区域
        
        Args:
            region: [(x1,y1), (x2,y2), ...] 形式的多边形区域
            name: 区域名称
        """
        self.alert_regions.append({
            'points': np.array(region, dtype=np.int32),
            'name': name,
            'color': (0, 0, 255),
            'thickness': 2
        })
        logger.info(f"已添加告警区域: {name}")
        return len(self.alert_regions) - 1  # 返回区域ID
    
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
            stats['max_magnitude'] = features['flow_max_magnitude']
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
        """分析危险行为
        
        Args:
            frame: 当前视频帧
            features: 特征列表
            object_detections: 物体检测结果
            
        Returns:
            alerts: 告警列表
        """
        if len(self.history) < 3:
            return []
        
        # 检查冷却时间（摔倒检测除外）
        if self.current_frame - self.last_alert_frame <= self.config['alert_cooldown']:
            # 即使有冷却时间，也要检查摔倒检测
            fall_alerts = []
            if len(self.history) >= 10:  # 需要更多历史记录来判断摔倒事件
                recent_vertical_motions = [h['vertical_motion'] for h in self.history[-8:]]
                max_vertical_motion = np.max(recent_vertical_motions)
                recent_magnitudes = [h['avg_magnitude'] for h in self.history[-8:]]
                recent_avg = np.mean(recent_magnitudes[-3:])
                earlier_avg = np.mean(recent_magnitudes[:-3]) if len(recent_magnitudes) > 3 else 0
                current_features = self.history[-1]['feature_count']
                prev_features = self.history[-2]['feature_count'] if len(self.history) > 1 else 0

                confidence = 0.0
                condition_details = []
                # 条件1：垂直运动大（降低阈值）
                if max_vertical_motion > 12:
                    confidence += 0.5
                    condition_details.append("垂直运动大")
                # 条件2：运动后静止（放宽判据）
                if earlier_avg > 6 and recent_avg < 3:
                    confidence += 0.5
                    condition_details.append("运动后静止")
                # 条件3：特征点突增（可选，权重降低）
                if prev_features > 0 and current_features > prev_features * 1.5 and current_features > 5:
                    confidence += 0.2
                    condition_details.append("特征点突增")
                # 条件4：垂直运动持续（可选，权重降低）
                vertical_motion_count = sum(1 for v in recent_vertical_motions if v > 8)
                if vertical_motion_count >= 2:
                    confidence += 0.2
                    condition_details.append("垂直运动持续")

                fall_cooldown_frames = 10
                cooldown_ok = self.current_frame - getattr(self, 'last_fall_frame', 0) > fall_cooldown_frames
                if (confidence >= 0.8 and cooldown_ok):
                    print(f"[调试] 摔倒事件检测触发: 置信度={confidence:.2f}, 满足条件: {condition_details}")
                    print(f"[调试] 详细参数: max_vertical_motion={max_vertical_motion:.2f}, earlier_avg={earlier_avg:.2f}, recent_avg={recent_avg:.2f}, vertical_motion_count={vertical_motion_count}")
                    fall_alerts.append({
                        'type': self.DANGER_TYPES['fall'],
                        'confidence': confidence,
                        'frame': self.current_frame,
                        'vertical_motion': max_vertical_motion,
                        'threshold': self.config['fall_motion_threshold'],
                        'event_id': f"fall_{self.current_frame}",
                    })
                    self.last_fall_frame = self.current_frame
                elif confidence >= 0.5:
                    print(f"[调试] 摔倒检测接近触发但未达到阈值: 置信度={confidence:.2f}, 满足条件: {condition_details}")
                    print(f"[调试] 详细参数: max_vertical_motion={max_vertical_motion:.2f}, earlier_avg={earlier_avg:.2f}, recent_avg={recent_avg:.2f}, vertical_motion_count={vertical_motion_count}, cooldown_ok={cooldown_ok}")
                    if not cooldown_ok:
                        print(f"[调试] 冷却时间阻止: 当前帧={self.current_frame}, 上次摔倒帧={getattr(self, 'last_fall_frame', 0)}, 需要等待={fall_cooldown_frames - (self.current_frame - getattr(self, 'last_fall_frame', 0))}帧")
            return fall_alerts  # 只返回摔倒检测结果
        
        alerts = []
        feature_count = len(features) if features else 0
        # 调试输出每帧特征
        print(f"[调试] 帧号: {self.current_frame}, 特征点数: {feature_count}, "
              f"平均幅度: {self.history[-1]['avg_magnitude'] if self.history else 0:.2f}, "
              f"最大幅度: {self.history[-1]['max_magnitude'] if self.history else 0:.2f}, "
              f"运动面积: {self.history[-1]['motion_area'] if self.history else 0:.4f}, "
              f"垂直运动: {self.history[-1]['vertical_motion'] if self.history else 0:.2f}")
        
        # 摔倒检测独立冷却时间（更短，确保及时检测）
        fall_cooldown = 5  # 摔倒检测冷却时间设为5帧
        last_fall_frame = getattr(self, 'last_fall_frame', 0)
        can_detect_fall = self.current_frame - last_fall_frame > fall_cooldown
        
        # 1. 特征数量检测（优化版）
        if feature_count > self.config['feature_count_threshold']:
            # 检查历史记录，避免持续的高特征点数量触发告警
            if len(self.history) >= 3:
                recent_features = [h['feature_count'] for h in self.history[-3:]]
                avg_recent_features = sum(recent_features) / len(recent_features)
                
                # 只有当当前特征点数量显著高于最近平均值时才告警
                if feature_count > avg_recent_features * 1.5:  # 当前特征点比平均值高50%以上
                    confidence = min(1.0, feature_count / (self.config['feature_count_threshold'] * 3))
                    if confidence >= self.config['min_confidence']:
                        print(f"[判定] Sudden Motion: 特征点数={feature_count}, 阈值={self.config['feature_count_threshold']}, 置信度={confidence:.2f}, 平均特征点={avg_recent_features:.1f}")
                        alerts.append({
                            'type': self.DANGER_TYPES['sudden_motion'],
                            'confidence': confidence,
                            'frame': self.current_frame,
                            'feature_count': feature_count,
                            'threshold': self.config['feature_count_threshold'],
                            'avg_recent_features': avg_recent_features,
                        })
            else:
                # 历史记录不足时，使用简单阈值
                confidence = min(1.0, feature_count / (self.config['feature_count_threshold'] * 3))
                if confidence >= self.config['min_confidence']:
                    print(f"[判定] Sudden Motion: 特征点数={feature_count}, 阈值={self.config['feature_count_threshold']}, 置信度={confidence:.2f}")
                    alerts.append({
                        'type': self.DANGER_TYPES['sudden_motion'],
                        'confidence': confidence,
                        'frame': self.current_frame,
                        'feature_count': feature_count,
                        'threshold': self.config['feature_count_threshold'],
                    })
        
        # 2. 特征变化率检测
        if self.last_features_count > 0:
            feature_change_ratio = feature_count / max(1, self.last_features_count)
            if (feature_change_ratio > self.config['feature_change_ratio'] and 
                feature_count > self.config['feature_count_threshold'] / 2):
                
                confidence = min(1.0, (feature_change_ratio - 1) / self.config['feature_change_ratio'])
                if confidence >= self.config['min_confidence']:
                    print(f"[判定] Feature Change: 变化率={feature_change_ratio:.2f}, 阈值={self.config['feature_change_ratio']}, 置信度={confidence:.2f}")
                    alerts.append({
                        'type': self.DANGER_TYPES['sudden_motion'],
                        'confidence': confidence,
                        'frame': self.current_frame,
                        'change_ratio': feature_change_ratio,
                        'threshold': self.config['feature_change_ratio'],
                    })
        
        # 3. 运动幅度检测
        current = self.history[-1]['avg_magnitude']
        if len(self.history) >= 5:
            prev_avg = sum(h['avg_magnitude'] for h in self.history[-6:-1]) / 5
            magnitude_ratio = current / max(0.1, prev_avg)
            
            if (magnitude_ratio > 1.2 and 
                current > self.config['motion_magnitude_threshold']):
                
                confidence = min(1.0, current / (self.config['motion_magnitude_threshold'] * 2))
                if confidence >= self.config['min_confidence']:
                    print(f"[判定] Motion Magnitude: 当前={current:.2f}, 阈值={self.config['motion_magnitude_threshold']}, 置信度={confidence:.2f}")
                    alerts.append({
                        'type': self.DANGER_TYPES['sudden_motion'],
                        'confidence': confidence,
                        'frame': self.current_frame,
                        'magnitude': current,
                        'threshold': self.config['motion_magnitude_threshold'],
                    })
        
        # 4. 大面积运动检测（简化版）
        motion_area = self.history[-1]['motion_area']
        if motion_area > self.config['motion_area_threshold']:
            print(f"[判定] Large Area Motion: 面积={motion_area:.4f}, 阈值={self.config['motion_area_threshold']}, 置信度={min(1.0, motion_area / self.config['motion_area_threshold']):.2f}")
            confidence = min(1.0, motion_area / self.config['motion_area_threshold'])
            if confidence >= self.config['min_confidence']:
                alerts.append({
                    'type': self.DANGER_TYPES['large_area_motion'],
                    'confidence': confidence,
                    'frame': self.current_frame,
                    'area': motion_area,
                    'threshold': self.config['motion_area_threshold'],
                })
        else:
            print(f"[调试] 大面积运动检测: 当前面积={motion_area:.4f} 未超过阈值={self.config['motion_area_threshold']:.4f}")
        
        # 5. 检测警戒区域入侵
        if object_detections and self.alert_regions:
            for obj in object_detections:
                if 'bbox' in obj:  # 确保对象有边界框
                    x1, y1, x2, y2 = obj['bbox']
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    for region_idx, region in enumerate(self.alert_regions):
                        if cv2.pointPolygonTest(region['points'], (center_x, center_y), False) >= 0:
                            print(f"[判定] Intrusion Alert: 目标={obj.get('class', 'unknown')}, 区域={region['name']}, 置信度={obj.get('confidence', 0.8):.2f}")
                            # 目标在警戒区域内
                            alerts.append({
                                'type': self.DANGER_TYPES['intrusion'],
                                'confidence': obj.get('confidence', 0.8),
                                'frame': self.current_frame,
                                'object': obj.get('class', 'unknown'),
                                'region': region_idx,
                                'region_name': region['name'],
                            })
        
        # 6. 摔倒检测（事件级检测）
        if len(self.history) >= 10:  # 需要更多历史记录来判断摔倒事件
            recent_vertical_motions = [h['vertical_motion'] for h in self.history[-8:]]
            max_vertical_motion = np.max(recent_vertical_motions)
            avg_vertical_motion = np.mean(recent_vertical_motions)
            recent_magnitudes = [h['avg_magnitude'] for h in self.history[-8:]]
            recent_avg = np.mean(recent_magnitudes[-3:])
            earlier_avg = np.mean(recent_magnitudes[:-3]) if len(recent_magnitudes) > 3 else 0
            current_features = self.history[-1]['feature_count']
            prev_features = self.history[-2]['feature_count'] if len(self.history) > 1 else 0

            confidence = 0.0
            condition_details = []
            
            # 条件1：垂直运动大
            if max_vertical_motion > 12:
                confidence += 0.5
                condition_details.append("垂直运动大")
            # 条件2：运动后静止
            if earlier_avg > 6 and recent_avg < 3:
                confidence += 0.5
                condition_details.append("运动后静止")
            # 条件3：特征点突增（可选，权重降低）
            if prev_features > 0 and current_features > prev_features * 1.5 and current_features > 5:
                confidence += 0.2
                condition_details.append("特征点突增")
            # 条件4：垂直运动持续（可选，权重降低）
            vertical_motion_count = sum(1 for v in recent_vertical_motions if v > 8)
            if vertical_motion_count >= 2:
                confidence += 0.2
                condition_details.append("垂直运动持续")

            fall_cooldown_frames = 10
            cooldown_ok = self.current_frame - getattr(self, 'last_fall_frame', 0) > fall_cooldown_frames
            
            if (confidence >= 0.8 and cooldown_ok):
                print(f"[调试] 摔倒事件检测触发: 置信度={confidence:.2f}, 满足条件: {condition_details}")
                print(f"[调试] 详细参数: max_vertical_motion={max_vertical_motion:.2f}, earlier_avg={earlier_avg:.2f}, recent_avg={recent_avg:.2f}, vertical_motion_count={vertical_motion_count}")
                alerts.append({
                    'type': self.DANGER_TYPES['fall'],
                    'confidence': confidence,
                    'frame': self.current_frame,
                    'vertical_motion': max_vertical_motion,
                    'threshold': self.config['fall_motion_threshold'],
                    'event_id': f"fall_{self.current_frame}",
                })
                self.last_fall_frame = self.current_frame
            elif confidence >= 0.5:
                print(f"[调试] 摔倒检测接近触发但未达到阈值: 置信度={confidence:.2f}, 满足条件: {condition_details}")
                print(f"[调试] 详细参数: max_vertical_motion={max_vertical_motion:.2f}, earlier_avg={earlier_avg:.2f}, recent_avg={recent_avg:.2f}, vertical_motion_count={vertical_motion_count}, cooldown_ok={cooldown_ok}")
                if not cooldown_ok:
                    print(f"[调试] 冷却时间阻止: 当前帧={self.current_frame}, 上次摔倒帧={getattr(self, 'last_fall_frame', 0)}, 需要等待={fall_cooldown_frames - (self.current_frame - getattr(self, 'last_fall_frame', 0))}帧")
        
        # 如果有告警，更新最后告警帧
        if alerts:
            self.last_alert_frame = self.current_frame
        
        return alerts
    
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
        
        # 绘制告警信息
        alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
        cv2.putText(vis_frame, alert_text, (10, vis_frame.shape[0] - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加告警标识（不再使用全局红色边框）
        cv2.putText(vis_frame, "ALERT DETECTED", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 添加更多的调试信息
        y_offset = 30
        for key, value in self.debug_info.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # 绘制警戒区域
        for region in self.alert_regions:
            cv2.polylines(vis_frame, [region['points']], True, region['color'], region['thickness'])
        
        # 保存图像
        cv2.imwrite(filepath, vis_frame)
        logger.info(f"已保存告警帧: {filepath}")
    
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
        
        # 绘制警戒区域
        for region in self.alert_regions:
            cv2.polylines(vis_frame, [region['points']], True, region['color'], region['thickness'])
            # 添加区域名称
            x, y = region['points'].mean(axis=0).astype(int)
            cv2.putText(vis_frame, region['name'], (x, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, region['color'], 1)
        
        # 智能告警可视化
        if alerts:
            # 显示告警文本（顶部）
            for i, alert in enumerate(alerts):
                alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (10, 10 + i*35), (10 + text_size[0] + 10, 10 + i*35 + text_size[1] + 10), 
                            (0, 0, 0), -1)
                cv2.putText(vis_frame, alert_text, (15, 10 + i*35 + text_size[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 优化：使用告警对象跟踪来精确显示红框
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cls = det['class']
                conf = det['confidence']
                
                # 检查该对象是否处于告警状态
                is_alerted = self.is_object_alerted(det)
                # 新增：如果当前有大面积运动告警，且是person，也标红
                if not is_alerted and alerts:
                    if any(alert.get('type') == self.DANGER_TYPES['large_area_motion'] for alert in alerts):
                        if str(cls).lower() == 'person':
                            is_alerted = True
                
                if str(cls).lower() == 'person':
                    # 只有处于告警状态的person才显示红框
                    if is_alerted:
                        color = (0, 0, 255)  # 红色 - 告警状态
                        thickness = 3
                        # 添加告警标识
                        cv2.putText(vis_frame, "ALERT", (x1, y1 - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        color = (0, 255, 0)  # 绿色 - 正常状态
                        thickness = 2
                else:
                    color = (0, 255, 0)  # 绿色 - 非person对象
                    thickness = 2
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(vis_frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示调试信息
        if show_debug:
            y_offset = 30
            for key, value in self.debug_info.items():
                text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # 显示告警对象统计
            alerted_count = len([obj for obj in self.alerted_objects.values() if obj['class'] == 'person'])
            cv2.putText(vis_frame, f"Alerted Objects: {alerted_count}", (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            
            for i, (alert_type, count) in enumerate(self.alerts_count.items()):
                if count > 0:
                    cv2.putText(vis_frame, f"{alert_type}: {count}", (vis_frame.shape[1] - 180, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 20
        return vis_frame
    
    def get_alert_stats(self):
        """获取告警统计
        
        Returns:
            stats: 告警统计信息
        """
        with self.alert_lock:
            return self.alerts_count.copy()
    
    def reset_stats(self):
        """重置告警统计"""
        with self.alert_lock:
            self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
    
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
            if alert_type == self.DANGER_TYPES['large_area_motion']:
                # 大范围移动：标记所有person对象
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person':
                        self._add_alerted_object(det, alert_type)
            
            elif alert_type == self.DANGER_TYPES['intrusion']:
                # 入侵告警：只标记在警戒区域内的person
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person' and 'bbox' in det:
                        x1, y1, x2, y2 = det['bbox']
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        for region in self.alert_regions:
                            if cv2.pointPolygonTest(region['points'], (center_x, center_y), False) >= 0:
                                self._add_alerted_object(det, alert_type)
                                break
            
            elif alert_type == self.DANGER_TYPES['fall']:
                # 摔倒检测：标记所有person（因为摔倒检测是基于整体运动）
                for det in object_detections:
                    if str(det.get('class', '')).lower() == 'person':
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
    cv2.imshow("Test", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 