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
        'sudden_motion': '突然剧烈运动',
        'large_area_motion': '大范围异常运动',
        'fall': '可能摔倒',
        'abnormal_pattern': '异常移动模式',
        'intrusion': '入侵警告区域',
        'loitering': '可疑徘徊',
        'violent_motion': '打架',
    }
    
    def __init__(self, config=None):
        """初始化危险行为识别器
        
        Args:
            config: 配置字典，包含检测参数和阈值
        """
        # 默认配置
        self.config = {
            'feature_count_threshold': 100,  # 特征点数量阈值
            'feature_change_ratio': 1.5,     # 特征变化比率阈值
            'motion_magnitude_threshold': 10, # 运动幅度阈值
            'motion_area_threshold': 0.05,   # 运动区域阈值(占比)
            'fall_motion_threshold': 10,     # 摔倒检测阈值
            'alert_cooldown': 10,            # 告警冷却帧数
            'history_length': 30,            # 历史帧数量
            'save_alerts': True,             # 是否保存告警
            'alert_dir': 'alerts',           # 告警保存目录
            'min_confidence': 0.5,           # 最小置信度
        }
        
        # 更新用户配置
        if config:
            self.config.update(config)
        
        # 创建告警保存目录
        if self.config['save_alerts']:
            os.makedirs(self.config['alert_dir'], exist_ok=True)
        
        # 初始化状态
        self.currently_alerting = False
        self.safe_frames = 0
        self.safe_reset = 5  # 连续 5 帧安全才算真正结束

        self.history = []
        self.current_frame = 0
        self.last_alert_frame = 0
        self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
        self.last_features_count = 0
        self.debug_info = {}
        self.alert_lock = Lock()
        
        # 添加用于ROI区域的属性
        self.alert_regions = []  # 告警区域列表
        self.config['violent_mag_threshold'] = 8

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

    # === DangerRecognizer 补丁 begin ==================================
    def process_frame(self, frame, features, object_detections=None):
        """
        每帧调用：把上一模块给的运动特征 → 危险告警
        兼容 features 为 字典 / 对象列表 两种格式
        """
        self.current_frame += 1

        # ---------- 1. 解析特征 ----------
        if isinstance(features, dict):  # 新版字典格式
            feature_count = features.get('keypoints_count', 0)
            motion_area = features.get('motion_area', 0)
            avg_mag = features.get('avg_magnitude', 0)
        else:  # 旧版对象列表
            feature_count = len(features) if features else 0
            stats = self._extract_motion_stats(
                features, (frame.shape[1], frame.shape[0]))
            motion_area = stats['motion_area']
            avg_mag = stats['avg_magnitude']

        # ---------- 2. 更新 debug 信息 ----------
        self.debug_info = {
            'frame': self.current_frame,
            'feature_count': feature_count,
            'motion_area': motion_area,
            'avg_magnitude': avg_mag,
        }

        # ---------- 3. 写入历史 ----------
        self.history.append({
            'avg_magnitude': avg_mag,
            'motion_area': motion_area,
            'vertical_motion': 0  # 若需要垂直位移可在此填入
        })
        if len(self.history) > self.config['history_length']:
            self.history.pop(0)

        # ---------- 4. 真正的危险分析 ----------
        alerts = self._analyze_danger(
            frame, features, object_detections, feature_count, motion_area)

        # ★★★ 在此统一补充时间戳，确保前端显示的是“发生时刻”而非刷新时刻
        current_ts = time.time()  # 秒级 UNIX 时间
        for a in alerts:
            a['timestamp'] = current_ts

        # ---------- 5. 统计 & 保存告警帧（首帧才执行） ----------
        if alerts:
            with self.alert_lock:
                for a in alerts:
                    self.alerts_count[a['type']] += 1
                if self.config['save_alerts'] and frame is not None:
                    self._save_alert_frame(frame, alerts[0])

        self.last_features_count = feature_count
        return alerts

    # === DangerRecognizer 补丁 end ====================================

    def _extract_motion_stats(self, features, frame_size):
        """从特征中提取运动统计
        
        Args:
            features: 特征列表
            frame_size: 帧大小 (width, height)
            
        Returns:
            stats: 统计信息字典
        """
        frame_area = frame_size[0] * frame_size[1]
        '''stats = {
            'timestamp': time.time(),
            'avg_magnitude': 0,
            'max_magnitude': 0,
            'motion_directions': {},
            'motion_area': 0,
            'vertical_motion': 0,
            'feature_count': len(features) if features else 0,
        }'''
        if isinstance(features, dict):  # 字典格式
            stats = {
                'timestamp': time.time(),
                'avg_magnitude': features.get('avg_magnitude', 0),
                'max_magnitude': features.get('max_magnitude', 0),
                'motion_directions': {},  # 如需方向可自行追加
                'motion_area': features.get('motion_area', 0),
                'vertical_motion': 0,
                'feature_count': features.get('keypoints_count', 0),
            }
            # 这里直接把 motion_area 等字段取出来即可
            return stats  # 字典模式直接返回，不再走下面的“列表循环”
        else:  # 旧的“列表特征”逻辑保持
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
        
        # 计算运动统计
        magnitudes = []
        motion_area = 0
        vertical_motion = 0
        
        for feature in features:
            if hasattr(feature, 'magnitude'):
                magnitudes.append(feature.magnitude)
            
            if hasattr(feature, 'position') and hasattr(feature, 'end_position'):
                # 计算方向
                dx = feature.end_position[0] - feature.position[0]
                dy = feature.end_position[1] - feature.position[1]
                
                vertical_motion += dy
                
                # 量化方向
                angle = np.degrees(np.arctan2(dy, dx))
                direction = round(angle / 45) * 45
                if direction in stats['motion_directions']:
                    stats['motion_directions'][direction] += 1
                else:
                    stats['motion_directions'][direction] = 1
                
                # 检查是否在告警区域内
                if self.alert_regions and hasattr(feature, 'position'):
                    for region in self.alert_regions:
                        if cv2.pointPolygonTest(region['points'], 
                                               (int(feature.position[0]), int(feature.position[1])), 
                                               False) >= 0:
                            stats['in_alert_region'] = True
                            break
            
            # 估算运动区域
            if hasattr(feature, 'magnitude'):
                motion_area += feature.magnitude * 10
        
        if magnitudes:
            stats['avg_magnitude'] = sum(magnitudes) / len(magnitudes)
            stats['max_magnitude'] = max(magnitudes)
        
        stats['motion_area'] = min(1.0, motion_area / frame_area)
        stats['vertical_motion'] = vertical_motion / max(1, len(features))
        
        return stats

    def _analyze_danger(self, frame, features, object_detections,
                        feature_count, motion_area):
        """
        只在“进入危险期的首帧”返回 alerts；
        连续危险帧 → 仅维持 currently_alerting 标志，不重复计数。
        """
        alerts = []

        # ───── ① 基本阈值：特征点数 & 大面积运动 ─────
        if feature_count > self.config['feature_count_threshold']:
            alerts.append({
                'type': self.DANGER_TYPES['sudden_motion'],
                'confidence': min(1.0,
                                  feature_count / (self.config['feature_count_threshold'] * 2))
            })

        if motion_area > self.config['motion_area_threshold']:
            alerts.append({
                'type': self.DANGER_TYPES['large_area_motion'],
                'confidence': min(1.0,
                                  motion_area / self.config['motion_area_threshold'])
            })

        # ───── ② 打架 / 剧烈运动（规则版）─────
        person_cnt = sum(1 for d in (object_detections or []) if d['class'] == 'person')
        avg_mag = features.get('avg_magnitude', 0.0)  # 在 MotionFeatureManager 中统计

        if person_cnt >= 2 and avg_mag > self.config['violent_mag_threshold']:
            alerts.append({
                'type': self.DANGER_TYPES['violent_motion'],
                'confidence': min(1.0, avg_mag /
                                  (self.config['violent_mag_threshold'] * 2)),
                'person_cnt': person_cnt,
                'avg_mag': avg_mag,
            })

        # ───── ③ 连续动作只计一次 ─────
        danger_now = bool(alerts)

        if danger_now and not self.currently_alerting:
            # 首帧进入危险期 → 允许上层计数
            self.currently_alerting = True
            self.safe_frames = 0
            return alerts

        elif danger_now and self.currently_alerting:
            # 已在危险期 → 本帧不计数
            return []

        else:
            # 安全帧处理
            if self.currently_alerting:
                self.safe_frames += 1
                if self.safe_frames >= self.safe_reset:
                    # 连续安全 enough 帧，结束本次告警期
                    self.currently_alerting = False
                    self.safe_frames = 0
            return []

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
        
        # 添加红色边框
        cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], vis_frame.shape[0]), (0, 0, 255), 3)
        
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

    def visualize(self, frame, alerts=None, features=None, show_debug=True):
        """
        可视化危险检测结果
        - alerts 非空  → 首帧，画红框+详细文字
        - currently_alerting 为 True 且 alerts 空 → 连续帧，保留红框+简洁 'ALERT'
        """
        if frame is None:
            return None

        vis_frame = frame.copy()

        # ---------- ROI ----------
        for region in self.alert_regions:
            cv2.polylines(vis_frame, [region['points']], True,
                          region['color'], region['thickness'])
            cx, cy = region['points'].mean(axis=0).astype(int)
            cv2.putText(vis_frame, region['name'], (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, region['color'], 1)

        # ---------- 红框 ----------
        #if alerts or self.currently_alerting:
            #cv2.rectangle(vis_frame, (0, 0),
                          #(vis_frame.shape[1] - 1, vis_frame.shape[0] - 1),
                          #(0, 0, 255), 3)

        # ---------- 告警文字 ----------
        if alerts:  # 首帧
            for i, alert in enumerate(alerts):
                txt = f"{alert['type']} ({alert['confidence']:.2f})"
                cv2.putText(vis_frame, txt,
                            (10, vis_frame.shape[0] - 30 - i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.currently_alerting:  # 连续帧
            cv2.putText(vis_frame, "ALERT",
                        (10, vis_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---------- 调试 ----------
        if show_debug:
            y = 30
            for k, v in self.debug_info.items():
                txt = f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                cv2.putText(vis_frame, txt, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 20
            y = 30
            for t, c in self.alerts_count.items():
                if c > 0:
                    cv2.putText(vis_frame, f"{t}: {c}",
                                (vis_frame.shape[1] - 180, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y += 20

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
            self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
            self.last_features_count = 0
            logger.info("危险行为识别器已重置")

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