import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from models.motion.motion_features import MotionFeature

from models.trajectory.trajectory_manager import TrajectoryManager

from models.behavior.behavior_types import BehaviorType, Behavior, Interaction


class BehaviorRecognizer:
    """
    行为识别器，基于轨迹和运动特征识别行为和交互。
    """
    
    def __init__(self, speed_threshold=10.0, interaction_threshold=50.0):
        """
        初始化行为识别器。
        
        Args:
            speed_threshold: 速度阈值，用于判断运动行为
            interaction_threshold: 交互检测阈值，用于判断多个对象之间的交互
        """
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BehaviorRecognizer')
        
        self.speed_threshold = speed_threshold
        self.interaction_threshold = interaction_threshold
        
        self.is_initialized = True
        self.logger.info("Behavior recognizer initialized")
    
    def analyze(self, trajectories, motion_features, interaction_detector):
        """
        分析轨迹和运动特征以识别行为和交互。
        
        Args:
            trajectories: 轨迹管理器中的轨迹列表
            motion_features: 运动管理器中的运动特征列表
            interaction_detector: 交互检测器实例
            
        Returns:
            Tuple of (行为, 交互)
        """
        # 如果轨迹列表为空，则返回空列表
        if not trajectories:
            return [], []
        
        # 分析单个行为
        behaviors = self.analyze_trajectories(trajectories, motion_features)
        
        # 检测对象之间的交互
        interactions = self.detect_interactions(trajectories, interaction_detector)
        
        return behaviors, interactions
    
    def analyze_trajectories(self, trajectories, motion_features):
        """
        分析单个轨迹以识别行为。
        
        Args:
            trajectories: 轨迹列表
            motion_features: 运动特征列表
            
        Returns:
            行为结果列表
        """
        behaviors = [] # 存储行为结果列表
        
        for trajectory in trajectories:
            # 如果轨迹没有足够的数据，则跳过
            if 'speeds' not in trajectory or not trajectory['speeds']:
                continue
            
            obj_id = trajectory['id'] # 获取轨迹的id
            
            # 获取最新位置
            position = trajectory['position']
            
            # 计算平均速度
            avg_speed = np.mean(trajectory['speeds'][-min(10, len(trajectory['speeds'])):])
            
            # 计算该对象的运动特征
            obj_motion_features = []
            for feature in motion_features:
                if hasattr(feature, 'object_id') and feature.object_id == obj_id:
                    obj_motion_features.append(feature)
            
            # 根据速度和轨迹特征确定行为
            behavior_type = BehaviorType.UNKNOWN
            confidence = 0.5
            # 判断是否静止
            if trajectory['is_stationary']:
                behavior_type = BehaviorType.STATIC
                confidence = 0.9
            # 判断是否行走
            elif avg_speed < self.speed_threshold / 2: # 速度小于阈值的一半
                behavior_type = BehaviorType.WALKING
                confidence = 0.8
            # 判断是否奔跑
            elif avg_speed >= self.speed_threshold: # 速度大于等于阈值
                behavior_type = BehaviorType.RUNNING
                confidence = 0.8
            
            # 检查徘徊行为
            if trajectory['is_stationary'] and trajectory['area_covered'] < 2000:
                behavior_type = BehaviorType.LOITERING
                confidence = 0.7
            
            # 检查游荡行为
            if trajectory['direction_changes'] > 3 and avg_speed < self.speed_threshold:
                behavior_type = BehaviorType.WANDERING
                confidence = 0.7
            
            # --- 新增打架检测 ---
        # 检查所有pair，若两人距离很近且速度都较大，判为打架
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                traj1 = trajectories[i]
                traj2 = trajectories[j]
                if 'position' in traj1 and 'position' in traj2 and 'speeds' in traj1 and 'speeds' in traj2:
                    pos1 = traj1['position']
                    pos2 = traj2['position']
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    speed1 = np.mean(traj1['speeds'][-min(5, len(traj1['speeds'])):])
                    speed2 = np.mean(traj2['speeds'][-min(5, len(traj2['speeds'])):])
                    if dist < 60 and speed1 > self.speed_threshold and speed2 > self.speed_threshold:
                        # 判为打架
                        behavior1 = Behavior(
                            object_id=traj1['id'],
                            behavior_type=BehaviorType.FIGHTING,
                            confidence=0.8,
                            position=traj1['position'],
                            timestamp=traj1['timestamps'][-1] if traj1['timestamps'] else None,
                            metadata={'class_name': traj1.get('class_name', 'unknown'), 'speed': speed1, 'fighting_with': traj2['id']}
                        )
                        behavior2 = Behavior(
                            object_id=traj2['id'],
                            behavior_type=BehaviorType.FIGHTING,
                            confidence=0.8,
                            position=traj2['position'],
                            timestamp=traj2['timestamps'][-1] if traj2['timestamps'] else None,
                            metadata={'class_name': traj2.get('class_name', 'unknown'), 'speed': speed2, 'fighting_with': traj1['id']}
                        )
                        behaviors.append(behavior1)
                        behaviors.append(behavior2)
        # --- 打架检测结束 ---
            # 创建行为对象
            behavior = Behavior(
                object_id=obj_id,
                behavior_type=behavior_type,
                confidence=confidence,
                position=position,
                timestamp=trajectory['timestamps'][-1] if trajectory['timestamps'] else None,
                metadata={
                    'class_name': trajectory.get('class_name', 'unknown'),
                    'speed': avg_speed,
                    'direction_changes': trajectory.get('direction_changes', 0),
                    'is_stationary': trajectory.get('is_stationary', False),
                    'area_covered': trajectory.get('area_covered', 0)
                }
            )
            
            behaviors.append(behavior)
        
        return behaviors
    
    def detect_interactions(self, trajectories, interaction_detector):
        """
        检测对象之间的交互。
        
        Args:
            trajectories: 轨迹列表
            interaction_detector: 交互检测器实例
            
        Returns:
            交互结果列表
        """
        interactions = []
        
        # 如果对象数量不足，则跳过
        if len(trajectories) < 2:
            return interactions
        
        # 从检测器获取当前交互
        detector_interactions = interaction_detector.get_interactions()
        
        for interaction_pair, details in detector_interactions.items():
            obj_id1, obj_id2 = interaction_pair
            
            # 获取涉及的对象的轨迹
            traj1 = next((t for t in trajectories if t['id'] == obj_id1), None)
            traj2 = next((t for t in trajectories if t['id'] == obj_id2), None)
            
            if traj1 is None or traj2 is None:
                continue
            
            # 根据轨迹的性质确定交互类型
            interaction_type = BehaviorType.GATHERING  # Default
            confidence = 0.6
            
            # 检查两个对象是否静止
            if traj1.get('is_stationary', False) and traj2.get('is_stationary', False):
                interaction_type = BehaviorType.GATHERING
                confidence = 0.8
            
            # 检查两个对象是否快速移动
            elif (np.mean(traj1.get('speeds', [0])[-5:]) > self.speed_threshold and 
                  np.mean(traj2.get('speeds', [0])[-5:]) > self.speed_threshold):
                interaction_type = BehaviorType.FOLLOWING
                confidence = 0.7
            
            # 计算两个对象之间的中点
            pos1 = traj1['position']
            pos2 = traj2['position']
            mid_point = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
            
            # 创建交互对象
            interaction = Interaction(
                object_ids=[obj_id1, obj_id2],
                behavior_type=interaction_type,
                confidence=confidence,
                position=mid_point,
                timestamp=max(traj1['timestamps'][-1], traj2['timestamps'][-1]) if traj1['timestamps'] and traj2['timestamps'] else None,
                metadata={
                    'distance': details.get('distance', 0),
                    'duration': details.get('count', 0),
                    'class1': traj1.get('class_name', 'unknown'),
                    'class2': traj2.get('class_name', 'unknown')
                }
            )
            
            interactions.append(interaction)
        
        return interactions
    
    def visualize_behaviors(self, frame, behaviors, interactions):
        """
        在帧上可视化行为和交互。
        
        Args:
            frame: 要可视化的帧
            behaviors: 行为结果列表
            interactions: 交互结果列表
            
        Returns:
            可视化帧
        """
        if frame is None:
            return None
        
        vis_frame = frame.copy()
        
        # 定义每种行为类型的颜色
        colors = {
            BehaviorType.STATIC: (200, 200, 200),     # Gray
            BehaviorType.WALKING: (0, 255, 0),        # Green
            BehaviorType.RUNNING: (0, 165, 255),      # Orange
            BehaviorType.FALLING: (0, 0, 255),        # Red
            BehaviorType.FIGHTING: (255, 0, 0),       # Blue
            BehaviorType.GATHERING: (255, 0, 255),    # Purple
            BehaviorType.WANDERING: (255, 255, 0),    # Cyan
            BehaviorType.LOITERING: (128, 0, 255),    # Purple
            BehaviorType.FOLLOWING: (255, 128, 0),    # Sky Blue
            BehaviorType.UNKNOWN: (200, 200, 200)     # Gray
        }
        
        # 绘制行为
        for behavior in behaviors:
            if not hasattr(behavior, 'position') or not behavior.position:
                continue
                
            x, y = behavior.position
            color = colors.get(behavior.behavior_type, (200, 200, 200))
            
            # 绘制行为类型和置信度
            label = f"{behavior.behavior_type.name}: {behavior.confidence:.2f}"
            cv2.putText(vis_frame, label, (int(x), int(y) - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制交互
        for interaction in interactions:
            if not hasattr(interaction, 'position') or not interaction.position:
                continue
                
            x, y = interaction.position
            color = colors.get(interaction.behavior_type, (200, 200, 200))
            
            # 绘制两个交互对象之间的线
            if hasattr(interaction, 'object_ids') and len(interaction.object_ids) >= 2:
                obj_id1, obj_id2 = interaction.object_ids[:2]
                
                # 绘制交互类型
                label = f"{interaction.behavior_type.name}: {interaction.confidence:.2f}"
                cv2.putText(vis_frame, label, (int(x), int(y) - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 绘制中点处的圆
                cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
        
        return vis_frame 
