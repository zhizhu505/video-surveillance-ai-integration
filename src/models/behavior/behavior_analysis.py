import cv2
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json

from models.motion.motion_features import MotionFeatureManager
from models.behavior.behavior_recognition import BehaviorRecognitionSystem, BehaviorAnalysisResult, BehaviorType
from models.trajectory.trajectory import ObjectTrajectory
from utils.motion_utils import (
    create_motion_heatmap,
    apply_colormap,
    blend_heatmap_with_frame,
    create_motion_field_visualization,
    visualize_trajectory,
    save_motion_features,
    save_behavior_results,
    plot_trajectory_features,
    extract_relevant_frames
)


class BehaviorAnalysisSystem:
    """
    集成系统,用于运动特征提取、行为识别和可视化。
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 use_optical_flow: bool = True, use_motion_history: bool = True,
                 max_trajectory_history: int = 60):
        """
        初始化行为分析系统。
        
        参数:
            frame_width: 帧宽度
            frame_height: 帧高度 
            use_optical_flow: 是否使用光流特征提取器
            use_motion_history: 是否使用运动历史特征提取器
            max_trajectory_history: 轨迹历史记录保留的最大帧数
        """
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BehaviorAnalysisSystem')
        
        # 初始化帧维度
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # 初始化运动特征管理器
        self.motion_feature_manager = MotionFeatureManager(
            use_optical_flow=use_optical_flow,
            use_motion_history=use_motion_history
        )
        
        # 初始化行为识别系统
        self.behavior_recognition = BehaviorRecognitionSystem(
            frame_width=frame_width,
            frame_height=frame_height,
            max_trajectory_history=max_trajectory_history
        )
        
        # 初始化历史
        self.motion_features_history = []
        self.behavior_results_history = []
        
        # 最大历史长度
        self.max_history_length = 300
        
        # 统计
        self.frame_count = 0
        self.processing_times = []
        
        # 初始化状态
        self.is_initialized = (
            self.motion_feature_manager.is_initialized and 
            self.behavior_recognition.is_initialized
        )
        
        # 如果初始化成功，则打印日志
        if self.is_initialized:
            self.logger.info("Behavior analysis system initialized")
    
    def process_frame(self, frame: np.ndarray, tracks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理一帧，提取运动特征并分析行为。
        
        Args:
            frame: 输入帧
            tracks: 对象跟踪器中的对象轨迹（可选）
            
        Returns:
            结果字典:
            - 'motion_features': 运动特征列表
            - 'behavior_results': 行为分析结果列表
            - 'processing_time': 处理时间（秒）
        """
        # 如果未初始化或帧为空，则返回空结果
        if not self.is_initialized or frame is None:
            return {'motion_features': [], 'behavior_results': [], 'processing_time': 0}
        
        start_time = cv2.getTickCount()
        
        # 第一步: 提取运动特征
        motion_features = self.motion_feature_manager.extract_features(frame, tracks)
        
        # 第二步: 分析行为
        behavior_results = self.behavior_recognition.update(tracks or [], motion_features)
        
        # 计算处理时间
        processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        # 更新历史
        self.motion_features_history.append(motion_features)
        self.behavior_results_history.append(behavior_results)
        
        # 如果需要，修剪历史
        if len(self.motion_features_history) > self.max_history_length:
            self.motion_features_history.pop(0)
        if len(self.behavior_results_history) > self.max_history_length:
            self.behavior_results_history.pop(0)
        
        # 更新统计
        self.frame_count += 1
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return {
            'motion_features': motion_features,
            'behavior_results': behavior_results,
            'processing_time': processing_time
        }
    
    def reset(self):
        """重置行为分析系统。"""
        self.motion_feature_manager.reset()
        self.behavior_recognition.reset()
        self.motion_features_history = []
        self.behavior_results_history = []
        self.frame_count = 0
        self.processing_times = []
    
    def visualize_motion_features(self, frame: np.ndarray, motion_features: List[Any] = None) -> np.ndarray:
        """
        在帧上可视化运动特征。
        
        Args:
            frame: 要绘制的帧
            motion_features: 运动特征列表（如果为None，则使用最新）
            
        Returns:
            带有可视化运动特征的帧
        """
        if frame is None:
            return frame
        
        # 如果未提供，则使用最新特征
        if motion_features is None:
            if not self.motion_features_history:
                return frame
            motion_features = self.motion_features_history[-1]
        
        return self.motion_feature_manager.visualize_features(frame, motion_features)
    
    def visualize_behaviors(self, frame: np.ndarray, behavior_results: List[BehaviorAnalysisResult] = None) -> np.ndarray:
        """
        在帧上可视化行为分析结果。
        
        Args:
            frame: 要绘制的帧
            behavior_results: 行为分析结果（如果为None，则使用最新）
            
        Returns:
            带有可视化行为的帧
        """
        if frame is None:
            return frame
        
        # 如果未提供，则使用最新结果
        if behavior_results is None:
            if not self.behavior_results_history:
                return frame
            behavior_results = self.behavior_results_history[-1]
        
        return self.behavior_recognition.visualize_behaviors(frame, behavior_results)
    
    def create_motion_heatmap(self, frame_shape: Tuple[int, int] = None) -> np.ndarray:
        """
        从最近的运动特征创建运动热图。
        
        Args:
            frame_shape: 帧的形状，如果为None，则使用初始化维度
            
        Returns:
            彩色热图作为BGR图像
        """
        if not self.motion_features_history:
            return None
        
        # 从最近的30帧中获取所有运动特征
        all_features = []
        for features in self.motion_features_history[-30:]:
            all_features.extend(features)
        
        # 如果未提供，则使用初始化维度
        if frame_shape is None:
            frame_shape = (self.frame_height, self.frame_width)
        
        # 创建热图
        heatmap = create_motion_heatmap(all_features, frame_shape)
        colored_heatmap = apply_colormap(heatmap)
        
        return colored_heatmap
    
    def visualize_combined_results(self, frame: np.ndarray, include_motion: bool = True, 
                                   include_behavior: bool = True, include_heatmap: bool = False,
                                   heatmap_alpha: float = 0.3) -> np.ndarray:
        """
        创建运动特征和行为分析的组合可视化。
        
        Args:
            frame: 输入帧
            include_motion: 是否包含运动特征可视化
            include_behavior: 是否包含行为可视化
            include_heatmap: 是否包含运动热图
            heatmap_alpha: 热图混合的Alpha值
            
        Returns:
            带有组合可视化的帧
        """
        if frame is None:
            return frame
        
        result = frame.copy()
        
        # 添加运动特征
        if include_motion and self.motion_features_history:
            result = self.visualize_motion_features(result)
        
        # 添加运动热图
        if include_heatmap:
            heatmap = self.create_motion_heatmap(frame.shape[:2])
            if heatmap is not None:
                result = blend_heatmap_with_frame(result, heatmap, heatmap_alpha)
        
        # 添加行为可视化
        if include_behavior and self.behavior_results_history:
            result = self.visualize_behaviors(result)
        
        # 添加处理统计
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            # 在画面左上角显示FPS和当前帧数
            cv2.putText(
                result,
                f"FPS: {fps:.1f} | Frame: {self.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return result
    
    def get_trajectory(self, object_id: int) -> Optional[ObjectTrajectory]:
        """
        获取特定对象的轨迹。
        
        Args:
            object_id: 对象ID
            
        Returns:
            对象轨迹或None（如果未找到）
        """
        return self.behavior_recognition.get_trajectory(object_id)
    
    def visualize_trajectory(self, frame: np.ndarray, object_id: int) -> np.ndarray:
        """
        为特定对象可视化轨迹。
        
        Args:
            frame: 要绘制的帧
            object_id: 对象ID
            
        Returns:
            带有可视化轨迹的帧
        """
        trajectory = self.get_trajectory(object_id)
        
        if frame is None or trajectory is None or not trajectory.positions:
            return frame
        
        return visualize_trajectory(frame, trajectory.positions)
    
    def save_analysis_results(self, output_dir: str, prefix: str = "analysis"):
        """
        保存运动特征和行为分析结果到文件。
        
        Args:
            output_dir: 输出目录
            prefix: 输出文件前缀
        """
        # 如果输出目录不存在，则创建
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存运动特征
        if self.motion_features_history:
            all_features = []
            for features in self.motion_features_history:
                all_features.extend(features)
            
            motion_filename = os.path.join(output_dir, f"{prefix}_motion_features.json")
            save_motion_features(all_features, motion_filename)
        
        # 保存行为分析结果
        if self.behavior_results_history:
            all_results = []
            for results in self.behavior_results_history:
                all_results.extend(results)
            
            behavior_filename = os.path.join(output_dir, f"{prefix}_behavior_results.json")
            save_behavior_results(all_results, behavior_filename)
        
        # 保存统计信息
        stats = {
            'frame_count': self.frame_count,
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'fps': 1.0 / (sum(self.processing_times) / len(self.processing_times)) if self.processing_times else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats_filename = os.path.join(output_dir, f"{prefix}_statistics.json")
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def generate_behavior_summary(self) -> Dict[str, Any]:
        """
        生成行为检测的统计摘要。
        
        Returns:
            包含行为统计的字典
        """
        if not self.behavior_results_history:
            return {'behaviors': {}, 'total_count': 0}
        
        # 扁平化所有行为结果
        all_results = []
        for results in self.behavior_results_history:
            all_results.extend(results)
        
        # 按类型计数行为
        behavior_counts = {}
        for behavior_type in BehaviorType:
            behavior_counts[behavior_type.name] = 0
        
        for result in all_results:
            behavior_counts[result.behavior_type.name] += 1
        
        # 计算其他统计信息
        total_count = len(all_results)
        
        # 按频率排序（降序）
        sorted_behaviors = sorted(
            [(name, count) for name, count in behavior_counts.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'behaviors': {name: {'count': count, 'percentage': (count / total_count) * 100 if total_count > 0 else 0}
                         for name, count in sorted_behaviors},
            'total_count': total_count
        } 
