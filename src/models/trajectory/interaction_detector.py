import numpy as np
import logging
from typing import Dict, List, Tuple, Set, Any


class InteractionDetector:
    """基于对象轨迹检测对象之间的交互。"""
    
    def __init__(self, distance_threshold: float = 100, count_threshold: int = 2):
        """
        初始化交互检测器。
        
        Args:
            distance_threshold: 对象之间最大距离，用于考虑交互
            count_threshold: 交互确认所需的最小连续帧数
        """
        # 配置日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('InteractionDetector')
        
        self.distance_threshold = distance_threshold
        self.count_threshold = count_threshold
        self.interaction_counts = {}  # (obj_id1, obj_id2) -> 计数
        self.detected_interactions = {}  # (obj_id1, obj_id2) -> 交互细节
    
    def update(self, object_positions):
        """
        更新交互检测器。
        
        Args:
            object_positions: 对象ID到位置（x, y）的字典
            
        Returns:
            检测到的交互的字典
        """
        # 检查所有对象对
        current_interacting_pairs = set()
        
        # 找到接近的对象对
        object_ids = list(object_positions.keys())
        for i, obj_id1 in enumerate(object_ids):
            pos1 = object_positions[obj_id1]
            
            for j in range(i + 1, len(object_ids)):
                obj_id2 = object_ids[j]
                pos2 = object_positions[obj_id2]
                
                # 计算距离
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # 确定对键（总是按较小的ID排序）
                pair = (min(obj_id1, obj_id2), max(obj_id1, obj_id2))
                
                if distance < self.distance_threshold:
                    current_interacting_pairs.add(pair)
                    
                    # 更新计数
                    if pair in self.interaction_counts:
                        self.interaction_counts[pair] += 1
                    else:
                        self.interaction_counts[pair] = 1
                    
                    # 检查是否达到交互阈值
                    if self.interaction_counts[pair] >= self.count_threshold:
                        self.detected_interactions[pair] = {
                            'distance': distance,
                            'count': self.interaction_counts[pair],
                            'position1': pos1,
                            'position2': pos2
                        }
                else:
                    # 如果对象距离较远，则减少计数
                    if pair in self.interaction_counts:
                        self.interaction_counts[pair] -= 1
                        if self.interaction_counts[pair] <= 0:
                            self.interaction_counts.pop(pair, None)
                            self.detected_interactions.pop(pair, None)
        
        # 移除当前未检测到的对
        for pair in list(self.interaction_counts.keys()):
            if pair not in current_interacting_pairs:
                self.interaction_counts[pair] -= 1
                if self.interaction_counts[pair] <= 0:
                    self.interaction_counts.pop(pair, None)
                    self.detected_interactions.pop(pair, None)
        
        return self.detected_interactions
    
    def get_interactions(self):
        """
        获取当前检测到的交互。
        
        Returns:
            检测到的交互的字典
        """
        return self.detected_interactions
    
    def detect_gatherings(self, object_positions):
        """
        检测接近的对象组。
        
        Args:
            object_positions: 对象ID到位置（x, y）的字典
            
        Returns:
            对象ID列表的列表
        """
        groups = []
        
        # 如果对象数量不足，则跳过
        if len(object_positions) < self.count_threshold:
            return groups
        
        # 构建邻接图
        adjacency = {obj_id: set() for obj_id in object_positions}
        
        # 找到接近的对象对
        object_ids = list(object_positions.keys())
        for i, obj_id1 in enumerate(object_ids):
            pos1 = object_positions[obj_id1]
            
            for j in range(i + 1, len(object_ids)):
                obj_id2 = object_ids[j]
                pos2 = object_positions[obj_id2]
                
                # 计算距离
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < self.distance_threshold:
                    adjacency[obj_id1].add(obj_id2)
                    adjacency[obj_id2].add(obj_id1)
        
        # 找到连通分量
        visited = set()
        
        for obj_id in object_positions:
            if obj_id not in visited:
                # 开始一个新的组
                group = set()
                queue = [obj_id]
                
                while queue:
                    current = queue.pop(0)
                    
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        
                        # 将邻居添加到队列
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                if len(group) >= self.count_threshold:
                    groups.append(group)
        
        return groups
    
    def calculate_group_positions(self, groups, object_positions):
        """
        计算每个组的平均位置。
        
        参数:
            groups: 对象ID集合的列表
            object_positions: 对象ID到位置(x, y)的字典映射
            
        返回:
            每个组的平均位置(x, y)列表
        """
        group_positions = []
        
        for group in groups:
            positions = []
            for obj_id in group:
                if obj_id in object_positions:
                    positions.append(object_positions[obj_id])
            
            if positions:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                group_positions.append((avg_x, avg_y))
            else:
                group_positions.append((0, 0))  # Fallback
        
        return group_positions
    
    def reset(self):
        """重置交互检测器。"""
        self.interaction_counts = {}
        self.detected_interactions = {} 