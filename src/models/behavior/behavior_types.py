from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
'''
行为分析系统的基础定义文件，它定义了系统能够识别的所有行为类型以及相关的数据结构。
主要包含两个核心类：BehaviorAnalysisResult（用于存储单个行为的分析结果，包括行为类型、置信度、时间戳等）
和 Interaction（用于描述多个对象之间的交互行为，如打架、聚集等）。
这个文件就像是行为分析系统的"词典"，定义了系统要识别的行为类型和数据结构
其他模块都需要基于这个文件的定义来进行行为识别和分析。
'''

class BehaviorType(Enum):
    """行为类型枚举。"""
    STATIC = 0          # 静止
    WALKING = 1         # 行走
    RUNNING = 2         # 奔跑
    FALLING = 3         # 跌倒
    FIGHTING = 4        # 打架
    GATHERING = 5       # 聚集
    WANDERING = 6       # 徘徊
    LOITERING = 7       # 徘徊/停留
    FOLLOWING = 8       # 跟随
    UNKNOWN = 9         # 未知行为


@dataclass
class BehaviorAnalysisResult:
    """行为分析结果数据类。"""
    behavior_type: BehaviorType    # 行为类型
    confidence: float              # 置信度
    object_id: Optional[int] = None  # 关联的目标ID (如果可用)
    frame_idx: int = 0            # 帧索引
    data: Dict[str, Any] = None   # 附加数据
    timestamp: str = ""           # 时间戳
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "behavior_type": self.behavior_type.name,
            "confidence": self.confidence,
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "data": self.data,
            "timestamp": self.timestamp
        }


@dataclass
class Behavior:
    """单个对象的行为数据类。"""
    object_id: int                     # 目标ID
    behavior_type: BehaviorType        # 行为类型
    confidence: float                  # 置信度
    position: Optional[Tuple[float, float]] = None  # 位置坐标 (x, y)
    timestamp: Optional[float] = None  # 时间戳
    metadata: Dict[str, Any] = None    # 元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "object_id": self.object_id,
            "behavior_type": self.behavior_type.name,
            "confidence": self.confidence,
            "position": self.position,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class Interaction:
    """多个对象之间的交互数据类。"""
    object_ids: List[int]              # 交互的目标ID列表
    behavior_type: BehaviorType        # 交互类型
    confidence: float                  # 置信度
    position: Optional[Tuple[float, float]] = None  # 交互中心点
    timestamp: Optional[float] = None  # 时间戳
    metadata: Dict[str, Any] = None    # 元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "object_ids": self.object_ids,
            "behavior_type": self.behavior_type.name,
            "confidence": self.confidence,
            "position": self.position,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        } 