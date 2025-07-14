import cv2
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class MotionFeature:
    """运动特征表示的数据类。"""
    type: str                # 特征类型
    data: np.ndarray         # 特征数据，例如：光流向量、特征点坐标等
    position: Tuple[int, int] # 特征位置 (x, y)
    frame_idx: int           # 帧索引
    object_id: Optional[int] = None  # 关联的目标ID (如果可用)
    confidence: float = 1.0  # 特征置信度


class MotionFeatureExtractor:
    """运动特征提取器的基类。"""
    
    def __init__(self):
        """初始化运动特征提取器。"""
        self.is_initialized = False
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        # 检查OpenCV版本，因为某些方法可能依赖于版本
        opencv_version = cv2.__version__
        self.logger.info(f"OpenCV version: {opencv_version}")
        
        # 子类应在成功初始化后设置此值为True
        # self.is_initialized 仅在初始化成功时为True
    
    def extract(self, frame: np.ndarray, prev_frame: np.ndarray = None, **kwargs) -> List[MotionFeature]:
        """
        从连续帧中提取运动特征。
        
        Args:
            frame: 当前帧
            prev_frame: 上一帧（可选）
            **kwargs: 额外参数
            
        Returns:
            运动特征列表
        """
        if not self.is_initialized:
            self.logger.error(f"{self.__class__.__name__} not properly initialized")
            return []
            
        try:
            # 由子类实现
            return self._extract_impl(frame, prev_frame, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}.extract: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []
    
    def _extract_impl(self, frame: np.ndarray, prev_frame: np.ndarray = None, **kwargs) -> List[MotionFeature]:
        """
        特征提取的实现，由子类重写。
        
        Args:
            frame: 当前帧
            prev_frame: 上一帧（可选）
            **kwargs: 额外参数
            
        Returns:
            运动特征列表
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement _extract_impl method") 
    
    def reset(self):
        """重置提取器状态。"""
        # 由子类实现，如果需要
        pass 