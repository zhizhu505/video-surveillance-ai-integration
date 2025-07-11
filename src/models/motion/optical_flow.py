import cv2
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any

from models.motion.motion_feature_base import MotionFeature, MotionFeatureExtractor


class OpticalFlowExtractor(MotionFeatureExtractor):
    """从连续帧中提取光流特征。"""
    
    METHODS = {
        'farneback': cv2.calcOpticalFlowFarneback,
        'pyr_lk': None,  # Implemented separately since it uses a different approach
    }
    
    def __init__(self, method: str = 'farneback', pyr_scale: float = 0.5, 
                 levels: int = 3, winsize: int = 15, iterations: int = 3,
                 poly_n: int = 5, poly_sigma: float = 1.2,
                 use_gpu: bool = False):
        """
        初始化光流特征提取器。
        
        Args:
            method: 光流方法（'farneback' 或 'pyr_lk'）
            pyr_scale: 金字塔图像缩放比例（<1表示更大的金字塔）
            levels: 金字塔层数
            winsize: 平均窗口大小
            iterations: 每个金字塔层级的迭代次数
            poly_n: 多项式近似中的像素邻域大小
            poly_sigma: 高斯分布的标准差
            use_gpu: 如果可用，使用GPU加速
        """
        super().__init__()
        
        try:
            self.method = method.lower()
            if self.method not in self.METHODS:
                self.logger.error(f"Unsupported optical flow method: {method}")
                return
            
            # 设置参数
            self.pyr_scale = pyr_scale
            self.levels = levels
            self.winsize = winsize
            self.iterations = iterations
            self.poly_n = poly_n
            self.poly_sigma = poly_sigma
            
            # 设置GPU加速设备
            self.use_gpu = use_gpu
            cuda_available = False
            
            try:
                cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if cuda_available:
                    self.logger.info(f"CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}")
            except Exception as e:
                self.logger.warning(f"Could not check CUDA availability: {str(e)}")
                cuda_available = False
                
            self.use_gpu = use_gpu and cuda_available
                
            if self.use_gpu and self.method == 'farneback':
                try:
                    self.flow_calculator = cv2.cuda.FarnebackOpticalFlow.create(
                        numLevels=self.levels,
                        pyrScale=self.pyr_scale,
                        winSize=self.winsize,
                        numIters=self.iterations,
                        polyN=self.poly_n,
                        polySigma=self.poly_sigma
                    )
                    self.logger.info("Successfully created GPU flow calculator")
                except Exception as e:
                    self.logger.error(f"Failed to create GPU flow calculator: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    self.use_gpu = False
                    
                if self.use_gpu:
                    self.logger.info("Using GPU acceleration for optical flow")
            elif self.use_gpu:
                self.logger.warning(f"GPU acceleration not available for {self.method} method")
                self.use_gpu = False
            
            # 对于Lucas-Kanade方法
            if self.method == 'pyr_lk':
                # ShiTomasi角点检测参数
                self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                
                # Lucas-Kanade光流参数
                self.lk_params = dict(
                    winSize=(self.winsize, self.winsize),
                    maxLevel=self.levels,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # 创建要跟踪的点
                self.prev_pts = None
            
            self.is_initialized = True
            self.prev_gray = None
            self.flow = None  # 初始化光流场
            self.logger.info(f"OpticalFlowExtractor initialized successfully with method: {self.method}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpticalFlowExtractor: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.is_initialized = False
    
    def _extract_impl(self, frame: np.ndarray, prev_frame: np.ndarray = None, 
                tracks: List[Dict[str, Any]] = None) -> List[MotionFeature]:
        """
        从连续帧中提取光流特征。
        
        Args:
            frame: 当前帧
            prev_frame: 上一帧（如果为None，将使用缓存的前一帧）
            tracks: 对象跟踪器中的对象轨迹（可选）
            
        Returns:
            光流特征列表
        """
        # 将帧转换为灰度图像
        if frame is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None and self.prev_gray is None:
            # 第一帧，没有光流
            self.prev_gray = gray
            
            # 对于Lucas-Kanade方法，初始化要跟踪的点
            if self.method == 'pyr_lk':
                self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            
            return []
        
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = self.prev_gray
        
        features = []
        
        try:
            if self.method == 'farneback':
                features = self._extract_farneback(prev_gray, gray, tracks)
            elif self.method == 'pyr_lk':
                features = self._extract_lucas_kanade(prev_gray, gray)
        
        except Exception as e:
            self.logger.error(f"Error calculating optical flow: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        # 更新上一帧灰度图像
        self.prev_gray = gray
        
        return features
    
    def _extract_farneback(self, prev_gray: np.ndarray, gray: np.ndarray, 
                          tracks: List[Dict[str, Any]] = None) -> List[MotionFeature]:
        """使用Farneback方法提取光流。"""
        features = []
        
        if self.use_gpu:
            # GPU实现
            prev_cuda = cv2.cuda_GpuMat(prev_gray)
            curr_cuda = cv2.cuda_GpuMat(gray)
            flow_cuda = self.flow_calculator.calc(prev_cuda, curr_cuda, None)
            flow = flow_cuda.download()
        else:
            # CPU实现
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, 0
            )
        
        # 保存计算的光流，以便后续可以访问
        self.flow = flow
        
        # 提取整个帧的光流特征
        step = 16  # 每16个像素采样一次光流
        h, w = flow.shape[:2]
        
        # 采样光流向量
        y_indices, x_indices = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        # 在采样位置获取光流向量
        fx, fy = flow[y_indices, x_indices].T
        
        # 过滤掉小运动
        mag = np.sqrt(fx*fx + fy*fy)
        mask = mag > 1.0
        
        # 创建运动特征
        for i in range(len(x_indices)):
            if mask[i]:
                features.append(MotionFeature(
                    type='optical_flow',
                    data=np.array([fx[i], fy[i]]),
                    position=(int(x_indices[i]), int(y_indices[i])),
                    frame_idx=0,  # 由调用者设置
                    confidence=float(mag[i] / np.max(mag) if np.max(mag) > 0 else 0)
                ))
        
        # 如果提供了轨迹，则计算每个轨迹的光流
        if tracks:
            for track in tracks:
                box = track['box']
                track_id = track['id']
                
                # 计算边界框的中心
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                
                # 在中心获取光流
                if 0 <= cy < h and 0 <= cx < w:
                    fx, fy = flow[cy, cx]
                    
                    # 只有当有显著运动时才添加
                    mag = np.sqrt(fx*fx + fy*fy)
                    if mag > 1.0:
                        features.append(MotionFeature(
                            type='object_flow',
                            data=np.array([fx, fy]),
                            position=(cx, cy),
                            frame_idx=0,  # 由调用者设置
                            object_id=track_id,
                            confidence=1.0
                        ))
        
        return features
    
    def _extract_lucas_kanade(self, prev_gray: np.ndarray, gray: np.ndarray) -> List[MotionFeature]:
        """使用Lucas-Kanade方法提取光流。"""
        features = []
        
        if self.prev_pts is None or len(self.prev_pts) == 0:
            # 初始化要跟踪的点
            self.prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
            return []
        
        # 使用Lucas-Kanade方法计算光流
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        # 选择好的点
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = self.prev_pts[status == 1]
            
            # 为每个点创建运动特征
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                nx, ny = new.ravel()
                ox, oy = old.ravel()
                
                # 计算位移
                dx = nx - ox
                dy = ny - oy
                
                # 计算幅度
                mag = np.sqrt(dx*dx + dy*dy)
                
                # 只有当有显著运动时才添加
                if mag > 1.0:
                    features.append(MotionFeature(
                        type='sparse_flow',
                        data=np.array([dx, dy]),
                        position=(int(nx), int(ny)),
                        frame_idx=0,  # 由调用者设置
                        confidence=float(1.0 / (err[i][0] + 1e-6))  # 使用误差的倒数作为置信度
                    ))

            
            # 更新上一帧的点
            self.prev_pts = good_new.reshape(-1, 1, 2)
        else:
            # 如果跟踪失败，则重置点
            self.prev_pts = None
        
        return features
    
    def reset(self):
        """重置光流提取器。"""
        self.prev_gray = None
        self.flow = None
        self.logger.info("Optical flow extractor reset")
        
    def get_flow(self):
        """
        获取当前计算的光流
        
        Returns:
            np.ndarray: 光流场或None（如果尚未计算）
        """
        return self.flow 