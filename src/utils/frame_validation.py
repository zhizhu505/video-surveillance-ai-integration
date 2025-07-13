import cv2
import numpy as np
import time
import os
from datetime import datetime


def validate_frame(frame, frame_number):
    """
    验证视频帧并返回诊断信息。
    
    参数:
        frame: 要验证的OpenCV帧
        frame_number: 帧序号
        
    返回:
        dict: 包含帧验证结果的字典
    """
    if frame is None:
        return {
            'valid': False,
            'error': 'Frame is None',
            'frame_number': frame_number
        }
    
    # 检查帧维度
    height, width = frame.shape[:2]
    
    # 检查帧是否有内容（不是完全黑色或白色）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    has_content = std_dev > 5.0  # 确定帧是否有内容的阈值
    
    # 检查是否损坏（NaN值）
    has_nan = np.isnan(frame).any()
    
    return {
        'valid': not has_nan and has_content,
        'frame_number': frame_number,
        'dimensions': (width, height),
        'brightness': brightness,
        'std_dev': std_dev,
        'has_content': has_content,
        'has_nan': has_nan
    }


def save_frame_sample(frame, frame_number, output_dir="frame_samples"):
    """
    将样本帧保存到磁盘以供手动检查。
    
    参数:
        frame: 要保存的OpenCV帧
        frame_number: 文件名中的帧序号
        output_dir: 帧应保存的目录
    
    返回:
        str: 保存的帧的路径或None（如果保存失败）
    """
    if frame is None:
        return None
        
    # 如果目录不存在，则创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带有时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/frame_{frame_number}_{timestamp}.jpg"
    
    try:
        cv2.imwrite(filename, frame)
        return filename
    except Exception as e:
        print(f"Error saving frame: {str(e)}")
        return None


def draw_diagnostics(frame, diagnostics):
    """
    在帧上绘制诊断信息。
    
    参数:
        frame: 要注释的OpenCV帧
        diagnostics: 包含诊断信息的字典
        
    返回:
        frame: 注释后的帧
    """
    if frame is None:
        return None
        
    # 创建帧的副本
    annotated = frame.copy()
    
    # 绘制帧序号
    cv2.putText(
        annotated,
        f"Frame: {diagnostics['frame_number']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    # 绘制维度
    cv2.putText(
        annotated,
        f"Dimensions: {diagnostics['dimensions'][0]}x{diagnostics['dimensions'][1]}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1
    )
    
    # 绘制亮度信息
    cv2.putText(
        annotated,
        f"Brightness: {diagnostics['brightness']:.1f}, StdDev: {diagnostics['std_dev']:.1f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1
    )
    
    # 绘制验证状态
    status_color = (0, 255, 0) if diagnostics['valid'] else (0, 0, 255)
    cv2.putText(
        annotated,
        f"Valid: {diagnostics['valid']}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        1
    )
    
    # 绘制时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        timestamp,
        (10, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return annotated 