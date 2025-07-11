#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频监控系统 - 配置管理模块
用于管理系统配置和参数
"""

import os
import json
import yaml

class Config:
    """配置管理类"""
    
    @staticmethod
    def load_config(config_path):
        # 加载配置文件,传入文件路径，返回解析之后的Python字典
        """加载配置文件
        
        Args:
            config_path: 配置文件路径，支持json和yaml格式
            
        Returns:
            配置字典
        """
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 获取文件扩展名
        _, ext = os.path.splitext(config_path)
        # 根据文件扩展名选择解析方式
        if ext.lower() in ('.yaml', '.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
    
    @staticmethod
    def save_config(config, config_path):
        """保存配置到文件
        
        Args:
            config: 配置字典，从上一个方法传入
            config_path: 配置文件保存路径，类似.json或者.yaml
        """
        # 获取文件扩展名
        _, ext = os.path.splitext(config_path)
        # 创建目标文件夹
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        # 如果扩展名是 .yaml 或 .yml，用 yaml.dump 保存为 YAML 格式
        if ext.lower() in ('.yaml', '.yml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        # 如果扩展名是 .json，用 json.dump 保存为 JSON 格式
        elif ext.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")

__all__ = ['Config'] 