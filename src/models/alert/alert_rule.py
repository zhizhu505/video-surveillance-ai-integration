import logging
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

# 告警级别枚举,用于定义不同的严重程度
class AlertLevel(Enum):
    INFO = 0     # 信息事件
    WARNING = 1  # 需要关注的警告
    ALERT = 2    # 需要立即注意的告警
    CRITICAL = 3 # 需要紧急行动的严重事件
    
    @classmethod
    def from_string(cls, level_str: str) -> 'AlertLevel':
        """将字符串转换为AlertLevel。"""
        level_map = {
            'info': cls.INFO,
            'warning': cls.WARNING,
            'alert': cls.ALERT,
            'critical': cls.CRITICAL
        }
        return level_map.get(level_str.lower(), cls.INFO)


@dataclass
class AlertRule:
    """告警规则定义，用于不同的检测场景。"""
    id: str                    # 唯一规则标识符
    name: str                  # 可读性好的名称
    description: str           # 规则描述
    level: AlertLevel          # 告警严重级别
    source_type: str           # 告警来源类型（行为、对象、运动等）
    conditions: Dict[str, Any] # 触发告警的条件
    enabled: bool = True       # 是否启用
    cooldown: int = 0          # 冷却期，避免告警洪水
    
    # 跟踪上次触发时间
    last_triggered: float = 0  
    
    # 触发次数
    trigger_count: int = 0     
    
    def to_dict(self) -> Dict[str, Any]:
        """将规则转换为字典，用于序列化。"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'level': self.level.name,
            'source_type': self.source_type,
            'conditions': self.conditions,
            'enabled': self.enabled,
            'cooldown': self.cooldown,
            'trigger_count': self.trigger_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """从字典创建规则。"""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            level=AlertLevel.from_string(data['level']),
            source_type=data['source_type'],
            conditions=data['conditions'],
            enabled=data.get('enabled', True),
            cooldown=data.get('cooldown', 0),
            trigger_count=data.get('trigger_count', 0)
        )


@dataclass
class AlertRuleConfig:
    """告警规则配置。"""
    rules: List[AlertRule] = field(default_factory=list)
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add a rule to the configuration."""
        # 检查是否已存在相同ID的规则
        for i, existing_rule in enumerate(self.rules):
            if existing_rule.id == rule.id:
                # 替换现有规则
                self.rules[i] = rule
                return
        # 添加新规则
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """从配置中删除规则。"""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                self.rules.pop(i)
                return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """根据ID获取规则。"""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def get_rules_by_source(self, source_type: str) -> List[AlertRule]:
        """获取特定来源类型的所有规则。"""
        return [rule for rule in self.rules if rule.source_type == source_type and rule.enabled]
    
    def save_to_file(self, file_path: str) -> bool:
        """将规则保存到JSON文件。"""
        try:
            with open(file_path, 'w') as f:
                json.dump([rule.to_dict() for rule in self.rules], f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save rules to {file_path}: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'AlertRuleConfig':
        """从JSON文件加载规则。"""
        try:
            with open(file_path, 'r') as f:
                rules_data = json.load(f)
            
            rules = [AlertRule.from_dict(rule_data) for rule_data in rules_data]
            return cls(rules=rules)
        except Exception as e:
            logging.error(f"Failed to load rules from {file_path}: {e}")
            return cls()
    
    def create_default_rules(self) -> None:
        """创建默认规则，用于常见场景。"""
        default_rules = [
            AlertRule(
                id="intrusion_detection", # 入侵检测
                name="Intrusion Detection",
                description="Detect when a person enters a restricted area",
                level=AlertLevel.ALERT,
                source_type="behavior",
                conditions={
                    "behavior_type": "INTRUSION",
                    "confidence_threshold": 0.7
                }
            ),
            AlertRule(
                id="loitering_detection", # 徘徊检测
                name="Loitering Detection",
                description="Detect when a person stays in an area for too long",
                level=AlertLevel.WARNING,
                source_type="behavior",
                conditions={
                    "behavior_type": "WANDERING",
                    "min_duration": 30,  # seconds
                    "confidence_threshold": 0.6
                }
            ),
            AlertRule(
                id="fast_motion_detection", # 快速运动检测
                name="Fast Motion Detection",
                description="Detect rapid movement in the scene",
                level=AlertLevel.INFO,
                source_type="motion",
                conditions={
                    "motion_threshold": 50,
                    "min_area": 1000
                }
            ),
            AlertRule(
                id="object_counting", # 对象计数阈值
                name="Object Counting Threshold",
                description="Alert when too many objects are detected",
                level=AlertLevel.WARNING,
                source_type="object",
                conditions={
                    "class_name": "person",
                    "min_count": 5,
                    "confidence_threshold": 0.5
                }
            ),
            AlertRule(
                id="abandoned_object", # 遗留对象检测
                name="Abandoned Object Detection",
                description="Detect stationary objects left behind",
                level=AlertLevel.WARNING,
                source_type="object",
                conditions={
                    "class_name": "suitcase|backpack|handbag",
                    "stationary_time": 60,  # seconds
                    "confidence_threshold": 0.6
                }
            ),
            AlertRule(
                id="abnormal_scene", # 异常场景检测
                name="Abnormal Scene Detection",
                description="Detect abnormal scenes using Qwen-VL",
                level=AlertLevel.ALERT,
                source_type="scene",
                conditions={
                    "normal_context": "people walking normally in a hallway",
                    "anomaly_threshold": 0.7
                }
            ),
            AlertRule(
                id="fighting_detection", # 打架检测 
                name="Fighting Detection",
                description="Detect when people are fighting",
                level=AlertLevel.CRITICAL,
                source_type="behavior",
                conditions={
                    "behavior_type": "FIGHTING",
                    "confidence_threshold": 0.7
                }
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule) 