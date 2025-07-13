#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版数据库初始化脚本
只创建基本的表结构，不包含存储过程
"""

import json
import os
import sys
import pymysql
from pathlib import Path

def load_database_config():
    """加载数据库配置"""
    config_path = Path("src/config/database.json")
    if not config_path.exists():
        print("错误: 数据库配置文件不存在")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}")
        return None

def create_database_simple(config):
    """创建数据库和基本表结构"""
    mysql_config = config['mysql']
    
    try:
        # 连接MySQL服务器（不指定数据库）
        connection = pymysql.connect(
            host=mysql_config['host'],
            port=mysql_config['port'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            charset=mysql_config['charset']
        )
        
        with connection.cursor() as cursor:
            # 创建数据库
            database_name = mysql_config['database']
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✓ 数据库 {database_name} 创建成功")
            
            # 使用数据库
            cursor.execute(f"USE {database_name}")
            
            # 创建基本表结构
            create_tables_sql = [
                # 1. 告警事件表
                """
                CREATE TABLE IF NOT EXISTS alert_events (
                    id VARCHAR(64) PRIMARY KEY COMMENT '告警事件唯一ID',
                    rule_id VARCHAR(64) NOT NULL COMMENT '触发该告警的规则ID',
                    level ENUM('INFO', 'WARNING', 'ALERT', 'CRITICAL') NOT NULL COMMENT '告警级别',
                    danger_level ENUM('low', 'medium', 'high') NOT NULL COMMENT '危险级别',
                    source_type VARCHAR(50) NOT NULL COMMENT '触发告警的来源类型',
                    timestamp DOUBLE NOT NULL COMMENT '告警生成时的Unix时间戳',
                    message TEXT NOT NULL COMMENT '可读的告警信息',
                    details JSON COMMENT '告警的详细信息(JSON格式)',
                    frame_idx INT NOT NULL COMMENT '触发告警时的视频帧编号',
                    acknowledged BOOLEAN DEFAULT FALSE COMMENT '告警是否已被确认',
                    related_events JSON COMMENT '相关事件的ID列表(JSON格式)',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',
                    
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_level (level),
                    INDEX idx_source_type (source_type),
                    INDEX idx_acknowledged (acknowledged),
                    INDEX idx_created_at (created_at),
                    INDEX idx_rule_id (rule_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警事件表'
                """,
                
                # 2. 告警图像表
                """
                CREATE TABLE IF NOT EXISTS alert_images (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '图像记录ID',
                    event_id VARCHAR(64) NOT NULL COMMENT '关联的告警事件ID',
                    image_type ENUM('frame', 'thumbnail') NOT NULL COMMENT '图像类型',
                    image_path VARCHAR(500) NOT NULL COMMENT '图像文件路径',
                    file_size BIGINT COMMENT '文件大小(字节)',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
                    
                    FOREIGN KEY (event_id) REFERENCES alert_events(id) ON DELETE CASCADE,
                    INDEX idx_event_id (event_id),
                    INDEX idx_image_type (image_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警图像表'
                """,
                
                # 3. 告警规则表
                """
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id VARCHAR(64) PRIMARY KEY COMMENT '规则唯一ID',
                    name VARCHAR(100) NOT NULL COMMENT '规则名称',
                    description TEXT COMMENT '规则描述',
                    level ENUM('INFO', 'WARNING', 'ALERT', 'CRITICAL') NOT NULL COMMENT '告警级别',
                    source_type VARCHAR(50) NOT NULL COMMENT '告警来源类型',
                    conditions JSON COMMENT '触发告警的条件(JSON格式)',
                    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
                    cooldown INT DEFAULT 0 COMMENT '冷却期(秒)',
                    trigger_count INT DEFAULT 0 COMMENT '触发次数',
                    last_triggered DOUBLE DEFAULT 0 COMMENT '上次触发时间戳',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',
                    
                    INDEX idx_enabled (enabled),
                    INDEX idx_source_type (source_type),
                    INDEX idx_level (level)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警规则表'
                """,
                
                # 4. 系统配置表
                """
                CREATE TABLE IF NOT EXISTS system_config (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '配置ID',
                    config_key VARCHAR(100) NOT NULL UNIQUE COMMENT '配置键',
                    config_value TEXT COMMENT '配置值',
                    config_type ENUM('string', 'int', 'float', 'boolean', 'json') DEFAULT 'string' COMMENT '配置值类型',
                    description TEXT COMMENT '配置描述',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',
                    
                    INDEX idx_config_key (config_key)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统配置表'
                """,
                
                # 5. 告警统计表
                """
                CREATE TABLE IF NOT EXISTS alert_statistics (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '统计记录ID',
                    stat_date DATE NOT NULL COMMENT '统计日期',
                    stat_type VARCHAR(50) NOT NULL COMMENT '统计类型',
                    stat_value JSON COMMENT '统计值(JSON格式)',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
                    
                    UNIQUE KEY uk_date_type (stat_date, stat_type),
                    INDEX idx_stat_date (stat_date),
                    INDEX idx_stat_type (stat_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警统计表'
                """
            ]
            
            # 执行创建表的SQL
            for i, sql in enumerate(create_tables_sql, 1):
                try:
                    cursor.execute(sql)
                    print(f"✓ 创建表 {i}/5 成功")
                except Exception as e:
                    print(f"⚠ 创建表 {i}/5 失败: {e}")
            
            # 插入默认配置
            default_configs = [
                ('database_version', '1.0', 'string', '数据库版本'),
                ('max_alert_retention_days', '90', 'int', '告警保留天数'),
                ('auto_cleanup_enabled', 'true', 'boolean', '是否启用自动清理'),
                ('alert_image_quality', '80', 'int', '告警图像质量(1-100)'),
                ('max_alert_images_per_event', '5', 'int', '每个告警事件最大图像数')
            ]
            
            for config_key, config_value, config_type, description in default_configs:
                try:
                    cursor.execute("""
                        INSERT INTO system_config (config_key, config_value, config_type, description)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        config_value = VALUES(config_value),
                        config_type = VALUES(config_type),
                        description = VALUES(description)
                    """, (config_key, config_value, config_type, description))
                except Exception as e:
                    print(f"⚠ 插入配置 {config_key} 失败: {e}")
            
            # 插入默认告警规则
            default_rules = [
                ('fall_detection', '摔倒检测', '检测人员摔倒行为', 'CRITICAL', 'fall_detection', '{"confidence_threshold": 0.7}'),
                ('danger_zone_dwell', '危险区域停留', '检测人员在危险区域停留', 'ALERT', 'danger_zone_dwell', '{"dwell_time_threshold": 1.0}'),
                ('sudden_motion', '突发运动', '检测突发性大范围运动', 'WARNING', 'sudden_motion', '{"motion_threshold": 0.6}'),
                ('large_area_motion', '大范围运动', '检测大范围运动', 'INFO', 'large_area_motion', '{"area_threshold": 0.3}')
            ]
            
            for rule_id, name, description, level, source_type, conditions in default_rules:
                try:
                    cursor.execute("""
                        INSERT INTO alert_rules (id, name, description, level, source_type, conditions)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        name = VALUES(name),
                        description = VALUES(description),
                        level = VALUES(level),
                        source_type = VALUES(source_type),
                        conditions = VALUES(conditions)
                    """, (rule_id, name, description, level, source_type, conditions))
                except Exception as e:
                    print(f"⚠ 插入规则 {rule_id} 失败: {e}")
            
            connection.commit()
            print("✓ 数据库表结构创建完成")
            
            # 验证表是否创建成功
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"✓ 已创建的表: {[table[0] for table in tables]}")
            
            return True
            
    except Exception as e:
        print(f"错误: 创建数据库失败: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

def test_connection(config):
    """测试数据库连接"""
    mysql_config = config['mysql']
    
    try:
        connection = pymysql.connect(
            host=mysql_config['host'],
            port=mysql_config['port'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            database=mysql_config['database'],
            charset=mysql_config['charset']
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print("✓ 数据库连接测试成功")
            
            # 检查表是否存在
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            expected_tables = ['alert_events', 'alert_images', 'alert_rules', 'system_config', 'alert_statistics']
            existing_tables = [table[0] for table in tables]
            
            missing_tables = set(expected_tables) - set(existing_tables)
            if missing_tables:
                print(f"⚠ 缺少表: {missing_tables}")
                return False
            else:
                print("✓ 所有必需的表都存在")
                return True
                
    except Exception as e:
        print(f"错误: 数据库连接测试失败: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

def main():
    """主函数"""
    print("=== 视频监控告警系统数据库初始化（简化版）===")
    
    # 加载配置
    config = load_database_config()
    if not config:
        return False
    
    print(f"数据库配置: {config['mysql']['host']}:{config['mysql']['port']}")
    
    # 创建数据库
    if not create_database_simple(config):
        return False
    
    # 测试连接
    if not test_connection(config):
        return False
    
    print("\n=== 数据库初始化完成 ===")
    print("现在可以运行视频监控系统了")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 