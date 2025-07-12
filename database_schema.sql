-- 告警系统数据库表结构设计
-- 数据库名: video_surveillance_alerts

-- 创建数据库
CREATE DATABASE IF NOT EXISTS video_surveillance_alerts 
CHARACTER SET utf8mb4 COLLATE=utf8mb4_unicode_ci;

USE video_surveillance_alerts;

-- 1. 告警事件表（重建，id自增主键，event_time为DATETIME）
DROP TABLE IF EXISTS alert_events;
CREATE TABLE alert_events (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增告警事件ID',
    rule_id VARCHAR(64) NOT NULL COMMENT '触发该告警的规则ID',
    level ENUM('INFO', 'WARNING', 'ALERT', 'CRITICAL') NOT NULL COMMENT '告警级别',
    danger_level ENUM('low', 'medium', 'high') NOT NULL COMMENT '危险级别',
    source_type VARCHAR(50) NOT NULL COMMENT '触发告警的来源类型',
    event_time DATETIME NOT NULL COMMENT '告警生成时间',
    message TEXT NOT NULL COMMENT '可读的告警信息',
    details JSON COMMENT '告警的详细信息(JSON格式)',
    frame_idx INT NOT NULL COMMENT '触发告警时的视频帧编号',
    acknowledged BOOLEAN DEFAULT FALSE COMMENT '告警是否已被确认',
    related_events JSON COMMENT '相关事件的ID列表(JSON格式)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',
    INDEX idx_event_time (event_time),
    INDEX idx_level (level),
    INDEX idx_source_type (source_type),
    INDEX idx_acknowledged (acknowledged),
    INDEX idx_created_at (created_at),
    INDEX idx_rule_id (rule_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警事件表';

-- 2. 告警图像表
CREATE TABLE IF NOT EXISTS alert_images (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '图像记录ID',
    event_id INT NOT NULL COMMENT '关联的告警事件自增ID',
    image_type ENUM('frame', 'thumbnail') NOT NULL COMMENT '图像类型',
    image_path VARCHAR(500) NOT NULL COMMENT '图像文件路径',
    file_size BIGINT COMMENT '文件大小(字节)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    FOREIGN KEY (event_id) REFERENCES alert_events(id) ON DELETE CASCADE,
    INDEX idx_event_id (event_id),
    INDEX idx_image_type (image_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警图像表';

-- 3. 告警规则表
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警规则表';

-- 4. 系统配置表
CREATE TABLE IF NOT EXISTS system_config (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '配置ID',
    config_key VARCHAR(100) NOT NULL UNIQUE COMMENT '配置键',
    config_value TEXT COMMENT '配置值',
    config_type ENUM('string','int','float','boolean','json') DEFAULT 'string' COMMENT '配置值类型',
    description TEXT COMMENT '配置描述',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',
    INDEX idx_config_key (config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统配置表';

-- 5. 告警统计表(用于缓存统计结果)
CREATE TABLE IF NOT EXISTS alert_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '统计记录ID',
    stat_date DATE NOT NULL COMMENT '统计日期',
    stat_type VARCHAR(50) NOT NULL COMMENT '统计类型',
    stat_value JSON COMMENT '统计值(JSON格式)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    UNIQUE KEY uk_date_type (stat_date, stat_type),
    INDEX idx_stat_date (stat_date),
    INDEX idx_stat_type (stat_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='告警统计表';

-- 插入默认配置（如果不存在）
INSERT IGNORE INTO system_config (config_key, config_value, config_type, description) VALUES
('database_version', '1.0', 'string', '数据库版本'),
('max_alert_retention_days', '90', 'int', '告警保留天数'),
('auto_cleanup_enabled', 'true', 'boolean', '是否启用自动清理'),
('alert_image_quality', '80', 'int', '告警图像质量(1-100)'),
('max_alert_images_per_event', '5', 'int', '每个告警事件最大图像数');

-- 插入默认告警规则（如果不存在）
INSERT IGNORE INTO alert_rules (id, name, description, level, source_type, conditions) VALUES
('fall_detection', '摔倒检测', '检测人员摔倒行为', 'CRITICAL', 'fall_detection', '{"confidence_threshold": 0.7}'),
('danger_zone_dwell', '危险区域停留', '检测人员在危险区域停留', 'ALERT', 'danger_zone_dwell', '{"dwell_time_threshold": 1.0}'),
('sudden_motion', '突发运动', '检测突发性大范围运动', 'WARNING', 'sudden_motion', '{"motion_threshold": 0.6}'),
('large_area_motion', '大范围运动', '检测大范围运动', 'INFO', 'large_area_motion', '{"area_threshold": 0.3}');

-- 创建视图：告警事件详情视图
CREATE OR REPLACE VIEW alert_events_detail AS
SELECT 
    e.*,
    GROUP_CONCAT(CONCAT(i.image_type, ':', i.image_path) SEPARATOR '|') as image_paths,
    COUNT(i.id) as image_count
FROM alert_events e
LEFT JOIN alert_images i ON e.id = i.event_id
GROUP BY e.id; 