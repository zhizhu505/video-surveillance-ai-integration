import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading

from models.alert.alert_event import AlertEvent
from models.alert.alert_rule import AlertLevel


class AlertDatabase:
    """
    告警数据库管理器，负责告警数据的持久化存储和查询
    """
    
    def __init__(self, db_path: str = "alerts.db"):
        """
        初始化告警数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.logger = logging.getLogger("AlertDatabase")
        self.lock = threading.Lock()
        
        # 初始化数据库
        self._init_database()
        
    def _init_database(self):
        """初始化数据库表结构"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建告警事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_events (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    danger_level TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    related_events TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建告警图像表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    image_type TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES alert_events (id) ON DELETE CASCADE
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON alert_events (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON alert_events (level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON alert_events (source_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_acknowledged ON alert_events (acknowledged)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON alert_events (created_at)')
            
            conn.commit()
            conn.close()
            
        self.logger.info(f"数据库初始化完成: {self.db_path}")
    
    def save_alert_event(self, event: AlertEvent, image_paths: Optional[Dict[str, str]] = None) -> bool:
        """
        保存告警事件到数据库
        
        Args:
            event: 告警事件对象
            image_paths: 相关图像路径字典
            
        Returns:
            是否保存成功
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 插入告警事件
                cursor.execute('''
                    INSERT INTO alert_events 
                    (id, rule_id, level, danger_level, source_type, timestamp, message, details, 
                     frame_idx, acknowledged, related_events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id,
                    event.rule_id,
                    event.level.name,
                    event.danger_level,
                    event.source_type,
                    event.timestamp,
                    event.message,
                    json.dumps(event.details),
                    event.frame_idx,
                    event.acknowledged,
                    json.dumps(event.related_events)
                ))
                
                # 保存相关图像路径
                if image_paths:
                    for image_type, image_path in image_paths.items():
                        cursor.execute('''
                            INSERT INTO alert_images (event_id, image_type, image_path)
                            VALUES (?, ?, ?)
                        ''', (event.id, image_type, image_path))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"告警事件已保存到数据库: {event.id}")
                return True
                
        except Exception as e:
            self.logger.error(f"保存告警事件失败: {str(e)}")
            return False
    
    def get_alert_events(self, 
                        limit: int = 100,
                        offset: int = 0,
                        level: Optional[str] = None,
                        source_type: Optional[str] = None,
                        acknowledged: Optional[bool] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        search_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询告警事件
        
        Args:
            limit: 返回结果数量限制
            offset: 分页偏移量
            level: 告警级别过滤
            source_type: 来源类型过滤
            acknowledged: 是否已确认过滤
            start_time: 开始时间戳
            end_time: 结束时间戳
            search_text: 搜索文本（在消息和详情中搜索）
            
        Returns:
            告警事件列表
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if level:
                    conditions.append("level = ?")
                    params.append(level)
                
                if source_type:
                    conditions.append("source_type = ?")
                    params.append(source_type)
                
                if acknowledged is not None:
                    conditions.append("acknowledged = ?")
                    params.append(acknowledged)
                
                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(start_time)
                
                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(end_time)
                
                if search_text:
                    conditions.append("(message LIKE ? OR details LIKE ?)")
                    search_pattern = f"%{search_text}%"
                    params.extend([search_pattern, search_pattern])
                
                # 构建SQL查询
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                sql = f'''
                    SELECT e.*, 
                           GROUP_CONCAT(i.image_path) as image_paths,
                           GROUP_CONCAT(i.image_type) as image_types
                    FROM alert_events e
                    LEFT JOIN alert_images i ON e.id = i.event_id
                    WHERE {where_clause}
                    GROUP BY e.id
                    ORDER BY e.timestamp DESC
                    LIMIT ? OFFSET ?
                '''
                
                params.extend([limit, offset])
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # 转换为字典列表
                results = []
                for row in rows:
                    event_dict = dict(row)
                    
                    # 解析JSON字段
                    try:
                        event_dict['details'] = json.loads(event_dict['details'])
                        event_dict['related_events'] = json.loads(event_dict['related_events'])
                    except:
                        event_dict['details'] = {}
                        event_dict['related_events'] = []
                    
                    # 处理图像路径
                    if event_dict['image_paths']:
                        image_paths = event_dict['image_paths'].split(',')
                        image_types = event_dict['image_types'].split(',')
                        event_dict['images'] = dict(zip(image_types, image_paths))
                    else:
                        event_dict['images'] = {}
                    
                    # 添加可读时间
                    event_dict['datetime'] = datetime.fromtimestamp(event_dict['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    results.append(event_dict)
                
                conn.close()
                return results
                
        except Exception as e:
            self.logger.error(f"查询告警事件失败: {str(e)}")
            return []
    
    def get_alert_count(self, 
                       level: Optional[str] = None,
                       source_type: Optional[str] = None,
                       acknowledged: Optional[bool] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> int:
        """
        获取告警事件数量
        
        Args:
            level: 告警级别过滤
            source_type: 来源类型过滤
            acknowledged: 是否已确认过滤
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            告警事件数量
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if level:
                    conditions.append("level = ?")
                    params.append(level)
                
                if source_type:
                    conditions.append("source_type = ?")
                    params.append(source_type)
                
                if acknowledged is not None:
                    conditions.append("acknowledged = ?")
                    params.append(acknowledged)
                
                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(start_time)
                
                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(end_time)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                sql = f"SELECT COUNT(*) FROM alert_events WHERE {where_clause}"
                
                cursor.execute(sql, params)
                count = cursor.fetchone()[0]
                
                conn.close()
                return count
                
        except Exception as e:
            self.logger.error(f"获取告警数量失败: {str(e)}")
            return 0
    
    def acknowledge_alert(self, event_id: str) -> bool:
        """
        确认告警事件
        
        Args:
            event_id: 告警事件ID
            
        Returns:
            是否确认成功
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE alert_events 
                    SET acknowledged = TRUE 
                    WHERE id = ?
                ''', (event_id,))
                
                affected_rows = cursor.rowcount
                conn.commit()
                conn.close()
                
                return affected_rows > 0
                
        except Exception as e:
            self.logger.error(f"确认告警失败: {str(e)}")
            return False
    
    def unacknowledge_alert(self, event_id: str) -> bool:
        """
        取消确认告警事件
        
        Args:
            event_id: 告警事件ID
            
        Returns:
            是否取消确认成功
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE alert_events 
                    SET acknowledged = FALSE 
                    WHERE id = ?
                ''', (event_id,))
                
                affected_rows = cursor.rowcount
                conn.commit()
                conn.close()
                
                return affected_rows > 0
                
        except Exception as e:
            self.logger.error(f"取消确认告警失败: {str(e)}")
            return False
    
    def get_alert_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        获取告警统计信息
        
        Args:
            days: 统计天数
            
        Returns:
            统计信息字典
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 计算时间范围
                end_time = time.time()
                start_time = end_time - (days * 24 * 3600)
                
                # 总告警数
                cursor.execute('''
                    SELECT COUNT(*) FROM alert_events 
                    WHERE timestamp >= ?
                ''', (start_time,))
                total_alerts = cursor.fetchone()[0]
                
                # 按级别统计
                cursor.execute('''
                    SELECT level, COUNT(*) FROM alert_events 
                    WHERE timestamp >= ?
                    GROUP BY level
                ''', (start_time,))
                level_stats = dict(cursor.fetchall())
                
                # 按来源类型统计
                cursor.execute('''
                    SELECT source_type, COUNT(*) FROM alert_events 
                    WHERE timestamp >= ?
                    GROUP BY source_type
                ''', (start_time,))
                source_stats = dict(cursor.fetchall())
                
                # 已确认和未确认统计
                cursor.execute('''
                    SELECT acknowledged, COUNT(*) FROM alert_events 
                    WHERE timestamp >= ?
                    GROUP BY acknowledged
                ''', (start_time,))
                ack_stats = dict(cursor.fetchall())
                
                # 每日告警趋势
                cursor.execute('''
                    SELECT DATE(datetime(timestamp, 'unixepoch')) as date, COUNT(*) 
                    FROM alert_events 
                    WHERE timestamp >= ?
                    GROUP BY date
                    ORDER BY date
                ''', (start_time,))
                daily_trend = dict(cursor.fetchall())
                
                conn.close()
                
                return {
                    'total_alerts': total_alerts,
                    'level_statistics': level_stats,
                    'source_statistics': source_stats,
                    'acknowledgment_statistics': ack_stats,
                    'daily_trend': daily_trend,
                    'period_days': days
                }
                
        except Exception as e:
            self.logger.error(f"获取告警统计失败: {str(e)}")
            return {}
    
    def delete_old_alerts(self, days: int = 90) -> int:
        """
        删除指定天数之前的告警数据
        
        Args:
            days: 保留天数
            
        Returns:
            删除的记录数
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff_time = time.time() - (days * 24 * 3600)
                
                # 删除相关图像文件
                cursor.execute('''
                    SELECT image_path FROM alert_images 
                    WHERE event_id IN (
                        SELECT id FROM alert_events WHERE timestamp < ?
                    )
                ''', (cutoff_time,))
                
                image_paths = [row[0] for row in cursor.fetchall()]
                for image_path in image_paths:
                    try:
                        Path(image_path).unlink(missing_ok=True)
                    except:
                        pass
                
                # 删除告警记录（级联删除会同时删除图像记录）
                cursor.execute('''
                    DELETE FROM alert_events WHERE timestamp < ?
                ''', (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                self.logger.info(f"删除了 {deleted_count} 条旧告警记录")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"删除旧告警失败: {str(e)}")
            return 0
    
    def close(self):
        """关闭数据库连接"""
        pass  # SQLite会自动管理连接 