import pymysql
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading
import os

from models.alert.alert_event import AlertEvent
from models.alert.alert_rule import AlertLevel


class MySQLAlertDatabase:
    """
    MySQL告警数据库管理器，负责告警数据的持久化存储和查询
    """
    
    def __init__(self, host: str = 'localhost', port: int = 3306, 
                 user: str = 'root', password: str = '', 
                 database: str = 'video_surveillance_alerts',
                 charset: str = 'utf8mb4'):
        """
        初始化MySQL告警数据库
        
        Args:
            host: MySQL服务器地址
            port: MySQL端口
            user: 用户名
            password: 密码
            database: 数据库名
            charset: 字符集
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.logger = logging.getLogger("MySQLAlertDatabase")
        self.lock = threading.Lock()
        
        # 测试连接
        self._test_connection()
        
    def _get_connection(self):
        """获取数据库连接"""
        try:
            connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=True,
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def _test_connection(self):
        """测试数据库连接"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    self.logger.info(f"MySQL数据库连接成功: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            self.logger.error(f"MySQL数据库连接测试失败: {str(e)}")
            raise
    
    def save_alert_event(self, event: AlertEvent, image_paths: Optional[Dict[str, str]] = None) -> Optional[int]:
        """
        保存告警事件到数据库
        Args:
            event: 告警事件对象
            image_paths: 相关图像路径字典
        Returns:
            新插入的自增id，失败返回None
        """
        try:
            with self.lock:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        # 强制格式化为MySQL DATETIME格式
                        try:
                            event_time_str = datetime.fromtimestamp(float(event.timestamp)).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            event_time_str = str(event.timestamp)[:19]
                        self.logger.info(f"插入的时间字符串: {event_time_str}")
                        # 插入告警事件（不再传id，event_time为DATETIME字符串）
                        sql = """
                            INSERT INTO alert_events 
                            (rule_id, level, danger_level, source_type, event_time, message, details, 
                             frame_idx, acknowledged, related_events)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(sql, (
                            event.rule_id,
                            event.level.name,
                            event.danger_level,
                            event.source_type,
                            event_time_str,
                            event.message,
                            json.dumps(event.details),
                            event.frame_idx,
                            event.acknowledged,
                            json.dumps(event.related_events)
                        ))
                        new_id = cursor.lastrowid
                        # 保存相关图像路径
                        if image_paths:
                            for image_type, image_path in image_paths.items():
                                file_size = None
                                try:
                                    if os.path.exists(image_path):
                                        file_size = os.path.getsize(image_path)
                                except:
                                    pass
                                cursor.execute("""
                                    INSERT INTO alert_images (event_id, image_type, image_path, file_size)
                                    VALUES (%s, %s, %s, %s)
                                """, (new_id, image_type, image_path, file_size))
                        self.logger.debug(f"告警事件已保存到MySQL数据库: {event.message}, 新id: {new_id}")
                        return new_id
        except Exception as e:
            self.logger.error(f"保存告警事件到MySQL失败: {str(e)}")
            return None
    
    def get_alert_events(self, 
                        limit: int = 100,
                        offset: int = 0,
                        danger_level: Optional[str] = None,
                        level: Optional[str] = None,
                        source_type: Optional[str] = None,
                        acknowledged: Optional[bool] = None,
                        start_time: Optional[str] = None,  # 起始时间DATETIME字符串
                        end_time: Optional[str] = None,    # 结束时间DATETIME字符串
                        search_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询告警事件
        
        Args:
            limit: 返回结果数量限制
            offset: 分页偏移量
            danger_level: 危险级别过滤
            level: 告警级别过滤
            source_type: 来源类型过滤
            acknowledged: 是否已确认过滤
            start_time: 起始时间DATETIME字符串
            end_time: 结束时间DATETIME字符串
            search_text: 搜索文本（在消息和详情中搜索）
            
        Returns:
            告警事件列表
        """
        try:
            with self.lock:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        
                        # 构建查询条件
                        conditions = []
                        params = []
                        
                        if danger_level:
                            conditions.append("danger_level = %s")
                            params.append(danger_level)
                        
                        if level:
                            conditions.append("level = %s")
                            params.append(level)
                        
                        if source_type:
                            conditions.append("source_type = %s")
                            params.append(source_type)
                        
                        if acknowledged is not None:
                            conditions.append("acknowledged = %s")
                            params.append(acknowledged)
                        
                        if start_time:
                            conditions.append("event_time >= %s")
                            params.append(start_time)
                        
                        if end_time:
                            conditions.append("event_time <= %s")
                            params.append(end_time)
                        
                        if search_text:
                            conditions.append("(message LIKE %s OR details LIKE %s)")
                            search_pattern = f"%{search_text}%"
                            params.extend([search_pattern, search_pattern])
                        
                        # 构建SQL查询
                        where_clause = " AND ".join(conditions) if conditions else "1=1"
                        
                        sql = f"""
                            SELECT e.*, 
                                   GROUP_CONCAT(CONCAT(i.image_type, ':', i.image_path) SEPARATOR ',') as image_paths,
                                   GROUP_CONCAT(CONCAT(i.image_type, ':', i.file_size) SEPARATOR ',') as image_sizes
                            FROM alert_events e
                            LEFT JOIN alert_images i ON e.id = i.event_id
                            WHERE {where_clause}
                            GROUP BY e.id
                            ORDER BY e.id ASC
                            LIMIT %s OFFSET %s
                        """
                        
                        params.extend([limit, offset])
                        
                        cursor.execute(sql, params)
                        rows = cursor.fetchall()
                        
                        # 转换为字典列表
                        results = []
                        for row in rows:
                            event_dict = dict(row)
                            
                            # 解析JSON字段
                            try:
                                event_dict['details'] = json.loads(event_dict['details']) if event_dict['details'] else {}
                                event_dict['related_events'] = json.loads(event_dict['related_events']) if event_dict['related_events'] else []
                            except:
                                event_dict['details'] = {}
                                event_dict['related_events'] = []
                            
                            # 处理图像路径
                            if event_dict['image_paths']:
                                image_paths = event_dict['image_paths'].split(',')
                                image_sizes = event_dict['image_sizes'].split(',') if event_dict['image_sizes'] else []
                                event_dict['images'] = {}
                                for i, path in enumerate(image_paths):
                                    if ':' in path:
                                        img_type, img_path = path.split(':', 1)
                                        event_dict['images'][img_type] = img_path
                            else:
                                event_dict['images'] = {}
                            
                            # 添加可读时间
                            et = event_dict['event_time']
                            if isinstance(et, datetime):
                                event_dict['datetime'] = et.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                try:
                                    event_dict['datetime'] = datetime.fromtimestamp(float(et)).strftime('%Y-%m-%d %H:%M:%S')
                                except Exception:
                                    event_dict['datetime'] = str(et)
                            
                            results.append(event_dict)
                        
                        return results
                        
        except Exception as e:
            self.logger.error(f"查询告警事件失败: {str(e)}")
            return []
    
    def get_alert_count(self, 
                       danger_level: Optional[str] = None,
                       level: Optional[str] = None,
                       source_type: Optional[str] = None,
                       acknowledged: Optional[bool] = None,
                       start_time: Optional[str] = None,  # 起始时间DATETIME字符串
                       end_time: Optional[str] = None    # 结束时间DATETIME字符串
    ) -> int:
        """
        获取告警事件数量
        
        Args:
            danger_level: 危险级别过滤
            level: 告警级别过滤
            source_type: 来源类型过滤
            acknowledged: 是否已确认过滤
            start_time: 起始时间DATETIME字符串
            end_time: 结束时间DATETIME字符串
            
        Returns:
            告警事件数量
        """
        try:
            with self.lock:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        
                        # 构建查询条件
                        conditions = []
                        params = []
                        
                        if danger_level:
                            conditions.append("danger_level = %s")
                            params.append(danger_level)
                        
                        if level:
                            conditions.append("level = %s")
                            params.append(level)
                        
                        if source_type:
                            conditions.append("source_type = %s")
                            params.append(source_type)
                        
                        if acknowledged is not None:
                            conditions.append("acknowledged = %s")
                            params.append(acknowledged)
                        
                        if start_time:
                            conditions.append("event_time >= %s")
                            params.append(start_time)
                        
                        if end_time:
                            conditions.append("event_time <= %s")
                            params.append(end_time)
                        
                        where_clause = " AND ".join(conditions) if conditions else "1=1"
                        
                        sql = f"SELECT COUNT(*) as count FROM alert_events WHERE {where_clause}"
                        
                        cursor.execute(sql, params)
                        result = cursor.fetchone()
                        
                        return result['count'] if result else 0
                        
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
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        
                        cursor.execute("""
                            UPDATE alert_events 
                            SET acknowledged = TRUE 
                            WHERE id = %s
                        """, (event_id,))
                        
                        return cursor.rowcount > 0
                        
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
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        
                        cursor.execute("""
                            UPDATE alert_events 
                            SET acknowledged = FALSE 
                            WHERE id = %s
                        """, (event_id,))
                        
                        return cursor.rowcount > 0
                        
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
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        end_time = time.time()
                        start_time = end_time - (days * 24 * 3600)
                        stats = {
                            'total_alerts': 0,
                            'unhandled_alerts': 0,
                            'handled_alerts': 0,
                            'high_level_alerts': 0,
                            'medium_level_alerts': 0,
                            'low_level_alerts': 0,
                            'today_alerts': 0,
                            'period_days': days
                        }
                        # 总告警数
                        cursor.execute("SELECT COUNT(*) as total FROM alert_events WHERE event_time >= %s", (start_time,))
                        row = cursor.fetchone()
                        stats['total_alerts'] = row['total'] if row and 'total' in row else 0
                        # 未处理告警
                        cursor.execute("SELECT COUNT(*) as count FROM alert_events WHERE event_time >= %s AND acknowledged = 0", (start_time,))
                        row = cursor.fetchone()
                        stats['unhandled_alerts'] = row['count'] if row and 'count' in row else 0
                        # 已处理告警
                        stats['handled_alerts'] = stats['total_alerts'] - stats['unhandled_alerts']
                        # 各级别
                        cursor.execute("SELECT COUNT(*) as count FROM alert_events WHERE event_time >= %s AND danger_level = 'high'", (start_time,))
                        row = cursor.fetchone()
                        stats['high_level_alerts'] = row['count'] if row and 'count' in row else 0
                        cursor.execute("SELECT COUNT(*) as count FROM alert_events WHERE event_time >= %s AND danger_level = 'medium'", (start_time,))
                        row = cursor.fetchone()
                        stats['medium_level_alerts'] = row['count'] if row and 'count' in row else 0
                        cursor.execute("SELECT COUNT(*) as count FROM alert_events WHERE event_time >= %s AND danger_level = 'low'", (start_time,))
                        row = cursor.fetchone()
                        stats['low_level_alerts'] = row['count'] if row and 'count' in row else 0
                        # 今日告警
                        import datetime
                        today = datetime.date.today()
                        today_start = int(datetime.datetime.combine(today, datetime.time.min).timestamp())
                        cursor.execute("SELECT COUNT(*) as count FROM alert_events WHERE event_time >= %s", (today_start,))
                        row = cursor.fetchone()
                        stats['today_alerts'] = row['count'] if row and 'count' in row else 0
                        return stats
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
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        
                        # 使用存储过程清理旧数据
                        cursor.callproc('cleanup_old_alerts', [days])
                        result = cursor.fetchone()
                        
                        deleted_count = result['deleted_count'] if result else 0
                        self.logger.info(f"删除了 {deleted_count} 条旧告警记录")
                        return deleted_count
                        
        except Exception as e:
            self.logger.error(f"删除旧告警失败: {str(e)}")
            return 0
    
    def close(self):
        """关闭数据库连接"""
        pass  # PyMySQL会自动管理连接
    
    def get_system_config(self, config_key: str, default_value: Any = None) -> Any:
        """
        获取系统配置
        
        Args:
            config_key: 配置键
            default_value: 默认值
            
        Returns:
            配置值
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT config_value, config_type 
                        FROM system_config 
                        WHERE config_key = %s
                    """, (config_key,))
                    
                    result = cursor.fetchone()
                    if result:
                        value = result['config_value']
                        config_type = result['config_type']
                        
                        # 根据类型转换值
                        if config_type == 'int':
                            return int(value)
                        elif config_type == 'float':
                            return float(value)
                        elif config_type == 'boolean':
                            return value.lower() == 'true'
                        elif config_type == 'json':
                            return json.loads(value)
                        else:
                            return value
                    else:
                        return default_value
                        
        except Exception as e:
            self.logger.error(f"获取系统配置失败: {str(e)}")
            return default_value
    
    def set_system_config(self, config_key: str, config_value: Any, 
                         config_type: str = 'string', description: str = '') -> bool:
        """
        设置系统配置
        
        Args:
            config_key: 配置键
            config_value: 配置值
            config_type: 配置值类型
            description: 配置描述
            
        Returns:
            是否设置成功
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO system_config (config_key, config_value, config_type, description)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        config_value = VALUES(config_value),
                        config_type = VALUES(config_type),
                        description = VALUES(description)
                    """, (config_key, str(config_value), config_type, description))
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"设置系统配置失败: {str(e)}")
            return False 