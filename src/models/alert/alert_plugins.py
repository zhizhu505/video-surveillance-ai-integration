import abc
import smtplib
import requests
import logging
import os
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, List, Any, Optional, Callable

from models.alert_event import AlertEvent
from models.alert_rule import AlertLevel


class AlertNotifier(abc.ABC):
    """告警通知插件的抽象基类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知器，使用配置字典。
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.name = "BaseNotifier"
        self.enabled = config.get("enabled", True)
        self.min_level = AlertLevel.from_string(config.get("min_level", "info"))
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"AlertNotifier.{self.name}")
    
    @abc.abstractmethod
    def notify(self, event: AlertEvent) -> bool:
        """
        发送告警事件通知。
        
        参数:
            event: 需要通知的告警事件
            
        返回:
            通知发送成功返回True，否则返回False
        """
        pass
    
    def should_notify(self, event: AlertEvent) -> bool:
        """
        检查此通知器是否应处理该事件。
        
        参数:
            event: 需要检查的告警事件
            
        返回:
            如果通知器应处理此事件则返回True，否则返回False
        """
        # Only notify if enabled and event level is >= min_level
        return self.enabled and event.level.value >= self.min_level.value


class EmailNotifier(AlertNotifier):
    """电子邮件通知插件。"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化电子邮件通知器。
        
        Args:
            config: 配置字典，包含以下键:
                smtp_server: SMTP服务器地址
                smtp_port: SMTP服务器端口
                smtp_username: SMTP用户名
                smtp_password: SMTP密码
                from_addr: 发送者电子邮件地址
                to_addrs: 收件人电子邮件地址列表
                use_ssl: 是否使用SSL
                enabled: 通知器是否启用
                min_level: 通知的最小告警级别
        """
        super().__init__(config)
        self.name = "EmailNotifier"
        
        # Email-specific config
        self.smtp_server = config.get("smtp_server", "")
        self.smtp_port = config.get("smtp_port", 587)
        self.smtp_username = config.get("smtp_username", "")
        self.smtp_password = config.get("smtp_password", "")
        self.from_addr = config.get("from_addr", "")
        self.to_addrs = config.get("to_addrs", [])
        self.use_ssl = config.get("use_ssl", False)
        
        # Validate required fields
        if not all([self.smtp_server, self.smtp_username, 
                    self.smtp_password, self.from_addr, self.to_addrs]):
            self.logger.error("Email notifier configuration is incomplete")
            self.enabled = False
    # 用于发送电子邮件通知
    def notify(self, event: AlertEvent) -> bool:
        """发送电子邮件通知。"""
        # 检查是否需要发送通知（基于告警级别和启用状态）
        if not self.should_notify(event):
            return False    
        
        try:
            # 创建消息
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ", ".join(self.to_addrs)
            msg['Subject'] = f"{event.level.name} Alert: {event.message}"
            
            # 创建HTML正文
            html = f"""
            <html>
            <head></head>
            <body>
                <h2>{event.level.name} Alert: {event.message}</h2>
                <p><strong>Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}</p>
                <p><strong>Source:</strong> {event.source_type}</p>
                <p><strong>Rule:</strong> {event.rule_id}</p>
                <h3>Details:</h3>
                <pre>{str(event.details)}</pre>
            </body>
            </html>
            """
            
            # 附加HTML正文
            msg.attach(MIMEText(html, 'html'))
            
            # 附加缩略图（如果可用）
            if event.thumbnail is not None:
                import cv2
                _, img_data = cv2.imencode('.jpg', event.thumbnail)
                image = MIMEImage(img_data.tobytes())
                image.add_header('Content-ID', '<image1>')
                msg.attach(image)
            
            # 连接到服务器并发送
            server = None
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
            
            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            server.quit()
            
            self.logger.info(f"Sent email notification for event {event.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class WebhookNotifier(AlertNotifier):
    """Webhook 通知插件。"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Webhook通知器。
        
        Args:
            config: 配置字典，包含以下键:
                webhook_url: Webhook URL    
                headers: HTTP头信息
                include_thumbnail: 是否包含缩略图
                enabled: 通知器是否启用
                min_level: 通知的最小告警级别
        """
        super().__init__(config)
        self.name = "WebhookNotifier"
        
        # Webhook 特定配置
        self.webhook_url = config.get("webhook_url", "")
        self.headers = config.get("headers", {})
        self.include_thumbnail = config.get("include_thumbnail", True)
        
        # Validate required fields
        if not self.webhook_url:
            self.logger.error("Webhook URL is required")
            self.enabled = False
    # 用于发送Webhook通知
    def notify(self, event: AlertEvent) -> bool:
        """发送Webhook通知。"""
        # 检查是否需要发送通知（基于告警级别和启用状态）
        if not self.should_notify(event):
            return False
        
        try:
            # Prepare payload
            payload = event.to_dict(include_images=self.include_thumbnail)
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            # Check response
            if response.status_code < 400:
                self.logger.info(f"Sent webhook notification for event {event.id}")
                return True
            else:
                self.logger.error(f"Webhook returned error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False

# 将告警信息写入日志文件
class FileNotifier(AlertNotifier):
    """基于文件的通知插件，将告警信息写入日志文件。"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文件通知器。
        
        Args:
            config: 配置字典，包含以下键:
                output_file: 输出文件路径
                append: 是否追加到文件或覆盖
                enabled: 通知器是否启用
                min_level: 通知的最小告警级别
        """
        super().__init__(config)
        self.name = "FileNotifier"
        
        # 文件特定配置
        self.output_file = config.get("output_file", "alerts.log")
        self.append = config.get("append", True)
        
        # 如果需要，创建目录
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        # 如果文件不存在或不追加，写入头
        if not os.path.exists(self.output_file) or not self.append:
            with open(self.output_file, 'w') as f:
                f.write("# Alert Log\n")
                f.write("# Timestamp | Level | Source | Message\n")
                f.write("# ----------------------------------------\n")
    
    def notify(self, event: AlertEvent) -> bool:
        """Write alert to log file."""
        if not self.should_notify(event):
            return False
        
        try:
            # Format log entry
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
            log_entry = f"{timestamp} | {event.level.name} | {event.source_type} | {event.message}\n"
            
            # Write to file
            with open(self.output_file, 'a') as f:
                f.write(log_entry)
            
            self.logger.info(f"Wrote alert to file for event {event.id}")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {e}")
            return False


class NotificationManager:
    """管理多个通知插件并处理告警分发。"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化通知管理器。
        
        Args:
            config: 配置字典
        """
        self.notifiers = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("NotificationManager")
        
        # 如果未提供配置，创建默认配置
        if config is None:
            config = {
                "file": {
                    "type": "file",
                    "enabled": True,
                    "min_level": "info",
                    "output_file": "alerts.log",
                    "append": True
                }
            }
        
        # Initialize notifiers from config
        self._init_notifiers(config)
        
        # Create notification thread
        self.notification_queue = []
        self.queue_lock = threading.Lock()
        self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.running = True
        self.notification_thread.start()
    
    def _init_notifiers(self, config: Dict[str, Any]):
        """Initialize notification plugins from config."""
        for name, notifier_config in config.items():
            notifier_type = notifier_config.get("type", "").lower()
            
            try:
                if notifier_type == "email":
                    self.notifiers.append(EmailNotifier(notifier_config))
                elif notifier_type == "webhook":
                    self.notifiers.append(WebhookNotifier(notifier_config))
                elif notifier_type == "file":
                    self.notifiers.append(FileNotifier(notifier_config))
                else:
                    self.logger.warning(f"Unknown notifier type: {notifier_type}")
            except Exception as e:
                self.logger.error(f"Failed to initialize notifier {name}: {e}")
        
        self.logger.info(f"Initialized {len(self.notifiers)} notification plugins")
    
    def add_notifier(self, notifier: AlertNotifier):
        """Add a notifier to the manager."""
        self.notifiers.append(notifier)
    
    def notify(self, event: AlertEvent, synchronous: bool = False):
        """
        Send notifications for an alert event.
        
        Args:
            event: Alert event to notify about
            synchronous: Whether to send notifications synchronously
        """
        if synchronous:
            # Send notifications immediately
            for notifier in self.notifiers:
                if notifier.should_notify(event):
                    notifier.notify(event)
        else:
            # Queue for background processing
            with self.queue_lock:
                self.notification_queue.append(event)
    
    def _notification_worker(self):
        """Background worker that processes the notification queue."""
        while self.running:
            # Process queue
            events_to_process = []
            with self.queue_lock:
                if self.notification_queue:
                    events_to_process = self.notification_queue
                    self.notification_queue = []
            
            # Process events
            for event in events_to_process:
                for notifier in self.notifiers:
                    try:
                        if notifier.should_notify(event):
                            notifier.notify(event)
                    except Exception as e:
                        self.logger.error(f"Error in notifier {notifier.name}: {e}")
            
            # Sleep if queue is empty
            if not events_to_process:
                time.sleep(0.5)
    
    def shutdown(self):
        """Shut down the notification manager."""
        self.running = False
        if self.notification_thread.is_alive():
            self.notification_thread.join(timeout=2.0)
        
        # Process any remaining notifications
        with self.queue_lock:
            events_to_process = self.notification_queue
            self.notification_queue = []
        
        # Process remaining events synchronously
        for event in events_to_process:
            for notifier in self.notifiers:
                try:
                    if notifier.should_notify(event):
                        notifier.notify(event)
                except Exception:
                    pass 