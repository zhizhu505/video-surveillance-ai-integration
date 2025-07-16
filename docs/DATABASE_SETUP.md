# 数据库设置指南

## 概述

本系统支持MySQL数据库来存储告警信息，提供更好的性能和可扩展性。

## 数据库表结构

### 主要表

1. **alert_events** - 告警事件表
   - 存储所有告警事件的基本信息
   - 包含告警级别、危险级别、来源类型等

2. **alert_images** - 告警图像表
   - 存储告警相关的图像文件路径
   - 支持frame和thumbnail两种图像类型

3. **alert_rules** - 告警规则表
   - 存储告警规则配置
   - 支持规则的启用/禁用和条件设置

4. **system_config** - 系统配置表
   - 存储系统级配置参数
   - 支持多种数据类型

5. **alert_statistics** - 告警统计表
   - 缓存统计结果以提高查询性能

## 安装步骤

### 1. 安装MySQL

#### Windows
```bash
# 下载并安装MySQL Community Server
# https://dev.mysql.com/downloads/mysql/
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

#### macOS
```bash
# 使用Homebrew
brew install mysql
brew services start mysql
```

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 配置数据库连接

编辑 `src/config/database.json` 文件：

```json
{
    "mysql": {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "your_password",
        "database": "video_surveillance_alerts",
        "charset": "utf8mb4"
    }
}
```

### 4. 初始化数据库

运行数据库初始化脚本：

```bash
python setup_database.py
```

这个脚本会：
- 创建数据库
- 创建所有必需的表
- 插入默认配置和规则
- 测试数据库连接

### 5. 验证安装

检查数据库是否正确创建：

```sql
USE video_surveillance_alerts;
SHOW TABLES;
SELECT * FROM system_config;
```

## 数据库配置选项

### 连接参数

- `host`: MySQL服务器地址
- `port`: MySQL端口（默认3306）
- `user`: 数据库用户名
- `password`: 数据库密码
- `database`: 数据库名称
- `charset`: 字符集（推荐utf8mb4）

### 性能优化

- `pool_size`: 连接池大小（默认10）
- `pool_recycle`: 连接回收时间（秒）
- `autocommit`: 自动提交（推荐true）

### 清理配置

- `retention_days`: 告警保留天数（默认90天）
- `auto_cleanup_hours`: 自动清理间隔（小时）

## 使用MySQL的优势

### 1. 性能优势
- 更好的并发处理能力
- 优化的查询性能
- 支持复杂的SQL查询

### 2. 功能优势
- 支持存储过程
- 支持视图
- 支持触发器
- 支持外键约束

### 3. 管理优势
- 更好的备份和恢复
- 支持主从复制
- 丰富的监控工具

## 故障排除

### 常见问题

1. **连接失败**
   ```
   错误: Can't connect to MySQL server
   ```
   解决：检查MySQL服务是否启动，连接参数是否正确

2. **权限错误**
   ```
   错误: Access denied for user
   ```
   解决：检查用户名和密码，确保用户有足够权限

3. **字符集错误**
   ```
   错误: Incorrect string value
   ```
   解决：确保使用utf8mb4字符集

### 调试命令

```bash
# 检查MySQL服务状态
sudo systemctl status mysql

# 连接MySQL
mysql -u root -p

# 查看数据库
SHOW DATABASES;

# 查看表结构
DESCRIBE alert_events;
```

## 备份和恢复

### 备份数据库

```bash
mysqldump -u root -p video_surveillance_alerts > backup.sql
```

### 恢复数据库

```bash
mysql -u root -p video_surveillance_alerts < backup.sql
```

## 性能监控

### 查看告警统计

```sql
-- 查看今日告警数
SELECT COUNT(*) FROM alert_events 
WHERE DATE(FROM_UNIXTIME(timestamp)) = CURDATE();

-- 查看未处理告警
SELECT COUNT(*) FROM alert_events 
WHERE acknowledged = FALSE;

-- 查看告警趋势
SELECT DATE(FROM_UNIXTIME(timestamp)) as date, COUNT(*) as count
FROM alert_events 
WHERE timestamp >= UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL 30 DAY))
GROUP BY date
ORDER BY date;
```

## 迁移指南

### 从SQLite迁移到MySQL

1. 导出SQLite数据
2. 转换数据格式
3. 导入MySQL数据库
4. 更新配置文件

### 数据同步

系统支持同时使用SQLite和MySQL，可以逐步迁移数据。

## 安全建议

1. **使用强密码**
2. **限制数据库访问IP**
3. **定期备份数据**
4. **监控数据库性能**
5. **及时更新MySQL版本**

## 技术支持

如果遇到问题，请检查：
1. MySQL服务是否正常运行
2. 连接参数是否正确
3. 用户权限是否足够
4. 防火墙设置是否允许连接 