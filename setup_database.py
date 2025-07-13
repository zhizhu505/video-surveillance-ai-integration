#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库初始化脚本
用于设置MySQL数据库和创建表结构
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

def create_database(config):
    """创建数据库和表结构"""
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
            
            # 读取SQL文件
            sql_file = Path("database_schema.sql")
            if not sql_file.exists():
                print("错误: database_schema.sql 文件不存在")
                return False
            
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句，正确处理存储过程
            statements = []
            current_statement = ""
            in_procedure = False
            delimiter = ";"
            
            for line in sql_content.split('\n'):
                line = line.strip()
                
                # 处理分隔符
                if line.startswith('DELIMITER'):
                    if in_procedure:
                        # 结束当前存储过程
                        if current_statement.strip():
                            statements.append(current_statement.strip())
                        current_statement = ""
                        in_procedure = False
                    delimiter = line.split()[1]
                    continue
                
                # 跳过注释
                if line.startswith('--') or not line:
                    continue
                
                # 处理存储过程
                if line.startswith('CREATE PROCEDURE'):
                    in_procedure = True
                    current_statement = line
                elif in_procedure:
                    current_statement += " " + line
                    if line.endswith(delimiter):
                        statements.append(current_statement.strip())
                        current_statement = ""
                        in_procedure = False
                else:
                    # 处理普通SQL语句
                    if current_statement:
                        current_statement += " " + line
                    else:
                        current_statement = line
                    
                    if line.endswith(';'):
                        statements.append(current_statement.strip())
                        current_statement = ""
            
            # 执行SQL语句
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement)
                        print(f"✓ 执行SQL: {statement[:50]}...")
                    except Exception as e:
                        # 检查是否是"已存在"的错误，这些是正常的
                        error_msg = str(e)
                        if any(keyword in error_msg.lower() for keyword in ['already exists', 'duplicate entry']):
                            print(f"ℹ 跳过已存在的对象: {statement[:50]}...")
                        else:
                            print(f"⚠ 执行SQL失败: {e}")
                            print(f"SQL: {statement[:100]}...")
            
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
    print("=== 视频监控告警系统数据库初始化 ===")
    
    # 加载配置
    config = load_database_config()
    if not config:
        return False
    
    print(f"数据库配置: {config['mysql']['host']}:{config['mysql']['port']}")
    
    # 创建数据库
    if not create_database(config):
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