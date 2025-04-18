import psycopg2

DB_CONFIG = {
    "host": "localhost",        # 本地主机
    "port": 5432,               # PostgreSQL 默认端口
    "database": "employee",     # 你创建的数据库名称
    "user": "postgres",         # 数据库用户名
    "password": "kali"          # 安装时设置的密码
}

try:
    print("正在连接到 PostgreSQL 数据库...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("成功连接到 PostgreSQL 数据库！")

    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"数据库版本: {version[0]}")

    cursor.close()
    conn.close()
    print("已关闭数据库连接。")

except Exception as e:
    print(f"连接 PostgreSQL 数据库时发生错误: {e}")