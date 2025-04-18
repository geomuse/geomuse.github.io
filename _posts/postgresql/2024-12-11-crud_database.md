---
layout: post
title:  postgresql crud database
date:   2024-12-11 11:24:29 +0800
categories: 
    - database
    - postgresql
---

PostgreSQL 数据库基本 CRUD（Create、Read、Update、Delete）操作的教学：

### 环境准备
1. **安装 PostgreSQL**：
   - 在官网下载安装适合的版本：[PostgreSQL 官网](https://www.postgresql.org/download/)
2. **安装管理工具**（可选）：
   - 使用 `psql` CLI 工具或图形界面工具如 pgAdmin。

3. **连接数据库**：
   使用以下命令连接 PostgreSQL：
   ```bash
   psql -U postgres
   ```
   系统会提示输入密码。

### 创建数据库和表
1. **创建数据库**：
   ```sql
   CREATE DATABASE my_database;
   ```

2. **切换到新数据库**：
   ```bash
   \c my_database
   ```

3. **创建表**：
   ```sql
   CREATE TABLE users (
       id SERIAL PRIMARY KEY,
       name VARCHAR(100),
       email VARCHAR(100),
       age INT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

### CRUD 操作

#### 1. 创建（INSERT）
向表中插入数据：
```sql
INSERT INTO users (name, email, age) 
VALUES ('Alice', 'alice@example.com', 30);
```

插入多条数据：
```sql
INSERT INTO users (name, email, age)
VALUES 
('Bob', 'bob@example.com', 25),
('Charlie', 'charlie@example.com', 35);
```

#### 2. 读取（SELECT）
获取所有用户：
```sql
SELECT * FROM users;
```

筛选数据：
```sql
SELECT name, email 
FROM users 
WHERE age > 30;
```

排序数据：
```sql
SELECT * FROM users 
ORDER BY age DESC;
```

分页：
```sql
SELECT * FROM users
LIMIT 10 OFFSET 0; -- 每页显示 10 条，从第 0 条开始
```

#### 3. 更新（UPDATE）
更新用户的年龄：
```sql
UPDATE users
SET age = 31
WHERE name = 'Alice';
```

批量更新：
```sql
UPDATE users
SET age = age + 1
WHERE age < 30;
```

#### 4. 删除（DELETE）
删除特定用户：
```sql
DELETE FROM users
WHERE email = 'alice@example.com';
```

删除所有用户：
```sql
DELETE FROM users;
```

### 事务管理
PostgreSQL 默认每个命令是独立事务，但也可以手动管理事务。

1. **启动事务**：
   ```sql
   BEGIN;
   ```

2. **执行操作**：
   ```sql
   INSERT INTO users (name, email, age) VALUES ('David', 'david@example.com', 28);
   ```

3. **提交事务**：
   ```sql
   COMMIT;
   ```

4. **回滚事务**：
   如果发生错误或需要撤销：
   ```sql
   ROLLBACK;
   ```

### 示例：完整的 CRUD 操作脚本
```sql
-- 创建数据库和表
CREATE DATABASE demo_db;
\c demo_db

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    position VARCHAR(100),
    salary NUMERIC,
    hire_date DATE DEFAULT CURRENT_DATE
);

-- 插入数据
INSERT INTO employees (name, position, salary)
VALUES 
('John Doe', 'Manager', 5000),
('Jane Smith', 'Developer', 4000),
('Alice Brown', 'Analyst', 3500);

-- 查询数据
SELECT * FROM employees;

-- 更新数据
UPDATE employees
SET salary = salary * 1.1
WHERE position = 'Developer';

-- 删除数据
DELETE FROM employees
WHERE name = 'Alice Brown';

-- 查看最终数据
SELECT * FROM employees;
```