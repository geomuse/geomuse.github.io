---
layout: post
title:  postgresql drop database
date:   2024-12-09 11:24:29 +0800
categories: 
    - database
    - postgresql
---

### **创建数据库**
- 使用 `CREATE DATABASE` 创建新数据库：
  ```sql
  CREATE DATABASE my_database;
  ```
  - **`my_database`** 是新数据库的名称。

- 检查数据库是否创建成功：
  ```bash
  \l
  ```
  - 列出所有数据库。

### **连接数据库**
- 使用 `\c` 命令连接到新创建的数据库：
  ```bash
  \c my_database
  ```

### **删除数据库**
- 使用 `DROP DATABASE` 删除数据库：
  ```sql
  DROP DATABASE my_database;
  ```
  - 注意：执行此操作将永久删除数据库及其所有数据。

---


### **创建表**
- 使用 `CREATE TABLE` 创建新表：
  ```sql
  CREATE TABLE users (
      id SERIAL PRIMARY KEY,
      name VARCHAR(50) NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```
  - **字段说明**：
    - `id`: 自增主键。
    - `name`: 用户名，`VARCHAR(50)` 限制长度为 50。
    - `email`: 邮箱地址，`UNIQUE` 确保值唯一。
    - `created_at`: 创建时间，默认值为当前时间。

### **查看表结构**
- 使用 `\d` 命令查看表结构：
  ```bash
  \d users
  ```

### **修改表**
- 添加新列：
  ```sql
  ALTER TABLE users ADD COLUMN age INT;
  ```

- 修改列数据类型：
  ```sql
  ALTER TABLE users ALTER COLUMN age TYPE VARCHAR(3);
  ```

- 删除列：
  ```sql
  ALTER TABLE users DROP COLUMN age;
  ```

### **删除表**
- 使用 `DROP TABLE` 删除表：
  ```sql
  DROP TABLE users;
  ```

| 数据类型      | 描述                                         | 示例                     |
|---------------|----------------------------------------------|--------------------------|
| `INT`         | 整数类型，存储数字                          | `123`, `-456`            |
| `SERIAL`      | 自增类型，通常用于主键                      | 自增值如 `1, 2, 3...`   |
| `VARCHAR(n)`  | 可变长度的字符串，限制最大长度 `n`           | `'Hello'`, `'Postgres'`  |
| `TEXT`        | 不限制长度的字符串                          | `'A very long text...'` |
| `BOOLEAN`     | 布尔类型，值为 `TRUE` 或 `FALSE`            | `TRUE`, `FALSE`          |
| `DATE`        | 日期类型                                    | `'2024-12-08'`           |
| `TIMESTAMP`   | 日期和时间类型                              | `'2024-12-08 12:00:00'`  |
| `DECIMAL`     | 精确小数类型，用于存储财务数据              | `123.45`                 |
| `JSON`        | 用于存储 JSON 格式数据                     | `{"key": "value"}`       |

### **练习**

1. **创建数据库**：
   ```sql
   CREATE DATABASE shop;
   ```

2. **切换到数据库**：
   ```bash
   \c shop
   ```

3. **创建表 `products`**：
   ```sql
   CREATE TABLE products (
       id SERIAL PRIMARY KEY,
       name VARCHAR(100) NOT NULL,
       price DECIMAL(10, 2) NOT NULL,
       stock INT NOT NULL DEFAULT 0,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

4. **修改表 `products`**：
   - 添加列 `category`：
     ```sql
     ALTER TABLE products ADD COLUMN category VARCHAR(50);
     ```
   - 删除列 `stock`：
     ```sql
     ALTER TABLE products DROP COLUMN stock;
     ```

5. **删除表**：
   ```sql
   DROP TABLE products;
   ```

---

以上内容帮助你掌握了 PostgreSQL 数据库和表的基本操作。如果需要更多练习或代码示例，请告诉我！