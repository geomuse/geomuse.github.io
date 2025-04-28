---
layout: post
title:  postgresql create database
date:   2024-12-08 11:24:29 +0800
categories: 
    - database
    - postgresql
---

### 创建第一个数据库并连接

```postgresql
CREATE DATABASE employee;
```

切换到新数据库：

```postgresql
\c employee
```

### 创建表

```postgresql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

插入数据：

```postgresql
INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com');
```

查询数据：

```postgresql
SELECT * FROM users;
```

### 备注

```
\q
```

```
psql
```