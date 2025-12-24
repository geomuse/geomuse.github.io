---
layout: post
title:  数据库管理postgresql 上手
date:   2025-10-05 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - postgresql
---

cmd 上输入进入数据库管理

```bash
psql -U postgres
```

如果看到 `postgres=#` 提示符，说明连接成功。输入 `\q` 退出

密码是123

```bash
CREATE DATABASE testdb;
CREATE TABLE users(id SERIAL PRIMARY KEY, name TEXT, age INT);
INSERT INTO users(name, age) VALUES ('Alice', 25);
SELECT * FROM users;
```