---
layout: post
title:  数据库管理postgresql 上手
date:   2025-10-05 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - database
    - postgresql
---

cmd 上输入进入数据库管理

```bash
sudo -i -u postgres psql
```

或者 

```bash
psql -U postgres
```

如果看到 `postgres=#` 提示符，说明连接成功。输入 `\q` 退出

密码是123

命令 |	作用
:-: | :-:
\l |	列出所有数据库 (List)
\c | 数据库名	切换/连接到指定数据库 (Connect)
\dt |	列出当前数据库中所有的表 (Describe Tables)
\d |表名	查看某个表的具体结构（列名、类型、索引）
\du	|列出所有角色/用户
\q	|退出 psql (Quit)

```bash
CREATE DATABASE testdb;
CREATE TABLE users(id SERIAL PRIMARY KEY, name TEXT, age INT);
INSERT INTO users(name, age) VALUES ('Alice', 25);
SELECT * FROM users;
```