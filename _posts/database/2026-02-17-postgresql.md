---
layout: post
title:  数据库管理postgresql 删除数据库
date:   2026-02-17 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - postgresql
    - database
---

删除数据库

```bash
DROP DATABASE 数据库名;
DROP DATABASE IF EXISTS 数据库名;
drop database testdb ; 
```

删除`table`

```bash
drop table users ; 
```

删除表格里的数据

```bash
DELETE FROM users 
WHERE id = (
    SELECT id FROM users 
    WHERE name = 'alice' 
    LIMIT 1
);

```