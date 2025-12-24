---
layout: post
title:  数据库管理postgresql 查询数据 python
date:   2025-10-07 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - postgresql
---

```sql
CREATE DATABASE testdb;
```

```py
import psycopg2

conn = psycopg2.connect(
    dbname="demodb",
    user="postgres",
    password="123",
    host="localhost",
    port="5432"
)

cur = conn.cursor()
```

设定数据库

```py
cur.execute("SELECT version();")
print(cur.fetchone())
cur.close()
conn.close()
```

查询数据库版本