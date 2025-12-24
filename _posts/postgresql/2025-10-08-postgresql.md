---
layout: post
title:  数据库管理postgresql 查询数据 python
date:   2025-10-08 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - postgresql
---

```py
import psycopg2

conn = psycopg2.connect(
    dbname="demodb",
    user="postgres",
    password="123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
```

设定数据库

```py
select_all_query = "SELECT * FROM users;"

cursor.execute(select_all_query)

all_users = cursor.fetchall()

for row in all_users:
    print(row)
```

查询数据