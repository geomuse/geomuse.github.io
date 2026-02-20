---
layout: post
title:  数据库管理mongodb connnect with python
date:   2026-02-24 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - mongodb
    - database
---

```py
from pymongo import MongoClient 
url = "mongodb://127.0.0.1:27017/"
client = MongoClient(url)

if __name__ == '__main__' : 

    for db_name in client.list_database_names():
        print(db_name)
```