---
layout: post
title:  数据库管理mongodb crud
date:   2026-02-25 09:01:00 +0800
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

db = client.db

acccount_collection = db.account

new_account = {
    "account_holder" : "Linu" , 
    "balance" : 50352434
}

if __name__ == '__main__' : 
    
    result = acccount_collection.insert_one(new_account)
    print(result.inserted_id)

    client.close()
```

`insert_many` 雷同

其他有用到再从`AI`上面获取