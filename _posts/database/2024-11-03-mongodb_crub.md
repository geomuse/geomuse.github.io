---
layout: post
title : mongodb crub
date : 2024-11-03 11:24:29 +0800
categories: 
    - mongodb
---

```py
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['fundamental']
collection = db['statment']
```

```py
# 插入单条数据
document = {"name": "Alice", "age": 25, "city": "New York"}
collection.insert_one(document)

# 插入多条数据
documents = [
    {"name": "Bob", "age": 30, "city": "Chicago"},
    {"name": "Charlie", "age": 35, "city": "San Francisco"}
]
collection.insert_many(documents)
```

```py
# 读取单条数据
result = collection.find_one({"name": "Alice"})
print(result)
```

```py
# 读取所有数据
results = collection.find()
for doc in results:
    print(doc)
```

```py
# 条件查询数据
age_query = {"age": {"$gt": 25}}
results = collection.find(age_query)
for doc in results:
    print(doc)
```

```py
# 更新单条数据
collection.update_one({"name": "Alice"}, {"$set": {"age": 26}})

# 更新多条数据
collection.update_many({"age": {"$gt": 30}}, {"$set": {"status": "senior"}})
```

```py
# 删除单条数据
collection.delete_one({"name": "Charlie"})

# 删除多条数据
collection.delete_many({"age": {"$lt": 30}})
```

https://learn.mongodb.com/pages/mongodb-associate-developer-exam