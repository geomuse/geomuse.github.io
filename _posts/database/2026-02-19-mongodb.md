---
layout: post
title:  数据库管理mongodb select
date:   2026-02-19 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - mongodb
    - database
---

涵盖了如何为您的应用程序用户创建搜索体验 , 并指导您了解聚合、索引、数据建模和事务等关键主题

# select 

```js
select * from zips limt 1 ; 
```

和

```js
select city , state from zips limit 1 ; 
```

完整说明

```py
db.collection.findOne(filter,projection,options)
```

简短

```py
db.zips.findOne({})
```

回传`json`格式数据

```js
select city from zips where state = 'AZ' and pop < 500 ; 
```

```py
db.zips.find({state: 'AZ', pop: {$lt: 500}})
```

```js
EXPLAIN SELECT city, state, pop 
            FROM zips 
WHERE state = 'NY' AND pop BETWEEN 1000 AND 5000 
ORDER BY pop DESC 
LIMIT 10;
```

```py
db.zips.explain().find(
    { state: "NY", pop: { $gte: 1000, $lte: 5000 }}, 
    {_id: 0, state: 1, city: 1, pop: 1}
    ).sort({pop: -1})
    .limit(10)
```