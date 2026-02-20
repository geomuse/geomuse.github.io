---
layout: post
title:  数据库管理mongodb update
date:   2026-02-21 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - mongodb
    - database
---

```js
UPDATE sales SET storeLocation = 'London' WHERE id = '1234567';
```

找到 ID 为 '1234567' 的那条销售记录，并将其门店位置修改为 'London'（伦敦）

```js
db.sales.updateOne(
  { _id: ObjectId("5bd761dcae323e45a93ccff1") },
  { $set: { storeLocation: "London" } }
);
```

我们可以利用`updateOne`

`{ upsert: true }`	更新或插入选项	这是关键点：如果数据库里不存在这个 ID，它会强制插入一条新数据

```js
 db.sales.updateOne(

{ _id: ObjectId("5bd761dcae323e45a93ccab2") },

{ $set: { storeLocation: "London" } },

{ upsert: true }

); 
```