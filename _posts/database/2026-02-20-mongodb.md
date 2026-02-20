---
layout: post
title:  数据库管理mongodb between , order by , limit
date:   2026-02-20 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - mongodb
    - database
---

```js
SELECT city, state, pop 
FROM zips 
WHERE state = 'NY' AND pop BETWEEN 1000 AND 5000 
ORDER BY pop DESC 
LIMIT 10;
```

```py
db.zips.find(
    { state: "NY", pop: { $gte: 1000, $lte: 5000 }}
    ).sort({pop: -1})
    .limit(10)
```

也可以用

```js
db.zips.find(
    { state: "NY", pop: { $gte: 1000, $lte: 5000 }}, 
    {_id: 0, state: 1, city: 1, pop: 1}
    ).sort({pop: -1})
    .limit(10)
```

```js
{_id: 0, state: 1, city: 1, pop: 1}

SELECT state, city, pop
```