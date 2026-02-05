---
layout: post
title:  nodejs mongodb 查询数据
date:   2026-02-09 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

# `findOne`：直接拿结果

`findOne` 就像去超市买特定的一个苹果，拿完就走。它直接返回一个 **JavaScript 对象**（如果没找到则返回 `null`）。

```javascript
const user = await users.findOne({ name: "geo" });
console.log(user.age); // 直接读取属性
```

```js
async function run() {
    try {
        await client.connect();
        const users = client.db("db").collection("users");

        //寻找名字为 "geo" 的文档
        const user = await users.findOne({ name: "geo" });

        if (user) {
            console.log("查询结果：", user);
            console.log("Geo 的年龄是：", user.age);
        } else {
            console.log("找不到名为 geo 的用户");
        }
        // const all = await users.find().toArray();
        // console.log("所有数据：", all);

    } catch (err) {
        console.error("查询出错:", err);
    } finally {
        await client.close();
    }
}

run(); 
```

---

# `find` 如何从游标取数据？

## 方式 A：`toArray()`（最常用，适合小数据量）

如果你确定数据量不大（比如几百条），用 `toArray()` 最方便。它会等待所有数据传输完毕并打包成一个数组。

```javascript
const cursor = users.find({ age: { $gt: 20 } });
const results = await cursor.toArray(); 
console.log(results[0]); // 这是一个标准的数组
```

## 方式 B：`forEach` 迭代（推荐，适合大数据量）

如果你要处理 10 万条数据，你应该用迭代的方法，这样内存占用极低：

```javascript
const cursor = users.find({ age: { $gt: 20 } });

await cursor.forEach(doc => {
    console.log("正在处理：", doc.name);
    // 处理完这一条，内存就会释放这一条，再去拿下一条
});
```

| 操作符 | 含义 | 英文全称 | 示例 |
| --- | --- | --- | --- |
| **`$gt`** | **>** 大于 | Greater Than | `{ age: { $gt: 20 } }` |
| **`$gte`** | **>=** 大于等于 | Greater Than or Equal | `{ age: { $gte: 20 } }` |
| **`$lt`** | **<** 小于 | Less Than | `{ age: { $lt: 30 } }` |
| **`$lte`** | **<=** 小于等于 | Less Than or Equal | `{ age: { $lte: 30 } }` |
| **`$ne`** | **!=** 不等于 | Not Equal | `{ age: { $ne: 20 } }` |


## 方式 C : 寻找全部数据

```js
const { MongoClient } = require("mongodb");

// 1. 定义连接地址
const url = "mongodb://127.0.0.1:27017/";
const client = new MongoClient(url);

async function run() {
    try {
        await client.connect();
        const users = client.db("db").collection("users");

        // 寻找名字为 "geo" 的文档
        // const user = await users.findOne({ name: "geo" });

        // if (user) {
        //     console.log("查询结果：", user);
        //     console.log("Geo 的年龄是：", user.age);
        // } else {
        //     console.log("找不到名为 geo 的用户");
        // }
        const all = await users.find().toArray();
        console.log("所有数据：", all);

    } catch (err) {
        console.error("查询出错:", err);
    } finally {
        await client.close();
    }
}

run(); 
```

---

## 常见的查询辅助方法（必须配合 find）

游标还允许你在数据传输之前进行“加工”：

* **排序 (`sort`)**：`1` 为升序，`-1` 为降序。
* **限制 (`limit`)**：只取前几条。
* **跳过 (`skip`)**：跳过前几条（常用于分页）。

```javascript
const results = await users.find()
    .sort({ age: -1 })  // 按年龄从大到小排
    .skip(5)            // 跳过前 5 条
    .limit(10)          // 只取 10 条
    .toArray();
```