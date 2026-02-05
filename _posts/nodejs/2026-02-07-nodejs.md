---
layout: post
title:  nodejs mongodb 更新数据
date:   2026-02-07 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

# 更新单笔数据 

```js
const { MongoClient } = require("mongodb");

// 1. 定义连接地址
const url = "mongodb://127.0.0.1:27017/";
const client = new MongoClient(url);

async function run() {
    try {
        await client.connect();
        const database = client.db("db");
        const users = database.collection("users");

        // 1. 定义查询条件（找谁？）
        const filter = { name: "john" };

        // 2. 定义更新动作（改什么？）
        // 使用 $set 确保只修改 name 字段，保留 age 字段
        const updateDoc = {
            $set: { name: "geo" },
        };

        // 3. 执行更新
        const result = await users.updateOne(filter, updateDoc);

        if (result.matchedCount === 0) {
            console.log("未找到名为 john 的用户");
        } else {
            console.log(`成功匹配 ${result.matchedCount} 条数据`);
            console.log(`成功修改 ${result.modifiedCount} 条数据`);
        }

    } catch (err) {
        console.error("更新失败:", err);
    } finally {
        await client.close();
    }
}

run();
```

# 更新多笔

## 方法1

```js
const { MongoClient } = require("mongodb");

// 1. 定义连接地址
const url = "mongodb://127.0.0.1:27017/";
const client = new MongoClient(url);

async function run() {
    try {
        await client.connect();
        const database = client.db("db");
        const users = database.collection("users");

        // 1. 定义查询条件（找谁？）
        const filter = { name: "john" };

        // 2. 定义更新动作（改什么？）
        // 使用 $set 确保只修改 name 字段，保留 age 字段
        const updateDoc = {
            $set: { name: "geo" },
        };

        // 3. 执行更新
        const result = await users.updateOne(filter, updateDoc);

        if (result.matchedCount === 0) {
            console.log("未找到名为 john 的用户");
        } else {
            console.log(`成功匹配 ${result.matchedCount} 条数据`);
            console.log(`成功修改 ${result.modifiedCount} 条数据`);
        }

    } catch (err) {
        console.error("更新失败:", err);
    } finally {
        await client.close();
    }
}

run();
```

## 方法2

```js
async function run() {
    try {
        await client.connect();
        const users = client.db("db").collection("users");

        const operations = [
            {
                updateOne: {
                    filter: { name: "geo" },
                    update: { $set: { name: "mark" } }
                }
            },
            {
                updateOne: {
                    filter: { name: "jane" },
                    update: { $set: { name: "geo" } }
                }
            }
        ];

        const result = await users.bulkWrite(operations);
        console.log(`匹配数: ${result.matchedCount}, 修改数: ${result.modifiedCount}`);

    } catch (err) {
        console.error("批量操作失败:", err);
    } finally {
        await client.close();
    }
}
```