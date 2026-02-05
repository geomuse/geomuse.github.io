---
layout: post
title:  nodejs 快速入门以及连接mongodb
date:   2026-02-05 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

# nodejs

下载nodejs

```bash
sudo apt install nodejs
```

# npm 

下载第三方套件库 

```bash
sudo apt install npm
```

# mongodb

下载mongodb composs GUI , MongoDB Community Server 和 npm

```bash
npm install mongodb
```

## 旧版写法 

```js
console.log("Hello Nodejs");

const mongo = require("mongodb") ;  
const client = mongo.MongoClient("mongodb://localhost:27017/"); // 建立客户端物件

client.connect(async function(err, db) {
    if (err) {
        console.log(err);
        return ; 
    } else {
        console.log("Connected to MongoDB");
        client.close();
    }
});
```

## 新版写法

```js
const { MongoClient } = require("mongodb");

// 建议使用 127.0.0.1 代替 localhost 避免某些环境下的 DNS 解析延迟
const url = "mongodb://127.0.0.1:27017/";
const client = new MongoClient(url);

async function run() {
    try {
        // 连接服务器
        await client.connect();
        console.log("成功连接到 MongoDB 服务器！");
        
        // 选择数据库
        const adminDb = client.db("admin");
        const info = await adminDb.command({ ping: 1 });
        console.log("服务器响应：", info);
        
    } catch (err) {
        console.error("连接失败：", err);
    } finally {
        // 确保最后关闭连接
        await client.close();
    }
}

run();
```