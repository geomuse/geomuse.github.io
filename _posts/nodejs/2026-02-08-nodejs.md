---
layout: post
title:  nodejs mongodb 删除数据
date:   2026-02-08 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

```js
async function run() {
    try {
        await client.connect();
        const users = client.db("db").collection("users");

        // 删除名字为 "jane" 的文档
        const result = await users.deleteOne({ name: "jane" });

        if (result.deletedCount === 1) {
            console.log("成功删除了 jane");
        } else {
            console.log("未找到名为 jane 的文档，删除失败");
        }

        // 验证：查看剩下的所有人
        const remainingUsers = await users.find().toArray();
        console.log("剩余用户：", remainingUsers);

    } catch (err) {
        console.error("删除操作出错:", err);
    } finally {
        await client.close();
    }
}
```