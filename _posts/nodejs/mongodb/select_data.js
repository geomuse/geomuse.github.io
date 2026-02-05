const { MongoClient } = require("mongodb");

// 1. 定义连接地址
const url = "mongodb://127.0.0.1:27017/";
const client = new MongoClient(url);

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