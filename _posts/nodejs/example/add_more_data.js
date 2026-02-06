const { MongoClient } = require("mongodb");

// 1. 定义连接地址
const url = "mongodb://127.0.0.1:27017/";
const client = new MongoClient(url);

async function run() {
    try {
        // 2. 连接服务器
        await client.connect();
        console.log("已连接数据库");

        // 3. 指定数据库名 (如果不存在，MongoDB 会在你插入数据时自动创建)
        const database = client.db("db");

        // 4. 指定集合名 (类似关系型数据库的表)
        const users = database.collection("users");

        // 5. 准备要插入的数据 (JSON 对象)
        const doc = [{ 
            name : "john",
            age : 30
        },{
            name : "jane",
            age : 25
        },{
            name : "jim",
            age : 35
        },{
            name : "jill",
            age : 28
        }];

        // 6. 执行插入操作
        const result = await users.insertMany(doc);
        
        console.log(`ID 为: ${result.insertedCount}`);

    } catch (err) {
        console.error("操作失败:", err);
    } finally {
        // 7. 关闭连接
        await client.close();
    }
}

run();