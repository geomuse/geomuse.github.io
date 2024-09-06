---
layout: post
title:  02/30-flask 与 mongodb连接
date:   2024-09-05 11:24:29 +0800
categories: 
    - python 
    - flask
    - challenge
---

要在 Flask 应用中连接 MongoDB，你可以使用 `PyMongo`，这是一个简单的库，它提供了与 MongoDB 交互的功能。以下是如何设置 Flask 和 MongoDB 连接的教程。

### 步骤 1: 安装依赖

首先，确保安装了 `flask` 和 `pymongo` 依赖库。你可以通过 pip 来安装这些库：

```bash
pip install Flask pymongo
```

### 步骤 2: 设置 MongoDB 连接

接下来，编写 Flask 应用代码，设置与 MongoDB 的连接。假设你已经在本地或者远程服务器上运行 MongoDB 实例。

```python
from flask import Flask, jsonify, request
from pymongo import MongoClient

app = Flask(__name__)

# 连接 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['test_database']  # 使用 'test_database' 数据库

# 示例集合（类似于表）
collection = db['test_collection']

# 插入数据的路由
@app.route('/insert', methods=['POST'])
def insert_data():
    data = request.json
    collection.insert_one(data)  # 将 JSON 数据插入到 MongoDB
    return jsonify({"message": "数据已插入"}), 201

# 获取数据的路由
@app.route('/data', methods=['GET'])
def get_data():
    data = list(collection.find())  # 从 MongoDB 中获取数据
    for item in data:
        item['_id'] = str(item['_id'])  # 将 ObjectId 转换为字符串
    return jsonify(data), 200

if __name__ == '__main__':
    app.run(debug=True)
```

### 步骤 3: 运行应用

1. 确保 MongoDB 已启动。
2. 运行 Flask 应用程序：

```bash
python app.py
```

Flask 应用将运行在 `http://127.0.0.1:5000/` 上。你可以使用 Postman 或 curl 向 `/insert` 路由发送 POST 请求来插入数据，向 `/data` 路由发送 GET 请求来获取数据。

### 步骤 4: 测试插入和获取数据

使用以下命令测试插入数据：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"name": "Alice", "age": 30}' http://127.0.0.1:5000/insert
```

测试获取数据：

```bash
curl http://127.0.0.1:5000/data
```

### 注意事项
- 确保你的 MongoDB 连接字符串是正确的，尤其是在连接到远程数据库时，可能需要包含用户名和密码。
- 可以通过 MongoDB Atlas 来设置云端 MongoDB 数据库，连接方式相似。

这样就能成功将 Flask 应用与 MongoDB 连接了！