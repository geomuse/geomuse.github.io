---
layout: post
title:  flask mongodb crud
date:   2024-09-14 11:24:29 +0800
categories: 
    - python 
    - mongodb
    - flask
---

### 设定数据与代码做结合

```py
from flask import Flask, jsonify, request
from pymongo import MongoClient

app = Flask(__name__)

# 连接到 MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["recipe"]  # 创建/选择数据库
collection = db["info"]  # 创建/选择集合
```

### 主页

```py
# 主页路由
@app.route('/')
def home():
    return "Welcome to the Flask MongoDB app!"
```

### 添加数据

```py
# 插入数据到 MongoDB
@app.route('/add', methods=['POST'])
# curl -X POST http://127.0.0.1:5000/add -H "Content-Type: application/json" -d '{"name": "Alice", "age": 30}'
def add_data():
    data = request.json
    collection.insert_one(data)
    return jsonify({"message": "Data inserted successfully!"}), 201
```

### 资料库

```py
# 获取数据
@app.route('/data', methods=['GET'])
def get_data():
    data = list(collection.find({}, {"_id": 0}))  # 不返回 _id 字段
    return jsonify(data), 200
```

### 更新数据

```py
@app.route('/update/<name>', methods=['PUT'])
def update_data(name):
    data = request.json
    collection.update_one({"name": name}, {"$set": data})
    return jsonify({"message": "Data updated successfully!"}), 200
```

### 删除数据

```py
@app.route('/delete/<name>', methods=['DELETE'])
def delete_data(name):
    collection.delete_one({"name": name})
    return jsonify({"message": "Data deleted successfully!"}), 200
```

### 发送数据 curl

```bash
curl -X POST http://127.0.0.1:5000/add -H "Content-Type: application/json" -d '{"name": "Alice", "age": 30}'

```

```bash
curl -X POST http://127.0.0.1:5000/add -H "Content-Type: application/json" -d @data.json
```


### 检查data内容

```bash
curl http://127.0.0.1:5000/data
```