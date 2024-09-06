要将 Flask 应用程序与 MongoDB 数据库连接，通常我们会使用 `PyMongo` 库来简化这一过程。以下是一个从零开始的详细教程，介绍如何将 Flask 应用连接到 MongoDB：

### 步骤 1：安装 Flask 和 PyMongo

首先，你需要确保安装了 Flask 和 `PyMongo`。可以通过以下命令安装：

```bash
pip install Flask
pip install flask-pymongo
```

### 步骤 2：安装 MongoDB

如果你还没有安装 MongoDB，可以在[MongoDB 官方网站](https://www.mongodb.com/try/download/community)下载并安装。安装完 MongoDB 后，启动 MongoDB 服务：

```bash
# 启动 MongoDB 服务
mongod
```

### 步骤 3：创建 Flask 项目

1. 创建一个新的 Flask 项目目录，并在其中创建一个 `app.py` 文件：

   ```bash
   mkdir flask_mongo_project
   cd flask_mongo_project
   touch app.py
   ```

2. 在 `app.py` 文件中，编写基础的 Flask 应用程序并配置 MongoDB 连接：

### 步骤 4：编写代码连接 Flask 与 MongoDB

```python
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.json_util import dumps

app = Flask(__name__)

# 配置 MongoDB 数据库
app.config["MONGO_URI"] = "mongodb://localhost:27017/mydatabase"
mongo = PyMongo(app)

# 创建一个插入数据的路由
@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    name = data['name']
    age = data['age']

    # 插入数据到 MongoDB
    mongo.db.users.insert_one({'name': name, 'age': age})
    
    return jsonify({"message": "User added successfully!"}), 201

# 查询所有用户数据的路由
@app.route('/get_users', methods=['GET'])
def get_users():
    users = mongo.db.users.find()
    return dumps(users)

# 查询单个用户数据
@app.route('/get_user/<name>', methods=['GET'])
def get_user(name):
    user = mongo.db.users.find_one({'name': name})
    if user:
        return dumps(user)
    else:
        return jsonify({"error": "User not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
```

### 代码说明：

1. **MongoDB 连接**：通过 `app.config["MONGO_URI"]` 配置 MongoDB 数据库的 URI，这里使用 `localhost` 作为 MongoDB 的地址，数据库名为 `mydatabase`。
2. **插入数据**：在 `/add_user` 路由中，使用 `mongo.db.users.insert_one()` 向 `users` 集合插入数据。
3. **获取数据**：在 `/get_users` 路由中，使用 `mongo.db.users.find()` 获取所有用户的数据；在 `/get_user/<name>` 中，可以根据用户名查询单个用户数据。

### 步骤 5：启动 Flask 应用

保存 `app.py` 文件后，启动 Flask 应用程序：

```bash
python app.py
```

应用程序将运行在 `http://127.0.0.1:5000/`。

### 步骤 6：测试 API

1. **插入用户数据**：

   使用 `curl` 或 Postman 测试 `POST` 请求：

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"name": "John", "age": 30}' http://127.0.0.1:5000/add_user
   ```

   返回结果：

   ```json
   {
       "message": "User added successfully!"
   }
   ```

2. **获取所有用户数据**：

   ```bash
   curl http://127.0.0.1:5000/get_users
   ```

   返回结果：

   ```json
   [
       {"_id":{"$oid":"..."},"name":"John","age":30}
   ]
   ```

3. **获取单个用户数据**：

   ```bash
   curl http://127.0.0.1:5000/get_user/John
   ```

   返回结果：

   ```json
   {
       "_id": {"$oid":"..."},
       "name": "John",
       "age": 30
   }
   ```

### 总结

通过这个教程，你可以轻松将 Flask 与 MongoDB 连接并实现基本的数据库操作。