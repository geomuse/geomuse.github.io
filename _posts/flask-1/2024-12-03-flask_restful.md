---
layout: post
title:  flask restful
date:   2024-12-03 11:24:29 +0800
categories: 
    - python 
    - flask
---

https://chatgpt.com/c/674ea567-7680-800f-b656-d0f66a0116cb

**Flask-RESTful** 是一个扩展 Flask 框架的 Python 库，专注于创建简洁而高效的 RESTful API。它对 Flask 提供了一些有用的补充功能，使得开发 REST API 更加直观和高效。

---

### **什么是 RESTful API？**
REST（Representational State Transfer）是一种基于 HTTP 协议的架构风格，常用于开发 API，具有以下特点：
1. **资源导向**：每个 URL 表示一个资源。
2. **使用 HTTP 方法**：
   - `GET`：获取资源。
   - `POST`：创建资源。
   - `PUT`：更新资源。
   - `DELETE`：删除资源。
3. **状态无关性**：每个请求都是独立的，不依赖于之前的请求。
4. **数据格式多样性**：通常使用 JSON 数据格式，但也支持 XML 或纯文本。

---

### **Flask-RESTful 的特点**
1. **简化路由和视图**：通过资源类和方法（如 `get`, `post`, `put`, `delete`）绑定 HTTP 请求。
2. **自动化 JSON 响应**：自动将 Python 对象转换为 JSON 格式。
3. **请求参数解析**：内置参数解析器 `reqparse`，方便处理和验证传入的数据。
4. **错误处理**：提供统一的错误处理机制。
5. **与 Flask 无缝集成**：可以与现有的 Flask 应用一起使用。

---

### **核心概念**
1. **Resource 类**：
   - 表示一个可通过 REST 操作的资源。
   - 使用 `get`, `post`, `put`, `delete` 方法处理不同的 HTTP 请求。

2. **Api 类**：
   - 是 Flask-RESTful 的核心组件。
   - 用于组织和管理资源。

---

### **安装**
```bash
pip install flask-restful
```

---

### **基本用法**

#### **1. 创建一个简单的 RESTful API**
```python
from flask import Flask
from flask_restful import Api, Resource

# 创建 Flask 应用
app = Flask(__name__)
api = Api(app)

# 定义资源类
class HelloWorld(Resource):
    def get(self):
        return {"message": "Hello, World!"}

# 添加资源和路由
api.add_resource(HelloWorld, '/')

if __name__ == "__main__":
    app.run(debug=True)
```

启动后访问 `http://127.0.0.1:5000/`，会返回：
```json
{"message": "Hello, World!"}
```

---

#### **2. 参数解析**
```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('name', type=str, required=True, help="Name cannot be blank!")
parser.add_argument('age', type=int, help="Age must be an integer.")

class HelloWorld(Resource):
    def get(self):
        args = parser.parse_args()
        return {"message": f"Hello, {args['name']}!", "age": args.get("age")}
```
在请求中附加参数：
```bash
curl "http://127.0.0.1:5000/?name=John&age=30"
```

返回：
```json
{"message": "Hello, John!", "age": 30}
```

---

#### **3. 创建和管理多个资源**
```python
class User(Resource):
    def get(self, user_id):
        return {"user_id": user_id}

class Item(Resource):
    def get(self, item_id):
        return {"item_id": item_id}

# 添加多个资源
api.add_resource(User, '/user/<int:user_id>')
api.add_resource(Item, '/item/<int:item_id>')
```
分别访问：
- `http://127.0.0.1:5000/user/1` 返回 `{"user_id": 1}`
- `http://127.0.0.1:5000/item/10` 返回 `{"item_id": 10}`

---

#### **4. 错误处理**
```python
from flask_restful import abort

class User(Resource):
    users = {1: "Alice", 2: "Bob"}
    
    def get(self, user_id):
        if user_id not in self.users:
            abort(404, message="User not found")
        return {"user": self.users[user_id]}

api.add_resource(User, '/user/<int:user_id>')
```
访问不存在的用户时，返回：
```json
{
  "message": "User not found"
}
```

---

### **适用场景**
1. 创建 RESTful 风格的 API 服务。
2. 为前端或第三方提供数据接口。
3. 简化后端 API 的开发流程。
