---
layout: post
title:  flask RESTful API
date:   2024-11-26 11:24:29 +0800
categories: 
    - python 
    - flask
---

1. **RESTful API 概念**：
   - **API（Application Programming Interface）**：应用程序接口，是客户端与服务器之间的通信方式。
   - **REST（Representational State Transfer）**：
     - 使用 HTTP 协议。
     - 基于资源的操作（如 GET、POST、PUT、DELETE）。
     - 响应以 JSON 格式为主，便于客户端解析。

2. **Flask API 路由**：
   - Flask 提供简单的路由定义，可以用作构建 RESTful API。
   - 使用 `jsonify` 方法返回 JSON 格式的响应。

### **创建简单的 API**

**示例代码：简单的 Flask API**
```py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api", methods=["GET"])
def api():
    data = {
        "message": "Welcome to Flask API!",
        "status": "success"
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
```

#### **效果**：
- 访问 `/api` 返回以下 JSON 数据：
  ```json
  {
    "message": "Welcome to Flask API!",
    "status": "success"
  }
  ```

---

### **API 示例：获取用户数据**

**示例代码：返回用户数据的 API**
```py
from flask import Flask, jsonify

app = Flask(__name__)

# 模拟用户数据
users = [
    {"id": 1, "username": "Alice", "email": "alice@example.com"},
    {"id": 2, "username": "Bob", "email": "bob@example.com"},
    {"id": 3, "username": "Charlie", "email": "charlie@example.com"}
]

@app.route("/api/users", methods=["GET"])
def get_users():
    return jsonify(users)

@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if user:
        return jsonify(user)
    else:
        return jsonify({"error": "User not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
```

#### **效果**：
1. 访问 `/api/users` 返回所有用户数据：
   ```json
   [
       {"id": 1, "username": "Alice", "email": "alice@example.com"},
       {"id": 2, "username": "Bob", "email": "bob@example.com"},
       {"id": 3, "username": "Charlie", "email": "charlie@example.com"}
   ]
   ```
2. 访问 `/api/users/1` 返回用户 ID 为 1 的数据：
   ```json
   {
       "id": 1,
       "username": "Alice",
       "email": "alice@example.com"
   }
   ```
3. 访问 `/api/users/99` 返回 404 错误：
   ```json
   {
       "error": "User not found"
   }
   ```

---

### **创建和更新数据**

扩展 API，支持添加用户和更新用户数据。

#### **示例代码：支持 POST 和 PUT 请求**
```py
from flask import Flask, jsonify, request

app = Flask(__name__)

users = [
    {"id": 1, "username": "Alice", "email": "alice@example.com"},
    {"id": 2, "username": "Bob", "email": "bob@example.com"}
]

# 添加用户
@app.route("/api/users", methods=["POST"])
def add_user():
    new_user = request.json  # 从请求体获取 JSON 数据
    new_user["id"] = len(users) + 1  # 自动分配 ID
    users.append(new_user)
    return jsonify(new_user), 201

# 更新用户
@app.route("/api/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return jsonify({"error": "User not found"}), 404
    data = request.json
    user.update(data)
    return jsonify(user)

if __name__ == "__main__":
    app.run(debug=True)
```

#### **效果**：
1. **添加用户**：
   - 请求：`POST /api/users`
   - 请求体：
     ```json
     {"username": "Charlie", "email": "charlie@example.com"}
     ```
   - 响应：
     ```json
     {"id": 3, "username": "Charlie", "email": "charlie@example.com"}
     ```

2. **更新用户**：
   - 请求：`PUT /api/users/1`
   - 请求体：
     ```json
     {"email": "newalice@example.com"}
     ```
   - 响应：
     ```json
     {"id": 1, "username": "Alice", "email": "newalice@example.com"}
     ```

---

### **练习**

#### **任务：**
1. 实现一个 API 路由 `/api/posts` 返回以下博客文章的列表：
   ```json
   [
       {"id": 1, "title": "Flask Basics", "author": "Alice"},
       {"id": 2, "title": "Understanding APIs", "author": "Bob"}
   ]
   ```
2. 实现一个 API 路由 `/api/posts/<int:post_id>` 返回特定文章数据。

---

### **练习答案**

```py
@app.route("/api/posts", methods=["GET"])
def get_posts():
    posts = [
        {"id": 1, "title": "Flask Basics", "author": "Alice"},
        {"id": 2, "title": "Understanding APIs", "author": "Bob"}
    ]
    return jsonify(posts)

@app.route("/api/posts/<int:post_id>", methods=["GET"])
def get_post(post_id):
    posts = [
        {"id": 1, "title": "Flask Basics", "author": "Alice"},
        {"id": 2, "title": "Understanding APIs", "author": "Bob"}
    ]
    post = next((p for p in posts if p["id"] == post_id), None)
    if post:
        return jsonify(post)
    else:
        return jsonify({"error": "Post not found"}), 404
```

访问 `/api/posts` 和 `/api/posts/1` 即可查看结果。

---

### **总结**
1. RESTful API 使用 HTTP 动词（如 GET、POST、PUT、DELETE）操作资源。
2. Flask 提供了便捷的 `jsonify` 方法和路由支持，快速构建 API。
3. 使用 `request.json` 获取客户端发送的 JSON 数据，动态处理数据。

明天将学习如何通过 **Flask-RESTful** 提供更结构化的 API 开发工具集。