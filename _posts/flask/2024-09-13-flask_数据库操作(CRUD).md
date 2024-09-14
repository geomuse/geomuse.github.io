---
layout: post
title:  challenge-09/30-flask 数据库操作（CRUD)
date:   2024-09-13 11:24:29 +0800
categories: 
    - python 
    - flask
    - challenge
---

了解如何在 Flask 应用中进行数据库的 CRUD（创建、读取、更新、删除）操作是开发动态 Web 应用的基础。以下是一个详细的教程，指导你如何使用 Flask 和 Flask-SQLAlchemy 实现这些操作。

### 前提条件

假设你已经按照之前的教程完成了 Flask 项目和数据库的基本设置，包括 `app.py`、`config.py` 和 `models.py` 文件，并且已经创建了数据库。以下内容将在此基础上进行扩展。

### 项目结构

```
your_project/
│
├── app.py
├── config.py
├── models.py
├── requirements.txt
└── app.db
```

### 安装必要的库

确保已安装 Flask 和 Flask-SQLAlchemy：

```bash
pip install Flask Flask-SQLAlchemy
```

### 数据库模型 (`models.py`)

以用户（User）模型为例：

```python
from flask_sqlalchemy import SQLAlchemy
from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'
```

### 创建数据库

在 Python 交互环境中运行以下命令以创建数据库和表：

```python
from app import db
db.create_all()
```

### CRUD 操作实现

#### 1. 创建（Create）

添加新用户的功能。

```python
from flask import Flask, request, jsonify
from models import User
from app import db

@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    if not data or not 'username' in data or not 'email' in data:
        return jsonify({'message': '缺少用户名或电子邮件'}), 400

    username = data['username']
    email = data['email']

    # 检查用户是否已存在
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({'message': '用户已存在'}), 400

    new_user = User(username=username, email=email)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': f'用户 {username} 添加成功!'}), 201
```

**测试方法：**

使用 Postman 或 cURL 发送 POST 请求：

```bash
curl -X POST http://localhost:5000/add_user \
     -H "Content-Type: application/json" \
     -d '{"username":"john_doe", "email":"john@example.com"}'
```

#### 2. 读取（Read）

##### 2.1 获取所有用户

```python
@app.route('/get_users', methods=['GET'])
def get_users():
    users = User.query.all()
    output = []
    for user in users:
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
        output.append(user_data)
    return jsonify({'users': output}), 200
```

##### 2.2 根据 ID 获取单个用户

```python
@app.route('/get_user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': '用户未找到'}), 404

    user_data = {
        'id': user.id,
        'username': user.username,
        'email': user.email
    }
    return jsonify({'user': user_data}), 200
```

**测试方法：**

- 获取所有用户：

  ```bash
  curl http://localhost:5000/get_users
  ```

- 获取特定用户（例如 ID 为 1）：

  ```bash
  curl http://localhost:5000/get_user/1
  ```

#### 3. 更新（Update）

更新现有用户的信息。

```python
@app.route('/update_user/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': '用户未找到'}), 404

    data = request.get_json()
    if 'username' in data:
        user.username = data['username']
    if 'email' in data:
        user.email = data['email']

    db.session.commit()
    return jsonify({'message': f'用户 {user.id} 更新成功!'}), 200
```

**测试方法：**

使用 Postman 或 cURL 发送 PUT 请求：

```bash
curl -X PUT http://localhost:5000/update_user/1 \
     -H "Content-Type: application/json" \
     -d '{"username":"jane_doe", "email":"jane@example.com"}'
```

#### 4. 删除（Delete）

删除指定用户。

```python
@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': '用户未找到'}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': f'用户 {user.id} 删除成功!'}), 200
```

**测试方法：**

使用 Postman 或 cURL 发送 DELETE 请求：

```bash
curl -X DELETE http://localhost:5000/delete_user/1
```

### 完整的 `app.py` 示例

将所有 CRUD 路由整合到 `app.py` 中：

```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('config')

db = SQLAlchemy(app)

# 导入模型
from models import User

# 创建用户
@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    if not data or not 'username' in data or not 'email' in data:
        return jsonify({'message': '缺少用户名或电子邮件'}), 400

    username = data['username']
    email = data['email']

    # 检查用户是否已存在
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({'message': '用户已存在'}), 400

    new_user = User(username=username, email=email)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': f'用户 {username} 添加成功!'}), 201

# 获取所有用户
@app.route('/get_users', methods=['GET'])
def get_users():
    users = User.query.all()
    output = []
    for user in users:
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
        output.append(user_data)
    return jsonify({'users': output}), 200

# 获取单个用户
@app.route('/get_user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': '用户未找到'}), 404

    user_data = {
        'id': user.id,
        'username': user.username,
        'email': user.email
    }
    return jsonify({'user': user_data}), 200

# 更新用户
@app.route('/update_user/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': '用户未找到'}), 404

    data = request.get_json()
    if 'username' in data:
        user.username = data['username']
    if 'email' in data:
        user.email = data['email']

    db.session.commit()
    return jsonify({'message': f'用户 {user.id} 更新成功!'}), 200

# 删除用户
@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': '用户未找到'}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': f'用户 {user.id} 删除成功!'}), 200

# 首页
@app.route('/')
def index():
    return "欢迎使用 Flask 数据库 CRUD 教程!"

if __name__ == '__main__':
    app.run(debug=True)
```

### 运行应用

启动 Flask 应用：

```bash
python app.py
```

应用将在默认的 `http://localhost:5000` 运行。

### 测试 CRUD 操作

你可以使用 **Postman** 或 **cURL** 来测试各个 API 路由。

#### 示例操作

1. **创建用户**

   ```bash
   curl -X POST http://localhost:5000/add_user \
        -H "Content-Type: application/json" \
        -d '{"username":"alice", "email":"alice@example.com"}'
   ```

2. **读取所有用户**

   ```bash
   curl http://localhost:5000/get_users
   ```

3. **读取单个用户**

   ```bash
   curl http://localhost:5000/get_user/1
   ```

4. **更新用户**

   ```bash
   curl -X PUT http://localhost:5000/update_user/1 \
        -H "Content-Type: application/json" \
        -d '{"username":"alice_updated", "email":"alice_new@example.com"}'
   ```

5. **删除用户**

   ```bash
   curl -X DELETE http://localhost:5000/delete_user/1
   ```

### 使用 Postman 进行测试

1. **安装 Postman**：[下载地址](https://www.postman.com/downloads/)

2. **创建请求**

   - **POST** `/add_user`：在 Body 选择 `raw` 和 `JSON`，输入用户数据。
   - **GET** `/get_users` 和 `/get_user/<id>`：直接发送 GET 请求。
   - **PUT** `/update_user/<id>`：在 Body 选择 `raw` 和 `JSON`，输入要更新的数据。
   - **DELETE** `/delete_user/<id>`：发送 DELETE 请求。

### 总结

通过以上步骤，你已经在 Flask 应用中实现了基本的 CRUD 操作。这些操作涵盖了创建、读取、更新和删除数据库中的记录。你可以根据需求扩展这些功能，例如添加用户认证、使用更复杂的查询或集成其他数据库（如 MySQL、PostgreSQL 等）。

### 进一步学习

- **验证和错误处理**：使用 Flask-WTF 或 Marshmallow 进行数据验证。
- **分页和过滤**：为读取操作添加分页和过滤功能。
- **用户认证**：集成 Flask-Login 或 Flask-JWT 实现用户认证和授权。
- **前端集成**：使用前端框架（如 React、Vue）与 Flask API 进行交互。

如果你在实现过程中遇到任何问题，欢迎随时提问！