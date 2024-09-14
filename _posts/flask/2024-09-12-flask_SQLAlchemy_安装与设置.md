---
layout: post
title:  challenge-08/30-Flask-SQLAlchemy 安装与设置
date:   2024-09-12 11:24:29 +0800
categories: 
    - python 
    - flask
    - challenge
---

要在 Flask 中建立资料库并进行基本的数据库操作，你可以使用 Flask-SQLAlchemy 作为 ORM（对象关系映射），这使得操作数据库变得更加直观和简便。以下是一个基本的 Flask 项目设置和数据库操作的教程：

### 安装 Flask 和 Flask-SQLAlchemy
首先，你需要安装 Flask 和 Flask-SQLAlchemy。

```bash
pip install Flask Flask-SQLAlchemy
```

### 第一步：建立 Flask 项目

1. 创建一个新的项目文件夹，并创建以下文件：
   - `app.py`：主程序文件
   - `config.py`：配置文件
   - `models.py`：数据库模型

2. 创建 `app.py` 文件，内容如下：

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('config')

db = SQLAlchemy(app)

# 导入模型
from models import User

@app.route('/')
def index():
    return "Welcome to the Flask database tutorial"

if __name__ == '__main__':
    app.run(debug=True)
```

### 第二步：设置数据库配置

3. 在项目根目录下创建 `config.py` 文件，用于存储数据库连接配置。

```python
import os

# 设置数据库连接 URI，使用 SQLite 数据库
basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
```

### 第三步：定义数据库模型

4. 在 `models.py` 文件中定义数据库模型。以用户（User）模型为例：

```python
from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'
```

### 第四步：创建数据库

5. 在命令行中运行以下命令来创建数据库：

```bash
from app import db
db.create_all()
```

这将会在你的项目文件夹中创建 `app.db` 文件，它是你的 SQLite 数据库。

### 第五步：执行基本的数据库操作

6. 在 `app.py` 文件中添加一些视图函数来执行基本的数据库操作，例如添加、查询和删除用户。

```python
from flask import Flask, request
from models import User

@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form['username']
    email = request.form['email']
    new_user = User(username=username, email=email)
    db.session.add(new_user)
    db.session.commit()
    return f"User {username} added successfully!"

@app.route('/get_users', methods=['GET'])
def get_users():
    users = User.query.all()
    return {user.id: user.username for user in users}
```

### 第六步：运行应用

7. 启动 Flask 应用：

```bash
python app.py
```

8. 使用 Postman 或 cURL 测试 `POST /add_user` 和 `GET /get_users` 路由来添加和查看用户。

### 总结

你已经学会了如何在 Flask 中设置数据库，并进行基本的操作。通过使用 Flask-SQLAlchemy，你可以轻松地对数据库进行管理，并将 Flask 与各种数据库相集成，如 MySQL、PostgreSQL 或 SQLite。

如果你需要更深入的教程或代码示例，请告诉我！