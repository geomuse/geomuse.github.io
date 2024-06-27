---
title: flask 从后端学习前端.
author: geo
date: 2024-06-23 14:18:44
---

### Flask 从后端学习前端.

教导写`Flask`,一天就能上手`Flask`,基础教学文章

`Flask`是基于`python` 后端的一个套件 相似有`Django`

`Flask`中小型的网页设计

### 前端

- HTML

```html
<!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<body>
    <h1>This is a Heading</h1>
    <p>This is a paragraph.</p>
</body>
</html>
```

- CSS

```css
body {
  background-color: lightblue;
}

h1 {
  color: white;
  text-align: center;
}

p {
  font-family: verdana;
  font-size: 20px;
}
```

- Js

```js
<button type="button" onclick="document.getElementById('demo').innerHTML = Date()"> Click me to display Date and Time.</button>
```

### 后端

#### 基础

- **Flask简介**
    - 什么是Flask？
    - Flask的特点和优点
    - 安装和设置Flask开发环境

- **第一个Flask应用**
   - 创建一个简单的Flask应用
   - 路由和视图函数
   - 使用Jinja2模板引擎

- **请求与响应**
    - 处理GET和POST请求
    - 读取表单数据
    - 返回响应

- **静态文件**
   - 管理静态文件（CSS、JavaScript、图片等）

- **模板**
   - 使用模板继承和布局

- **提供小型网页设计**

**--- close**

#### **进阶**  

- **Flask扩展**
   - 安装和使用Flask扩展
   - 常用扩展简介（如Flask-WTF、Flask-SQLAlchemy等）

- **数据库与Flask-SQLAlchemy**
    - 配置数据库连接
    - 定义模型和关系
    - 数据库迁移（Flask-Migrate）

- **表单处理与Flask-WTF**
    - 创建和验证表单
    - 处理文件上传

- **用户认证与授权**
   - 用户注册和登录
   - 使用Flask-Login管理用户会话
   - 基于角色的访问控制

- **安全性**
    - 常见Web安全威胁及防护措施
    - 数据加密和保护
    - 安全配置和最佳实践

- **建立中型的网页设计**

**--- unlock**

- **性能优化**
   - 使用缓存提升性能（Flask-Caching）
   - 异步任务处理（Celery与Flask）
   - 优化数据库查询

- **部署与运维**
   - 部署到云服务器（如Heroku、AWS）
   - 使用Docker容器化Flask应用
   - 持续集成与持续部署（CI/CD）

- **高级扩展与集成**
   - 集成第三方服务（如OAuth、支付网关）
   - 自定义Flask扩展


### 教学框架

- Flask 是一个轻量级的`Web`应用框架，适合用于开发小型到中型的应用程序

#### 环境设置

- 在开始之前，需要设置好开发环境。以下是基本步骤：

    - 安装 Python（推荐 Python 3.7 及以上版本）
    创建虚拟环境并激活

```py
python -m venv .venv
source .venv/bin/activate  # 对于 Windows，使用 venv\Scripts\activate
pip install Flask
```

- 后端：Flask 基础
    - 创建第一个 Flask 应用

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
```

- 路由和视图函数

```python
@app.route('/')
def home():
    return "Welcome to the Home Page!"

@app.route('/about')
def about():
    return "This is the About Page."
```
- 模板渲染
    - 在 `templates` 文件夹中创建 `index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask App</title>
</head>
<body>
    <h1>{{ message }}</h1>
</body>
</html>
```

- **修改视图函数以使用模板**

```python
   from flask import render_template

   @app.route('/')
   def hello_world():
       return render_template('index.html', message='Hello, Flask!')
```

- 进阶 :
    - 表单处理

```html
<form action="/submit" method="post">
    <input type="text" name="username" placeholder="Enter your username">
    <button type="submit">Submit</button>
</form>
```

```py
from flask import request

@app.route('/submit', methods=['POST'])
def submit():
    username = request.form['username']
    return f"Hello, {username}!"
```

- 数据库集成

- 前端：与 Flask 集成 on codepen.io
    - 静态文件 html
    - 使用模板引擎
    - css 模板 怎么塞入jinja2
    - 集成前端框架(Bootstrap)?
    - jquery?

- 专案 简单做前端和后端结合.

### 前端

```html
<p>Hello,I'm geo</p>

<p>你好,我是筑</p>
```

### 教导怎么使用html,css

- sass , pug 基本教学

- @mixin

- 前端基础动画互动

- JavaScirpt

- RWD

### 后端

```css
$color : blue
@mixin size($h,$w) :
    height : $h
    width  : $w
```

### 第一次使用`flask`

```bash
pip install Flask
```

```bash
mkdir flask_app
cd flask_app
```

建立在 `app.py` 在根目录

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
```

```bash
python app.py
```

### 模板

```python
@app.route('/')
def home():
    return "Welcome to the Home Page!"

@app.route('/about')
def about():
    return "This is the About Page."
```

Create a `templates` directory and add an `index.html` file Jinja2 :

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home Page</title>
</head>
<body>
    <h1>{{ message }}</h1>
</body>
</html>
```

#### Modify your `app.py` to render this template :

```py
from flask import render_template

@app.route('/')
def home():
    return render_template('index.html', message="Welcome to the Home Page!")
```

```html
<link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
```

### Handling POST Requests.

Create a form in your `templates` directory:

```html
<form action="/submit" method="post">
    <input type="text" name="username" placeholder="Enter your username">
    <button type="submit">Submit</button>
</form>
```

Update your `app.py` to handle form submissions:

```python
from flask import request

@app.route('/submit', methods=['POST'])
def submit():
    username = request.form['username']
    if not username:
        return "Username is required!"
    return f"Hello, {username}!"
```