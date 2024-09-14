---
layout: post
title:  challenge-11/30-Flask-Login 用户认证
date:   2024-09-15 11:24:29 +0800
categories: 
    - python 
    - flask
    - challenge
---

Flask-Login 是一个非常流行的 Flask 扩展，专门用于用户认证和管理用户登录状态。下面是一个基本的指南，帮助你学习如何使用 Flask-Login 实现用户登录、登出和访问保护。

### 步骤 1：安装 Flask-Login
你首先需要安装 Flask-Login 扩展，可以通过 pip 进行安装：
```bash
pip install flask-login
```

### 步骤 2：设置 Flask 应用

1. **创建 Flask 应用**
   你需要创建一个基本的 Flask 应用程序结构，确保应用有用户注册、登录和登出功能。
   
2. **设置 User 模型**
   Flask-Login 需要一个 `User` 类或模型来管理用户数据。该类需要实现一些必要的方法，如 `is_authenticated`、`is_active`、`is_anonymous` 和 `get_id`。通常，这个类会对应到你的用户数据库模型。

### 步骤 3：基本的 Flask-Login 设置

```python
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于保护 session

# Flask-Login 初始化
login_manager = LoginManager()
login_manager.init_app(app)

# 登录视图，如果用户没有登录，会重定向到此
login_manager.login_view = 'login'


# 用户类，继承 UserMixin，包含 Flask-Login 需要的属性和方法
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# 模拟用户数据库
users = {
    'user1': User(1, 'user1', 'password1'),
    'user2': User(2, 'user2', 'password2')
}


# 加载用户的回调函数
@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 检查用户是否在数据库中
        user = users.get(username)
        if user and user.password == password:
            login_user(user)  # 使用 login_user() 方法登录用户
            return redirect(url_for('protected'))
        return 'Invalid username or password'
    
    return render_template('login.html')


# 受保护的路由，只有登录后才能访问
@app.route('/protected')
@login_required
def protected():
    return f'Hello, {current_user.username}! You are logged in.'

# 登出路由
@app.route('/logout')
@login_required
def logout():
    logout_user()  # 使用 logout_user() 方法登出用户
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
```

### 解释：

1. **LoginManager 初始化**：`LoginManager` 是 Flask-Login 的核心，它会管理用户的登录状态。`login_view` 用于定义用户未登录时重定向的视图。

2. **User 类**：模拟了一个简单的用户类，它继承了 `UserMixin`，该类包含了 Flask-Login 要求的用户属性和方法。通常情况下，你会将此类替换为一个数据库模型类。

3. **login_user() 和 logout_user()**：`login_user(user)` 用于登录用户，`logout_user()` 用于登出用户。

4. **@login_required**：这个装饰器用于保护视图，只有登录的用户才能访问这些路由。

5. **user_loader**：`@login_manager.user_loader` 回调函数用来从数据库中加载用户，接受的参数是用户的 ID。

### 步骤 4：创建登录表单

你可以创建一个简单的 HTML 登录表单 `login.html`：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
</head>
<body>
    <form method="POST">
        <label for="username">Username:</label>
        <input type="text" name="username" id="username" required><br>
        <label for="password">Password:</label>
        <input type="password" name="password" id="password" required><br>
        <input type="submit" value="Login">
    </form>
</body>
</html>
```

### 步骤 5：保护路由
通过使用 `@login_required` 装饰器，可以轻松保护路由，使其只有在用户登录后才可以访问。如果用户未登录，系统会将其重定向到登录页面。

### 总结
通过以上步骤，你已经成功实现了用户认证功能，包括登录、登出和保护路由。如果你有数据库支持，还可以将 `User` 类替换为数据库中的模型，并进行更复杂的用户管理。

你可以根据需求进一步扩展，比如增加密码哈希、用户注册、忘记密码等功能。