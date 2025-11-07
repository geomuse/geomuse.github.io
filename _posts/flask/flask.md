---
layout: post
title : python flask 数据库基础（Flask-SQLAlchemy）-4
date:   2026-08-13 11:24:29 +0800
tags: 
    - python 
    - flask
image: 07.jpg
---

大家好，我是python网页后端flask的讲师geo

注册表单

```py
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, User
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'supersecret'  # 用於 flash 訊息
db.init_app(app)

@app.route('/')
def index():
    return "<h1>首頁</h1><a href='/register'>註冊</a> | <a href='/users'>查看使用者</a>"

# ✅ 註冊路由（GET 顯示表單，POST 處理表單）
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash("請填寫所有欄位！")
            return redirect(url_for('register'))

        # 檢查使用者是否存在
        if User.query.filter_by(username=username).first():
            flash("使用者名稱已存在")
            return redirect(url_for('register'))

        # 寫入資料庫
        new_user = User(
            username=username,
            email=email,
            password=password,
            created_at=datetime.utcnow()
        )
        db.session.add(new_user)
        db.session.commit()
        flash("註冊成功！")
        return redirect(url_for('show_users'))

    return render_template('register.html')

@app.route('/users')
def show_users():
    users = User.query.all()
    return render_template('users.html', users=users)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
```

`templates/register.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>註冊新使用者</title>
</head>
<body>
    <h1>註冊新使用者</h1>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul style="color: red;">
          {% for msg in messages %}
            <li>{{ msg }}</li>
          {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}

    <form method="POST">
        <label>使用者名稱：</label><br>
        <input type="text" name="username"><br><br>

        <label>Email：</label><br>
        <input type="email" name="email"><br><br>

        <label>密碼：</label><br>
        <input type="password" name="password"><br><br>

        <input type="submit" value="註冊">
    </form>

    <p><a href="/">回首頁</a></p>
</body>
</html>
```