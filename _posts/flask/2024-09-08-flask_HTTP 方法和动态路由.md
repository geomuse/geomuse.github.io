---
layout: post
title:  flask 动态路由和 http 方法
date:   2024-09-08 11:24:29 +0800
categories: 
    - python 
    - flask
---

### 动态路由

动态路由允许你通过 URL 捕获传递的参数。可以通过在路由定义中使用尖括号 <parameter> 来表示参数。

```py
from flask import Flask

app = Flask(__name__)

# 动态路由，接受 URL 中的 <name> 参数
@app.route('/user/<name>')
def greet_user(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    app.run(debug=True)
```

### HTTP 方法 (GET, POST)

Flask 通过 `methods` 参数来指定处理的 HTTP 请求类型。常用的 HTTP 方法包括 `GET` 和 `POST`。

#### **GET 请求**

默认情况下，Flask 处理 `GET` 请求。你可以通过 URL 传递查询参数，并在视图函数中使用 `request.args` 获取这些参数。

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('q', '')  # 获取查询参数 ?q=value
    return f"Search results for: {query}"

if __name__ == "__main__":
    app.run(debug=True)
```

访问 `/search?q=Flask` 将显示 "Search results for: Flask"。

#### **POST 请求**

`POST` 请求通常用于表单提交。你可以通过 `methods` 参数指定路由处理 `POST` 请求，并通过 `request.form` 访问表单数据。

```python
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 简单的 HTML 表单
html_form = '''
    <form method="POST">
        <label for="name">Name:</label>
        <input type="text" name="name" id="name">
        <input type="submit" value="Submit">
    </form>
'''

@app.route('/submit', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':
        name = request.form['name']  # 获取表单提交的 name 字段
        return f"Form submitted with name: {name}"
    return render_template_string(html_form)

if __name__ == "__main__":
    app.run(debug=True)
```

在这个例子中，访问 `/submit` 会显示一个 HTML 表单。提交表单后，页面将显示表单中输入的名字。
