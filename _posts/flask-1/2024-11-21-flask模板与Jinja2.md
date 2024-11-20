---
layout: post
title : flask 模板与 Jinja2
date:   2024-11-21 11:24:29 +0800
categories: 
    - python 
    - flask
---

模板引擎 Jinja2：

- Flask 默认使用 Jinja2 渲染 HTML。
- 可以将动态数据传递到 HTML 模板中。
- 模板中的代码使用 `{{ ... }}` 语法显示变量，`{% ... %}` 定义控制结构（如条件和循环）。

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask Template</title>
</head>
<body>
    <h1>Hello, {{ username }}!</h1>
    <p>Your favorite color is {{ color }}.</p>
</body>
</html>
```

```py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    username = "Alice"
    color = "blue"
    return render_template("index.html", username=username, color=color)

if __name__ == "__main__":
    app.run(debug=True)
```

任务：

创建一个动态路由 `/profile/<username>/<int:age>`，将用户姓名和年龄传递到 HTML 模板并展示。

```py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    username = "Alice"
    color = "blue"
    return render_template("index.html", username=username, color=color)

@app.route("/profile/<username>/<int:age>")
def profile(username,age):
    return render_template('form.html',username=username, age=age)

if __name__ == "__main__":

    app.run(debug=True)
```

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask Form</title>
</head>
<body>
    <h3>Hello, {{ username }}!</h3>
    <p>Your age is {{ age }}.</p>
</body>
</html>
```