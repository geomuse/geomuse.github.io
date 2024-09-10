---
layout: post
title:  challenge-07/30-flask GET | POST
date:   2024-09-11 11:24:29 +0800
categories: 
    - python 
    - flask
    - challenge
---

在 Flask 中，表单的处理可以通过 `GET` 和 `POST` 方法来完成。下面是一个简单的示例，演示如何使用 HTML 表单提交数据，并通过 Flask 的 `request` 对象处理用户输入。

### Flask 示例代码

1. **创建一个简单的 HTML 表单：**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Form Example</title>
</head>
<body>
    <h2>Submit Your Info</h2>
    <form method="POST" action="/submit">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required><br><br>
        
        <label for="age">Age:</label>
        <input type="number" id="age" name="age" required><br><br>
        
        <input type="submit" value="Submit">
    </form>
</body>
</html>
```

2. **Flask 处理 GET 和 POST 请求：**

```python
from flask import Flask, request, render_template_string

app = Flask(__name__)

# HTML 表单的内容作为字符串
form_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Form Example</title>
</head>
<body>
    <h2>Submit Your Info</h2>
    <form method="POST" action="/submit">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required><br><br>
        
        <label for="age">Age:</label>
        <input type="number" id="age" name="age" required><br><br>
        
        <input type="submit" value="Submit">
    </form>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # 从表单获取数据
        name = request.form['name']
        age = request.form['age']
        return f"Received POST request! Name: {name}, Age: {age}"
    else:
        return "Please use the form to submit data."

if __name__ == '__main__':
    app.run(debug=True)
```

### 说明：

1. **HTML 表单**：
   - 创建了一个简单的表单，用户可以输入 `name` 和 `age`，并使用 `POST` 方法提交到 `/submit` 路径。

2. **Flask 路由**：
   - `/` 路由显示 HTML 表单。
   - `/submit` 路由处理表单提交，检查请求方法是否为 `POST`，并从表单中提取用户输入的数据。

### 运行方法：

1. 将代码保存为 `app.py`。
2. 在终端运行 `python app.py` 启动 Flask 服务器。
3. 打开浏览器，访问 `http://127.0.0.1:5000/` 填写表单并提交。

这段代码展示了如何处理来自表单的 `GET` 和 `POST` 请求，并提取用户输入的数据。