---
layout: post
title : flask 静态文件与url_for
date:   2024-11-23 11:24:29 +0800
categories: 
    - python 
    - flask
---

静态文件用于处理 CSS、JavaScript 和 图片资源，提升页面的外观和交互性

```py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
```

静态 CSS 文件 `(static/css/style.css)`:

```css
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    color: #333;
}

h1 {
    color: #007BFF;
    text-align: center;
}
```

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask Static Example</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <h1>Welcome to Flask!</h1>
</body>
</html>
```

```md
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
```

任务：

- 创建一个页面 /about，显示带有 CSS 样式的自定义消息。
- 在页面中添加一张图片。

```py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("indexx.html")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":

    app.run(debug=True)
```

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>about</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <h1>I'm geo</h1>
    <img src="{{ url_for('static', filename='images/1.png') }}" alt="Flask Logo" width="200">
</body>
</html>
```