---
layout: post
title:  flask mongodb 设置
date:   2024-09-12 11:24:29 +0800
categories: 
    - python 
    - flask
    - mongodb
---

建设 `index.html` 

```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
        <title>...</title>
    </head>
    <body>
        <h1>Hello, {{ name }}!</h1>
        <p> {{ content_ }} </p>  
    </body>
</html>
```

后端就负责把数据全部丢到 `index.html`

```py
from flask import Flask, request, render_template
import smtplib
from email.mime.text import MIMEText
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient('mongodb://localhost:27017/')
db = client['recipe']
collection = db['info']

r = [ recipe for recipe in collection.find() ]

@app.route('/')
def index():
    return render_template('index.html',name='geo',content_=r)

if __name__ == '__main__':

    app.run(debug=True)
```