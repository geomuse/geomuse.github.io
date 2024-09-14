---
layout: post
title:  challenge-07/30-flask mongodb
date:   2024-09-12 11:24:29 +0800
categories: 
    - python 
    - flask
    - mongodb
    - challenge
---

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <title>geo</title>
</head>
<body>
    <h1>Hello, {{ name }}!</h1>
    <!-- <p>You are {{ age }} years old.</p> -->
    <p> {{ content }}</p>
</body>
</html>
```

```py
from flask import Flask, request, render_template
import smtplib
from email.mime.text import MIMEText
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['recipe']
collection = db['info']

r = [ recipe for recipe in collection.find() ]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',name='geo',content=r)

if __name__ == '__main__':

    app.run(debug=True)
```