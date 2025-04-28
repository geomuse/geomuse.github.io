---
layout: post
title:  flask index 
date:   2025-04-20 11:24:29 +0800
categories: 
    - flask
    - python 
---

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static',filename='style.css')}}">
    <title>Document</title>
</head>
<body>
    <h3> {{ name }} </h3>
    {% for movie in movies %}
    <li>
        {{movie.title}} - {{movie.year}} - {{movie.star}}
    </li>
    {% endfor %}
</body>
</html>
```