### 第四课：静态文件和模板

#### 1. 管理静态文件（CSS、JavaScript、图片等）
- 创建`static`文件夹存放静态文件。
- 在HTML模板中引用：

  ```html
  <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
  ```

#### 2. 使用模板继承和布局
- 创建一个基础模板（`base.html`）：

  ```html
  <!doctype html>
  <html>
  <head>
      <title>{% block title %}My Site{% endblock %}</title>
  </head>
  <body>
      {% block content %}{% endblock %}
  </body>
  </html>
  ```
  
- 创建继承模板：

  ```html
  {% extends "base.html" %}

  {% block title %}Home{% endblock %}
  {% block content %}
      <h1>Welcome to My Site</h1>
  {% endblock %}
  ```
