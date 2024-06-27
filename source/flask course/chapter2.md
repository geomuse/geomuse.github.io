### 第二课：第一个Flask应用

#### 1. 创建一个简单的Flask应用
- 创建一个名为`app.py`的文件：

  ```python
  from flask import Flask

  app = Flask(__name__)

  @app.route('/')
  def hello_world():
      return 'Hello, World!'

  if __name__ == '__main__':
      app.run(debug=True)
  ```
- 运行应用：

  ```bash
  python app.py
  ```
- 访问`http://127.0.0.1:5000/`查看结果。

#### 2. 路由和视图函数

- 使用`@app.route`定义URL路由。
- 路由可以包含变量：

  ```python
  @app.route('/')
  def index():
      return render_template('index.html')
  ```

  ```python
  @app.route('/user/<username>')
  def show_user_profile(username):
      return f'User {username}'
  ```

#### 3. 使用Jinja2模板引擎
- 创建`templates`文件夹并添加HTML文件（如`index.html`）。
- 渲染模板：

  ```python
  from flask import render_template

  @app.route('/')
  def home():
      return render_template('index.html')
  ```
