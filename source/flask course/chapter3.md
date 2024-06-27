### 第三课：请求与响应

#### 1. 处理GET和POST请求
- 使用不同的HTTP方法：

  ```python
  @app.route('/login', methods=['GET', 'POST'])
  def login():
      if request.method == 'POST':
          return 'Do the login'
      else:
          return 'Show the login form'
  ```

#### 2. 读取表单数据
- 获取表单数据：

  ```python
  from flask import request

  @app.route('/login', methods=['POST'])
  def login():
      username = request.form['username']
      password = request.form['password']
      return f'Username: {username}, Password: {password}'
  ```

#### 3. 返回响应
- 返回不同类型的响应：

  ```python
  @app.route('/json')
  def json_response():
      return jsonify({'key': 'value'})
  ```