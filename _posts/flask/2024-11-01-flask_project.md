---
layout: post
title : 简单的图书管理 api
date:   2024-11-01 11:24:29 +0800
categories: 
    - python 
    - flask
---

# 练习：创建一个简单的图书管理 API

在这个练习中，我们将创建一个简单的图书管理 RESTful API，实现对图书信息的增删改查操作。我们将使用 Flask 或 Flask-RESTful 来构建这个 API，并使用内存中的列表来模拟数据存储。

## 功能需求

- **获取所有图书信息**（GET `/books`）
- **获取单本图书信息**（GET `/books/<book_id>`）
- **添加新图书**（POST `/books`）
- **更新图书信息**（PUT `/books/<book_id>`）
- **删除图书**（DELETE `/books/<book_id>`）

## 使用 Flask 构建 API

首先，我们将使用纯 Flask 来构建这个 API。

### 完整代码

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# 模拟的图书数据
books = [
    {'id': 1, 'title': '红楼梦', 'author': '曹雪芹', 'publisher': '人民文学出版社'},
    {'id': 2, 'title': '三国演义', 'author': '罗贯中', 'publisher': '中华书局'}
]

# 获取所有图书信息
@app.route('/books', methods=['GET'])
def get_books():
    return jsonify({'books': books})

# 获取单本图书信息
@app.route('/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if book:
        return jsonify(book)
    else:
        return jsonify({'message': '图书未找到'}), 404

# 添加新图书
@app.route('/books', methods=['POST'])
def create_book():
    data = request.get_json()
    if not data or not 'title' in data or not 'author' in data or not 'publisher' in data:
        return jsonify({'message': '数据格式错误'}), 400

    new_id = books[-1]['id'] + 1 if books else 1
    new_book = {
        'id': new_id,
        'title': data['title'],
        'author': data['author'],
        'publisher': data['publisher']
    }
    books.append(new_book)
    return jsonify(new_book), 201

# 更新图书信息
@app.route('/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if not book:
        return jsonify({'message': '图书未找到'}), 404

    data = request.get_json()
    book['title'] = data.get('title', book['title'])
    book['author'] = data.get('author', book['author'])
    book['publisher'] = data.get('publisher', book['publisher'])
    return jsonify(book)

# 删除图书
@app.route('/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    global books
    books = [b for b in books if b['id'] != book_id]
    return jsonify({'message': '图书已删除'})

if __name__ == '__main__':
    app.run(debug=True)
```

### 代码解释

- **导入必要的模块**：
  - `Flask`：用于创建 Flask 应用。
  - `request`：用于处理请求数据。
  - `jsonify`：用于将 Python 数据结构转换为 JSON 响应。

- **数据模型**：
  - 使用列表 `books` 来存储图书信息，每本图书是一个包含 `id`、`title`、`author`、`publisher` 的字典。

- **路由和视图函数**：
  - `@app.route('/books', methods=['GET'])`：获取所有图书信息。
  - `@app.route('/books/<int:book_id>', methods=['GET'])`：获取单本图书信息。
  - `@app.route('/books', methods=['POST'])`：添加新图书。
  - `@app.route('/books/<int:book_id>', methods=['PUT'])`：更新图书信息。
  - `@app.route('/books/<int:book_id>', methods=['DELETE'])`：删除图书。

- **请求处理**：
  - **GET 请求**：直接返回数据，使用 `jsonify` 包装。
  - **POST 请求**：从请求中获取 JSON 数据，验证必要的字段，创建新图书并添加到列表中。
  - **PUT 请求**：查找要更新的图书，更新提供的字段。
  - **DELETE 请求**：从列表中移除指定的图书。

- **错误处理**：
  - 当请求的图书不存在时，返回 `404` 状态码和相应的消息。
  - 当请求数据格式错误时，返回 `400` 状态码。

### 运行应用

将上述代码保存为 `app.py`，在命令行中运行：

```bash
python app.py
```

服务器将运行在 `http://localhost:5000`。

### 测试 API

#### 获取所有图书

```bash
curl http://localhost:5000/books
```

**响应：**

```json
{
  "books": [
    {
      "author": "曹雪芹",
      "id": 1,
      "publisher": "人民文学出版社",
      "title": "红楼梦"
    },
    {
      "author": "罗贯中",
      "id": 2,
      "publisher": "中华书局",
      "title": "三国演义"
    }
  ]
}
```

#### 获取单本图书

```bash
curl http://localhost:5000/books/1
```

**响应：**

```json
{
  "author": "曹雪芹",
  "id": 1,
  "publisher": "人民文学出版社",
  "title": "红楼梦"
}
```

#### 添加新图书

```bash
curl -X POST -H "Content-Type: application/json" -d '{"title": "西游记", "author": "吴承恩", "publisher": "人民文学出版社"}' http://localhost:5000/books
```

**响应：**

```json
{
  "author": "吴承恩",
  "id": 3,
  "publisher": "人民文学出版社",
  "title": "西游记"
}
```

#### 更新图书信息

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"publisher": "新出版社"}' http://localhost:5000/books/1
```

**响应：**

```json
{
  "author": "曹雪芹",
  "id": 1,
  "publisher": "新出版社",
  "title": "红楼梦"
}
```

#### 删除图书

```bash
curl -X DELETE http://localhost:5000/books/1
```

**响应：**

```json
{
  "message": "图书已删除"
}
```

## 使用 Flask-RESTful 构建 API

接下来，我们使用 Flask-RESTful 来实现相同的功能。

### 安装 Flask-RESTful

```bash
pip install flask-restful
```

### 完整代码

```python
from flask import Flask, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

# 模拟的图书数据
books = [
    {'id': 1, 'title': '红楼梦', 'author': '曹雪芹', 'publisher': '人民文学出版社'},
    {'id': 2, 'title': '三国演义', 'author': '罗贯中', 'publisher': '中华书局'}
]

# 请求参数解析器
book_parser = reqparse.RequestParser()
book_parser.add_argument('title', type=str, required=True, help='Title is required')
book_parser.add_argument('author', type=str, required=True, help='Author is required')
book_parser.add_argument('publisher', type=str, required=True, help='Publisher is required')

# 定义资源类
class BookList(Resource):
    def get(self):
        return {'books': books}

    def post(self):
        args = book_parser.parse_args()
        new_id = books[-1]['id'] + 1 if books else 1
        new_book = {
            'id': new_id,
            'title': args['title'],
            'author': args['author'],
            'publisher': args['publisher']
        }
        books.append(new_book)
        return new_book, 201

class Book(Resource):
    def get(self, book_id):
        book = next((b for b in books if b['id'] == book_id), None)
        if book:
            return book
        else:
            return {'message': '图书未找到'}, 404

    def put(self, book_id):
        book = next((b for b in books if b['id'] == book_id), None)
        if not book:
            return {'message': '图书未找到'}, 404

        data = request.get_json()
        book['title'] = data.get('title', book['title'])
        book['author'] = data.get('author', book['author'])
        book['publisher'] = data.get('publisher', book['publisher'])
        return book

    def delete(self, book_id):
        global books
        books = [b for b in books if b['id'] != book_id]
        return {'message': '图书已删除'}

# 添加资源到 API
api.add_resource(BookList, '/books')
api.add_resource(Book, '/books/<int:book_id>')

if __name__ == '__main__':
    app.run(debug=True)
```

### 代码解释

- **导入模块**：
  - `Flask`、`request`：Flask 核心模块。
  - `flask_restful.Resource`：用于定义资源类。
  - `flask_restful.Api`：用于创建 API 对象。
  - `flask_restful.reqparse`：用于请求参数解析。

- **数据模型**：与之前相同，使用列表 `books` 存储图书信息。

- **请求参数解析器**：
  - 使用 `reqparse.RequestParser` 定义需要的请求参数，并进行验证。

- **资源类**：
  - `BookList`：处理 `/books` 路由，包含 `get` 和 `post` 方法。
  - `Book`：处理 `/books/<book_id>` 路由，包含 `get`、`put` 和 `delete` 方法。

- **添加资源到 API**：
  - `api.add_resource(BookList, '/books')`
  - `api.add_resource(Book, '/books/<int:book_id>')`

### 运行应用

将上述代码保存为 `app.py`，在命令行中运行：

```bash
python app.py
```

### 测试 API

测试方法与使用纯 Flask 时相同，可以使用 `curl` 或 Postman 等工具。

## 注意事项

- **线程安全**：使用全局变量（如列表）存储数据在多线程环境下可能会出现问题。在生产环境中，应使用数据库来存储和管理数据。
- **输入验证**：在实际应用中，应对输入的数据进行严格的验证和清洗，防止安全漏洞。
- **错误处理**：应提供更全面的错误处理机制，返回更有意义的错误信息和状态码。

## 进一步扩展

- **数据库集成**：将数据存储在数据库中，使用 ORM（如 SQLAlchemy）进行管理。
- **身份验证**：为 API 添加身份验证（如 Token、JWT），保护敏感操作。
- **分页和过滤**：实现对数据的分页、排序和过滤，提高 API 的实用性。
- **API 文档**：使用 Swagger 等工具生成 API 文档，方便他人使用你的 API。

## 小结

通过这个练习，我们学习了如何使用 Flask 和 Flask-RESTful 构建一个简单的 RESTful API，包括了对资源的增删改查操作。希望这对你理解 RESTful API 的开发有所帮助。