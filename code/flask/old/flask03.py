from flask import Flask, request, render_template, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)

# 连接 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['List']
collection = db['tasks']

# 显示数据的主页
@app.route('/')
def index():
    data = list(collection.find())
    for item in data:
        item['_id'] = str(item['_id'])  # 将 ObjectId 转换为字符串，以便在 HTML 中显示
    return render_template('index.html', data=data)

# 插入数据的路由
@app.route('/insert', methods=['POST'])
def insert_data():
    name = request.form['name']
    age = request.form['age']
    city = request.form['city']
    collection.insert_one({'name': name, 'age': int(age), 'city': city})
    return redirect(url_for('index'))

# 删除数据的路由
@app.route('/delete/<id>', methods=['POST'])
def delete_data(id):
    collection.delete_one({'_id': ObjectId(id)})
    return redirect(url_for('index'))

if __name__ == '__main__':
    
    app.run(debug=True)
