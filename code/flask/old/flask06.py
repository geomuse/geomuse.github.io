from flask import Flask, jsonify, request
from pymongo import MongoClient

app = Flask(__name__)

# 连接 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['List']  # 使用 'test_database' 数据库

# 示例集合（类似于表）
collection = db['tasks']

# 插入数据的路由
@app.route('/insert', methods=['POST'])
def insert_data():
    data = request.json
    collection.insert_one(data)  # 将 JSON 数据插入到 MongoDB
    return jsonify({"message": "数据已插入"}), 201

# 获取数据的路由
@app.route('/data', methods=['GET'])
def get_data():
    data = list(collection.find())  # 从 MongoDB 中获取数据
    for item in data:
        item['_id'] = str(item['_id'])  # 将 ObjectId 转换为字符串
    return jsonify(data), 200

if __name__ == '__main__':

    app.run(debug=True)