from flask import Flask, request, jsonify
from pymongo import MongoClient
import json

app = Flask(__name__)

# 设置 MongoDB 连接
client = MongoClient("mongodb://localhost:27017/")
db = client['List']  # 创建或选择数据库
collection = db['tasks']  # 创建或选择集合

@app.route('/add', methods=['POST'])
def add_data():
    data = request.json  # 获取 JSON 数据
    if data:
        # 将数据插入 MongoDB
        collection.insert_one(data)
        return jsonify({"message": "Data added successfully!"}), 201
    return jsonify({"error": "No data provided"}), 400

@app.route('/get', methods=['GET'])
def get_data():
    # 从 MongoDB 获取数据
    data = list(collection.find({}, {"_id": 0}))  # 排除 MongoDB 的 _id 字段
    return jsonify(data), 200

if __name__ == '__main__':
    app.run(debug=True)
