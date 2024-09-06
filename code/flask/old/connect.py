from flask import Flask, jsonify, request
from pymongo import MongoClient

app = Flask(__name__)

# 连接 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['List'] 

# 示例集合（类似于表）
collection = db['tasks']

data = {"name": "Alice", "age": 30, "city": "New York"}
result = collection.insert_one(data)

# 输出插入的数据的 ObjectId
print(f"数据插入成功,ID : {result.inserted_id}")

# 插入多条文档
data_list = [
    {"name": "Bob", "age": 25, "city": "Los Angeles"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]
result = collection.insert_many(data_list)

# 输出所有插入的数据的 ObjectId
print(f"数据插入成功，IDs: {result.inserted_ids}")