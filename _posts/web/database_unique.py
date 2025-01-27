from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["stock_data"]  # 数据库名称
print(collection:=db['AAPL'])

data = []
for record in collection.find():
    if record not in data : 
        data.append(record)

print(len(data)) 