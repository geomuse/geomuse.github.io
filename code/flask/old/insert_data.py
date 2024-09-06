import requests
import json

url = "http://127.0.0.1:5000/insert"
data = {"name": "Alice", "age": 25}

# 发送POST请求
response = requests.post(url, json=data)

# 打印响应结果
print(response.json())
