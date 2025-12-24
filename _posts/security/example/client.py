import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接到服务器
client.connect(("127.0.0.1", 8080))

# 发送数据
client.send("你好，服务器！".encode())

# 接收数据
response = client.recv(1024).decode()
print("收到:", response)

client.close()
