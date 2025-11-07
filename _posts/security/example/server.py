import socket

# 创建 TCP socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定 IP 和端口
server.bind(("127.0.0.1", 8080))
server.listen(1)
print("服务器已启动，等待连接...")

# 接收客户端连接
conn, addr = server.accept()
print("连接来自:", addr)

# 接收消息
data = conn.recv(1024).decode()
print("收到:", data)

# 回复消息
conn.send("你好，客户端！".encode())

# 关闭连接
conn.close()
