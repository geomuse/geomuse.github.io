import zmq

def main():
    context = zmq.Context()
    # 用來接收訊息的 PULL socket
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:5555")
    
    # 用來廣播訊息的 PUB socket
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5556")
    
    print("聊天室伺服器啟動，等待訊息...")
    while True:
        try:
            # 接收來自客戶端的訊息
            message = receiver.recv_string()
            print(f"收到訊息: {message}")
            # 廣播訊息給所有訂閱者
            publisher.send_string(message)
        except KeyboardInterrupt:
            break

    receiver.close()
    publisher.close()
    context.term()

if __name__ == "__main__":
    main()
