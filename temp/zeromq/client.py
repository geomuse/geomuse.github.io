import zmq
import threading

def receive_messages(sub_socket):
    while True:
        try:
            message = sub_socket.recv_string()
            print(f"\n[收到] {message}")
        except Exception as e:
            print("接收訊息時發生錯誤:", e)
            break

def main():
    context = zmq.Context()
    
    # PUSH socket: 用來傳送訊息給伺服器
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5555")  # 如伺服器在其他主機，修改此IP地址
    
    # SUB socket: 用來訂閱伺服器廣播的訊息
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5556")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # 訂閱所有訊息
    
    # 開一個執行緒來接收訊息
    thread = threading.Thread(target=receive_messages, args=(subscriber,), daemon=True)
    thread.start()
    
    print("請開始輸入訊息 (按 Ctrl+C 結束)：")
    try:
        while True:
            text = input()
            if text.strip() != "":
                sender.send_string(text)
    except KeyboardInterrupt:
        print("\n離開聊天室...")
    
    sender.close()
    subscriber.close()
    context.term()

if __name__ == "__main__":
    main()
