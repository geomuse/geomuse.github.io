---
layout: post
title:  zeromq chat room
date:   2025-01-02 11:24:29 +0800
categories: 
    - python
    - socket
---

server 

```py
import zmq

def chat_server():
    context = zmq.Context()

    # Socket to receive messages from clients
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:5555")  # Bind to a port

    # Socket to send messages to clients
    broadcaster = context.socket(zmq.PUB)
    broadcaster.bind("tcp://*:5556")  # Bind to another port

    print("Chat server started...")

    while True:
        # Receive message from a client
        message = receiver.recv_string()
        print(f"Received: {message}")

        # Broadcast message to all clients
        broadcaster.send_string(message)

if __name__ == "__main__":
    chat_server()
```

client 

```py
import zmq
import threading

def receive_messages(subscriber):
    while True:
        message = subscriber.recv_string()
        print(f"Broadcast: {message}")

def chat_client(username):
    context = zmq.Context()

    # Socket to send messages to server
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5555")  # Connect to server's receiver socket

    # Socket to receive broadcast messages from server
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5556")  # Connect to server's broadcaster socket
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

    # Start thread for receiving messages
    threading.Thread(target=receive_messages, args=(subscriber,), daemon=True).start()

    print(f"Welcome to the chat, {username}!")

    while True:
        message = input("You: ")
        if message.lower() == "exit":
            print("Exiting chat...")
            break
        # Send message to server
        sender.send_string(f"{username}: {message}")

if __name__ == "__main__":
    username = input("Enter your username: ")
    chat_client(username)
```