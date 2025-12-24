import socket
import threading

def scan_port(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex((ip, port)) == 0:
            print(f"[+] 端口 {port} 开放")
        sock.close()
    except:
        pass

def run_scan(ip, ports):
    threads = []
    for port in ports:
        t = threading.Thread(target=scan_port, args=(ip, port))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

# 扫描本机常见端口
target_ip = "127.0.0.1"
ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 3389]
run_scan(target_ip, ports)

import socket

def banner_grab(ip, port):
    try:
        # 建立 TCP 连接
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((ip, port))

        # 发送空请求（有些服务会主动回 Banner）
        s.send(b"\r\n")
        
        # 接收返回数据
        banner = s.recv(1024)
        print(f"[+] {ip}:{port} - {banner.decode().strip()}")
        
    except Exception as e:
        print(f"[-] {ip}:{port} - 无响应 ({e})")
    finally:
        s.close()

# 示例：抓取 80 端口（HTTP）
banner_grab("scanme.nmap.org", 80)
