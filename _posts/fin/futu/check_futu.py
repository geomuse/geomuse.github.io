import socket

def check_futuopend(host='127.0.0.1', port=11111, timeout=3):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        print(f"✅ 成功连接 {host}:{port}，FutuOpenD 服务可用！")
    except socket.error as e:
        print(f"❌ 无法连接 {host}:{port}，错误信息: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    check_futuopend()