import threading
import requests

urls = [
    'http://example.com/page1',
    'http://example.com/page2',
    # 添加更多的URL
]

def fetch(url):
    try:
        response = requests.get(url)
        print(f"{url}: {response.status_code}")
    except Exception as e:
        print(f"请求失败 {url}: {e}")

threads = []

for url in urls:
    t = threading.Thread(target=fetch, args=(url,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()