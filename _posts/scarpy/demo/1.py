import concurrent.futures
import requests

urls = [
    'http://example.com/page1',
    'http://example.com/page2',
    # 添加更多的URL
]

def fetch(url):
    try:
        response = requests.get(url)
        return f"{url}: {response.status_code}"
    except Exception as e:
        return f"请求失败 {url}: {e}"

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetch, urls)

for result in results:
    print(result)
