import asyncio
import aiohttp

urls = [
    'http://example.com/page1',
    'http://example.com/page2',
    # 添加更多的URL
]

async def fetch(session, url):
    try:
        async with session.get(url) as response:
            print(f"{url}: {response.status}")
    except Exception as e:
        print(f"请求失败 {url}: {e}")

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        await asyncio.gather(*tasks)

asyncio.run(main())
