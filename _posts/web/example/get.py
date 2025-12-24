import requests
from bs4 import BeautifulSoup

# 目标网址
url = "https://www.malaymail.com/"

# 发起 GET 请求
response = requests.get(url)
response.encoding = 'utf-8'

# # 解析 HTML 内容
soup = BeautifulSoup(response.text, 'html.parser')

# # 抓取所有文章标题
articles = soup.find_all('h2', class_='article-title')

print("马来邮报头条新闻：")
for i, article in enumerate(articles): 
    title = article.get_text(strip=True)
    link = article.find('a')['href']
    print(f"{i}. {title} - {link}")