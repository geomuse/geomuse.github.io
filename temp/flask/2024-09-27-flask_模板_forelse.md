---
layout: post
title:  flask 模板 forelse
date:   2024-09-19 11:24:29 +0800
categories: 
    - python 
    - flask
---

```py
from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# 自由时报国际新闻页面的 URL
url = "https://news.ltn.com.tw/list/breakingnews/world"

def scrape_news():
    # 发出 HTTP 请求
    response = requests.get(url)
    # 使用 BeautifulSoup 解析网页
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找新闻标题和链接
    news_items = []
    for item in soup.find_all('li', class_='searchlist'):
        title = item.find('a').get_text().strip()  # 提取标题
        link = item.find('a')['href']  # 提取链接
        news_items.append({'title': title, 'link': link})

    return news_items

@app.route('/')
def show_news():
    # 爬取新闻
    news_list = scrape_news()
    return render_template('news.html', news_list=news_list)

if __name__ == '__main__':
    app.run(debug=True)
```

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>国际新闻</title>
</head>
<body>
    <h1>自由时报国际新闻</h1>
    <ul>
        {% for news in news_list %}
        <li>
            <a href="{{ news.link }}" target="_blank">{{ news.title }}</a>
        </li>
        {% endfor %}
    </ul>
</body>
</html>
```