---
layout: post
title : basic data collection
date : 2024-11-05 11:24:29 +0800
categories: 
    - stats
    - scrapy
---

- 静态网页使用 requests。
- 结构化内容使用 BeautifulSoup。
- 动态内容使用 Selenium。
- 特定格式数据使用正则表达式。

### 使用 requests 抓取静态网页内容

这是最常用的方式，适合不需要 JavaScript 渲染的静态网页。我们可以直接获取页面内容并解析 HTML。

```py
import requests
from bs4 import BeautifulSoup

url = "https://example.com"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}

# 发起请求
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# 示例：提取标题和链接
titles = [title.text for title in soup.find_all("h2")]
links = [link.get("href") for link in soup.find_all("a")]

# 打印提取的内容
for title, link in zip(titles, links):
    print("Title:", title)
    print("Link:", link)
```

### 使用 BeautifulSoup 解析 HTML

BeautifulSoup 允许我们使用标签、类名和 ID 等选择器来解析内容。
基础解析

```py
# 查找页面中第一个标题
title = soup.find("h1").text
print("Page Title:", title)

# 查找所有具有特定类名的元素
items = soup.find_all("div", class_="product")
for item in items:
    product_name = item.find("h2").text
    price = item.find("span", class_="price").text
    print(f"Product: {product_name}, Price: {price}")
```

### 使用 CSS 选择器

CSS 选择器可以帮助我们通过复杂的选择条件进行查找，比如嵌套结构。

```py
# 使用 select() 方法
titles = soup.select("div.article h2")
for title in titles:
    print("Article Title:", title.text)
```

### 使用 Selenium 抓取动态内容

对于需要 JavaScript 渲染的页面，可以使用 Selenium 进行抓取。

```py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 设置 ChromeDriver 路径
chrome_options = Options()
chrome_options.add_argument("--headless")  # 无头模式
driver = webdriver.Chrome(service=Service('/path/to/chromedriver'), options=chrome_options)

# 打开网页
driver.get("https://example.com")

# 等待页面加载
driver.implicitly_wait(5)

# 获取动态内容
elements = driver.find_elements(By.CLASS_NAME, "dynamic-item")
for element in elements:
    print("Dynamic Content:", element.text)

driver.quit()
```

### 使用正则表达式提取内容

正则表达式适合从网页中提取特定模式的数据，比如电话号码、电子邮件、特定格式的文本等。

```py
import re

html_text = response.text  # 使用 requests 获取的页面内容

# 提取所有电话号码
phone_numbers = re.findall(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", html_text)
print("Phone Numbers:", phone_numbers)

# 提取所有电子邮件地址
emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", html_text)
print("Emails:", emails)
```

### 完整爬虫解析

将使用 requests 抓取网页内容，BeautifulSoup 解析 HTML，Selenium 处理动态内容，以及正则表达式提取特定数据。

- 静态网页使用 requests。
- 结构化内容使用 BeautifulSoup。
- 动态内容使用 Selenium。
- 特定格式数据使用正则表达式。

```py
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import re

# Step 1: Requests 获取静态内容
url = "https://example.com"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# 提取标题和链接
titles = [title.text for title in soup.find_all("h2")]
links = [link.get("href") for link in soup.find_all("a")]

# Step 2: 使用 Selenium 抓取动态内容
driver = webdriver.Chrome()  # 设置 ChromeDriver
driver.get(url)

# 等待加载并抓取动态元素
driver.implicitly_wait(5)
dynamic_elements = driver.find_elements(By.CLASS_NAME, "dynamic-item")
dynamic_content = [element.text for element in dynamic_elements]
driver.quit()

# Step 3: 使用正则表达式提取电话和电子邮件
html_text = response.text
phone_numbers = re.findall(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", html_text)
emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", html_text)

# 输出所有提取的数据
print("Titles and Links:", list(zip(titles, links)))
print("Dynamic Content:", dynamic_content)
print("Phone Numbers:", phone_numbers)
print("Emails:", emails)
```