import requests 
from bs4 import BeautifulSoup

def get_ltn_news():
    url = "https://news.ltn.com.tw/list/breakingnews/world"
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ 无法访问自由时报新闻网站。"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_list = soup.find_all('h3')
    
    news_data = ""
    for news in news_list[:20]:  
        title = news.text.strip()
        link = news.find_previous('a')['href']
        news_data += f"{title}\n {link}\n\n"
    
    return news_data if news_data else "今日无新闻更新。"

print(get_ltn_news())