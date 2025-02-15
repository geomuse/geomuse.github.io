import requests
from bs4 import BeautifulSoup
import pandas as pd

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from bs4 import BeautifulSoup

# 创建回复键盘
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["查看金十黄金新闻"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("请选择操作：", reply_markup=reply_markup)

# 处理用户点击的按钮
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text

    if user_input == "查看金十黄金新闻":
        news = get_jin10_news()
        await update.message.reply_text(news, disable_web_page_preview=True)
    else:
        await update.message.reply_text("请选择有效的选项！")

def get_jin10_news():
    # 金十黄金新闻URL
    url = "https://www.jin10.com/"

    # 请求网页
    response = requests.get(url)
    response.encoding = 'utf-8'

    # 使用 BeautifulSoup 解析网页
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到新闻列表
    news_list = soup.find_all('div', class_='jin-flash_list')

    soup = news_list[0].find_all('div',class_='flash-text')

    # 存储新闻数据
    news_data = ""
    # 遍历新闻
    for news in soup[20:]:
        title = news.text.strip()
        news_data += f"🔹 {title}\n"

    return news_data if news_data else "今日无新闻更新。"

# 爬取新闻函数
def get_ltn_news():
    url = "https://news.ltn.com.tw/list/breakingnews/world"
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ 无法访问自由时报新闻网站。"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_list = soup.find_all('h3')
    # link = news_list[0].find_previous('a')['href']
    news_data = ""
    for news in news_list[:20]:  # 只取前 5 条新闻
        title = news.text.strip()
        link = news.find_previous('a')['href']
        news_data += f"🔹 {title}\n {link}\n\n"
    
    return news_data if news_data else "今日无新闻更新。"

# 主函数
if __name__ == '__main__':
    app = ApplicationBuilder().token("8197503885:AAEuf50LBEhRdb3UrR2bXvsI1RmNdK6HhYQ").build()

    # 命令处理器
    app.add_handler(CommandHandler("start", start))

    # 消息处理器
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot 正在运行...")
    app.run_polling()
