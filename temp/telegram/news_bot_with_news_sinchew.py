from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from bs4 import BeautifulSoup

# 创建回复键盘
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["社会新闻","即时国际新闻","国际头条新闻"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("请选择操作：", reply_markup=reply_markup)

# 处理用户点击的按钮
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text

    if user_input == "社会新闻":
        news = get_sinchew_news_1()
        await update.message.reply_text(news, disable_web_page_preview=True)
    elif user_input == "即时国际新闻":
        news = get_sinchew_news_2()
        await update.message.reply_text(news, disable_web_page_preview=True)
    elif user_input == "国际头条新闻":
        news = get_sinchew_news_3()
        await update.message.reply_text(news, disable_web_page_preview=True)

    else:
        await update.message.reply_text("请选择有效的选项！")

def get_sinchew_news_1():
    url = "https://www.sinchew.com.my/category/%e5%85%a8%e5%9b%bd/%e7%a4%be%e4%bc%9a"
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ 无法访问自由时报新闻网站。"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_list = soup.find_all('a',class_="internalLink")
    # print(news_list)
    # link = news_list.find_previous('a')['href']
    news_data = ""
    for news in news_list[120:]:  # 只取前 5 条新闻
        title = news.text.strip()
        link = news.find_previous('a')['href']
        if title.__len__() not in (2,3,4) :
            news_data += f"🔹 {title}\n {link}\n\n"
    
    return news_data if news_data else "今日无新闻更新。"

def get_sinchew_news_2():
    url = "https://www.sinchew.com.my/category/%e5%9b%bd%e9%99%85/%e5%8d%b3%e6%97%b6%e5%9b%bd%e9%99%85"
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ 无法访问自由时报新闻网站。"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_list = soup.find_all('a',class_="internalLink")
    # print(news_list)
    # link = news_list.find_previous('a')['href']
    news_data = ""
    for news in news_list[120:]:  # 只取前 5 条新闻
        title = news.text.strip()
        link = news.find_previous('a')['href']
        if title.__len__() not in (3,4) :
            news_data += f"🔹 {title}\n {link}\n\n"
    
    return news_data if news_data else "今日无新闻更新。"

def get_sinchew_news_3():
    url = "https://www.sinchew.com.my/category/%e5%9b%bd%e9%99%85/%e5%a4%a9%e4%b8%8b%e4%ba%8b"
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ 无法访问自由时报新闻网站。"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_list = soup.find_all('a',class_="internalLink")
    # print(news_list)
    # link = news_list.find_previous('a')['href']
    news_data = ""
    for news in news_list[120:]:  # 只取前 5 条新闻
        title = news.text.strip()
        link = news.find_previous('a')['href']
        if title.__len__() not in (3,4) :
            news_data += f"🔹 {title}\n {link}\n\n"
    
    return news_data if news_data else "今日无新闻更新。"

# 主函数
if __name__ == '__main__':
    app = ApplicationBuilder().token("7635177967:AAF7rlTudsp_CjZWtV4SJVK5-CDUudJfBDo").build()

    # 命令处理器
    app.add_handler(CommandHandler("start", start))

    # 消息处理器
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot 正在运行...")
    app.run_polling()
