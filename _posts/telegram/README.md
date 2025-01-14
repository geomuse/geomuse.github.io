以下是 **Telegram Bot 开发的 Python 教学**，包括如何创建一个基本的 Telegram Bot、常见功能（如消息处理、回复、命令处理等），以及高级功能（如 Webhook、数据库集成和文件上传）。

---

## 📚 **Telegram Bot 开发教学：30 天学习计划**
**开发工具**：Python + `python-telegram-bot`  
**目标**：从基础到高级，学习如何使用 Python 构建功能丰富的 Telegram Bot。

---

### 📅 **Day 1-5：环境准备与基础 Bot 搭建**

#### 1️⃣ **步骤 1：创建 Telegram Bot**
1. 打开 Telegram，搜索 **BotFather**。
2. 输入 `/start` 并使用 `/newbot` 命令创建新 Bot。
3. BotFather 会给你一个 **Token**，格式如下：

```
123456789:ABCdefGHIjklMNOpqrstuvWXYZ
```

#### 2️⃣ **步骤 2：安装 `python-telegram-bot` 库**
```bash
pip install python-telegram-bot
```

#### 3️⃣ **步骤 3：编写基础代码**
```python
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Hello {update.effective_user.first_name}!')

if __name__ == '__main__':
    app = ApplicationBuilder().token('YOUR_BOT_TOKEN').build()
    app.add_handler(CommandHandler('start', start))
    app.run_polling()
```

#### ✅ **Day 1-5 任务总结**
- 学习 Telegram API 和 BotFather 的用法。
- 学会使用 `python-telegram-bot` 库。
- 完成基础 Bot 的启动和消息回复功能。

---

### 📅 **Day 6-10：常见功能实现**

#### 1️⃣ **处理命令**
```python
from telegram.ext import CommandHandler

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Here are the available commands: \n/start - Start the bot\n/help - Get help")

app.add_handler(CommandHandler('help', help_command))
```

#### 2️⃣ **处理文本消息**
```python
from telegram.ext import MessageHandler, filters

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(update.message.text)

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
```

---

### 📅 **Day 11-15：文件、图片和音频处理**

#### 1️⃣ **接收和保存图片**
```python
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download('received_photo.jpg')
    await update.message.reply_text('Photo received and saved!')

app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
```

#### 2️⃣ **发送文件**
```python
async def send_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_document(chat_id=update.effective_chat.id, document=open('file.txt', 'rb'))

app.add_handler(CommandHandler('sendfile', send_file))
```

---

### 📅 **Day 16-20：数据库集成**

#### 1️⃣ **使用 SQLite 保存用户数据**
```python
import sqlite3

# 创建数据库连接
conn = sqlite3.connect('bot_data.db')
cursor = conn.cursor()

# 创建用户表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT,
    chat_id INTEGER
)
''')
conn.commit()

# 保存用户信息的函数
async def save_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    cursor.execute('INSERT INTO users (username, chat_id) VALUES (?, ?)', (user.username, update.effective_chat.id))
    conn.commit()
    await update.message.reply_text('User saved!')
```

---

### 📅 **Day 21-25：Webhook 与部署**

#### 1️⃣ **使用 Webhook 替代 Polling**
```python
app.run_webhook(
    listen="0.0.0.0",
    port=8443,
    url_path="YOUR_BOT_TOKEN",
    webhook_url=f"https://yourdomain.com/YOUR_BOT_TOKEN"
)
```

#### 2️⃣ **部署到 Heroku**
1. 创建 `Procfile` 文件：
```
web: python your_bot.py
```

2. 提交到 Heroku：
```bash
git init
git add .
git commit -m "Initial commit"
heroku create
git push heroku main
```

---

### 📅 **Day 26-30：高级功能**

#### 1️⃣ **按钮与交互菜单**
```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Option 1", callback_data='1')],
        [InlineKeyboardButton("Option 2", callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Please choose:', reply_markup=reply_markup)

app.add_handler(CommandHandler('start', start))
```

#### 2️⃣ **处理 Callback**
```python
from telegram.ext import CallbackQueryHandler

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text=f"Selected option: {query.data}")

app.add_handler(CallbackQueryHandler(button))
```

---

### 🎯 **最终项目：Todo List Bot**
你可以结合所学的内容，开发一个 **Todo List Bot**，支持用户添加、查看和删除任务。

---

### 📦 **附加资源**
- [官方文档](https://python-telegram-bot.readthedocs.io/)
- [BotFather](https://t.me/BotFather)
- [部署指南](https://devcenter.heroku.com/articles/getting-started-with-python)

需要详细代码或调试帮助吗？可以随时提问！ 😊