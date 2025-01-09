from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# 1️⃣ 启动命令处理函数
async def start(update: Update, context):
    await update.message.reply_text(f"Hello, {update.effective_user.first_name}! I'm your bot 🤖")

# 2️⃣ 回应消息的处理函数
async def echo(update: Update, context):
    text = update.message.text
    await update.message.reply_text(f'You said: {text}')

# 3️⃣ 主函数：初始化应用
def main():
    TOKEN = "7252086234:AAHzFiph2NXqter34M1yjH5oR2ZHYw8gzFE"
    app = ApplicationBuilder().token(TOKEN).build()

    # 添加命令处理器
    app.add_handler(CommandHandler("start", start))

    # 添加消息处理器
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # 启动机器人
    app.run_polling()

if __name__ == "__main__":
    main()
