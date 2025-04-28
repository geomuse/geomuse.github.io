from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Hello {update.effective_user.first_name}! This is my name.')

if __name__ == '__main__':
    
    app = ApplicationBuilder().token('8197503885:AAEuf50LBEhRdb3UrR2bXvsI1RmNdK6HhYQ').build()
    app.add_handler(CommandHandler('start', start))
    app.run_polling()
