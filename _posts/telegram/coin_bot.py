from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Hello {update.effective_user.first_name}!')

if __name__ == '__main__':
    
    app = ApplicationBuilder().token('7881692586:AAEfQJgHEIaeD19ER4jnr4Ur-5YRuL8ihW0').build()
    app.add_handler(CommandHandler('start', start))
    app.run_polling()
