import requests

TOKEN = "8312172130:AAHVyEpIItPeuiAykeuN9CMCJya_Gz6U7uk"
CHAT_ID = "-1003370646305"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    url_response = requests.post(url, data=data)
    print(url_response.json())

send_telegram("🚀 Telegram 推送测试成功")
