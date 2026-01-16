import requests
import os

TOKEN = "8312172130:AAHVyEpIItPeuiAykeuN9CMCJya_Gz6U7uk"
CHAT_ID = "-1003370646305"

def send_telegram_markdown(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": markdown_content,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status() # Raise error for 4xx/5xx
        print("Response:", response.json())
        print("✅ Message sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending message: {e}")
        if response is not None:
             print("Response text:", response.text)

if __name__ == "__main__":

    # Get absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    markdown_file = os.path.join(script_dir, "daily_report.md")
    
    send_telegram_markdown(markdown_file)