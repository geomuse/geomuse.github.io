

import requests
import time

# --- 配置信息 ---
API_KEY = "d90a4220057a85976f9000fb"
TELEGRAM_TOKEN = "8312172130:AAHVyEpIItPeuiAykeuN9CMCJya_Gz6U7uk"
CHAT_ID = "-1003874137234"
TARGET_RATE = 7.50  # 设定的提醒阈值

def get_exchange_rate():
    """从 API 获取马币(MYR)对台币(TWD)的汇率"""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/MYR/TWD"
    try:
        response = requests.get(url)
        data = response.json()
        if data['result'] == 'success':
            return data['conversion_rate']
    except Exception as e:
        print(f"获取汇率失败: {e}")
    return None

def send_telegram_msg(message):
    """发送 Telegram 提醒"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

def main():
    print("汇率监控运行中...")
    # 获取当前汇率
    current_rate = get_exchange_rate()
    
    if current_rate:
        msg = f"📊 当前汇率提醒\n马币(MYR) -> 台币(TWD): {current_rate}"
        send_telegram_msg(msg)
        
        # 判断逻辑：如果汇率达到或超过 7.5
        if current_rate <= TARGET_RATE:
            alert_msg = f"🚀 汇率达标提醒！\n当前汇率: {current_rate}\n设定目标: {TARGET_RATE}\n赶紧去换钱吧！"
            send_telegram_msg(alert_msg)
            print("提醒已发送")
        else:
            print("未达目标，不发送提醒。")

if __name__ == "__main__":

    main()