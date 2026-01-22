from google import genai
from google.genai import types
from datetime import date
from config import GEMINI_API_KEY  # 你的 key

def generate_report(analysis, risk_event):
    today = date.today().isoformat()

    prompt = f"""
你是一位专业外汇分析师。
请根据以下结构化分析，撰写一份适合 Telegram 频道发布的中文外汇市场简报。
要求：专业、简洁、不夸张、不构成投资建议。

分析数据：
{analysis}

风险提示：
{risk_event}
"""

    try :
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        # 创建 Client，并传入 api_key
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
        content = f"response.text"
    # 直接在 create 调用中传入 api_key

    except Exception as e:
        # fallback：AI 不可用时
        content = ""
        for symbol, info in analysis.items():
            content += (
                f"{symbol}：{info['bias']}\n"
                f"- 价格：{info['price']}\n"
                f"- RSI：{info['rsi']}\n\n"
            )
        content += f"⚠️ 风险提示：{risk_event}"

    return f"📊 外汇市场晨报 | {today}\n\n{content}"
