import os
from datetime import date
import google.generativeai as genai
from config import GEMINI_API_KEY

# 设置 API Key
genai.configure(api_key=GEMINI_API_KEY)

def generate_report(analysis, risk_event):
    today = date.today().isoformat()

    # 构建 prompt
    prompt = f"""
你是一位专业外汇分析师。
请根据以下结构化分析，撰写一份适合 Telegram 频道发布的中文外汇市场简报。
要求:专业、简洁、不夸张、不构成投资建议。

分析数据:
{analysis}

风险提示:
{risk_event}
"""

    try:
        response = genai.chat.create(
            model="gemini-1.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        # Gemini 返回的文本
        content = response.candidates[0].content
    except Exception as e:
        # fallback，AI 不可用时自动用规则生成
        content = ""
        for symbol, info in analysis.items():
            content += (
                f"{symbol}:{info['bias']}\n"
                f"- 价格:{info['price']}\n"
                f"- RSI:{info['rsi']}\n\n"
            )
        content += f"⚠️ 风险提示:{risk_event}"

    return f"📊 外汇市场晨报 | {today}\n\n{content}"
