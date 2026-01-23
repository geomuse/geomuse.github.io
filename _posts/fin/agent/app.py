from data_provider import get_candles
from analyzer import analyze_symbol
from content_agent import generate_report
from telegram_sender import send_message

def run():
    symbols = {
        "EURUSD": "EUR/USD",
        "XAUUSD": "XAU/USD"
    }

    analysis = {}

    for name, symbol in symbols.items():
        df = get_candles(symbol)
        analysis[name] = analyze_symbol(df)

    risk_event = "本内容仅供一般资讯与市场观点分享，不构成任何投资建议、交易邀约或保证。\n外汇及金融市场具有高风险，过往表现不代表未来结果，所有交易决策与风险需自行承担。"
    report = generate_report(analysis, risk_event)

    send_message(report)

if __name__ == "__main__":
    run()
