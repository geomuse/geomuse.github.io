import requests
import pandas as pd
from config import TWELVE_API_KEY

BASE_URL = "https://api.twelvedata.com/time_series"

def get_candles(symbol, interval="4h", outputsize=300):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY
    }

    r = requests.get(BASE_URL, params=params).json()

    if "values" not in r:
        raise ValueError(f"Data error: {r}")

    df = pd.DataFrame(r["values"])
    df["close"] = df["close"].astype(float)
    df = df.sort_values("datetime")

    return df[["datetime", "close"]]

def get_risk_events():
    return "今晚有美国 CPI 数据，注意剧烈波动"
