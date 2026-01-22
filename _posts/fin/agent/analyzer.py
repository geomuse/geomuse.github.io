import pandas_ta as ta

def analyze_symbol(df):
    df["ema200"] = ta.ema(df["close"], length=200)
    df["rsi"] = ta.rsi(df["close"], length=14)

    last = df.iloc[-1]

    if last["close"] > last["ema200"] and last["rsi"] > 50:
        bias = "偏多"
    elif last["close"] < last["ema200"] and last["rsi"] < 50:
        bias = "偏空"
    else:
        bias = "震荡"

    return {
        "bias": bias,
        "price": round(last["close"], 5),
        "rsi": round(last["rsi"], 1)
    }
