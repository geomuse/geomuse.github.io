import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib as ta

# 连接 MT5
if not mt5.initialize():
    print("MT5 连接失败")
    quit()

# 交易品种
SYMBOL = "XAUUSD"

# 回测参数
# START_DATE = datetime(2023, 1, 1)
# END_DATE = datetime(2024, 1, 1)
LOT_SIZE = 0.01
SL = 20  # 止损 20 pips
TP = 40  # 止盈 40 pips
SPREAD = 0.0001  # 假设 1 pip 交易成本

# 获取历史数据
rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 500)
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

# 计算均线
df["EMA5"] = ta.EMA(df["close"],timeperiod=5)
df["EMA20"] = ta.EMA(df["close"],timeperiod=20)

# 计算信号（1 = 买入, -1 = 卖出, 0 = 无操作）
df["signal"] = 0
df.loc[(df["EMA5"] > df["EMA20"]) & (df["EMA5"].shift(1) <= df["EMA20"].shift(1)), "signal"] = 1
df.loc[(df["EMA5"] < df["EMA20"]) & (df["EMA5"].shift(1) >= df["EMA20"].shift(1)), "signal"] = -1

# 模拟交易
capital = 1000  # 初始资金
balance = capital
positions = []
equity_curve = []

for i, row in df.iterrows():
    if row["signal"] == 1:  # 买入信号
        entry_price = row["close"]
        sl_price = entry_price - SL * 0.0001
        tp_price = entry_price + TP * 0.0001
        positions.append({"type": "BUY", "entry": entry_price, "sl": sl_price, "tp": tp_price})

    elif row["signal"] == -1:  # 卖出信号
        entry_price = row["close"]
        sl_price = entry_price + SL * 0.0001
        tp_price = entry_price - TP * 0.0001
        positions.append({"type": "SELL", "entry": entry_price, "sl": sl_price, "tp": tp_price})

    # 检查是否止盈或止损
    for pos in positions[:]:  # 遍历持仓
        if pos["type"] == "BUY":
            if row["low"] <= pos["sl"]:  # 触发止损
                balance -= SL * LOT_SIZE * 10  # 计算损失
                positions.remove(pos)
            elif row["high"] >= pos["tp"]:  # 触发止盈
                balance += TP * LOT_SIZE * 10  # 计算盈利
                positions.remove(pos)
        elif pos["type"] == "SELL":
            if row["high"] >= pos["sl"]:  # 触发止损
                balance -= SL * LOT_SIZE * 10
                positions.remove(pos)
            elif row["low"] <= pos["tp"]:  # 触发止盈
                balance += TP * LOT_SIZE * 10
                positions.remove(pos)

    equity_curve.append(balance)

# 计算最终盈亏
profit = balance - capital
# win_rate = (profit > 0).sum() / len(df)

# 绘制资金曲线
import matplotlib.pyplot as pt
from matplotlib import style 
style.use('ggplot')

pt.figure(figsize=(10, 5))
pt.plot(df["time"], equity_curve, label="Equity Curve")
pt.xlabel("Time")
pt.ylabel("Balance")
pt.title("Backtest Equity Curve")
pt.legend()
pt.show()

# 输出回测结果
print(f"最终资金: ${balance:.2f}")
print(f"总盈亏: ${profit:.2f}")
