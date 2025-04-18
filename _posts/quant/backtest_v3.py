import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from datetime import datetime
import talib as ta
from matplotlib import style
style.use('ggplot')

# 连接 MT5
if not mt5.initialize():
    print("MT5 连接失败")
    quit()

# 交易品种 & 参数
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01
SL = 20  # 止损 20 Pips
TP = 40  # 止盈 40 Pips
SPREAD = 0.0002  # 2 Pips
SLIPPAGE = 0.0001  # 滑点 1 Pip
COMMISSION_PER_LOT = 7  # 每手佣金（假设单边 $7）

# 获取历史数据
rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 500)
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

# 计算均线
df["short"] = ta.EMA(df["close"],5)
df["long"] = ta.EMA(df["close"],20)

# 计算信号
df["signal"] = 0
df.loc[(df["short"] > df["long"]) & (df["short"].shift(1) <= df["long"].shift(1)), "signal"] = 1
df.loc[(df["short"] < df["long"]) & (df["short"].shift(1) >= df["long"].shift(1)), "signal"] = -1

# 模拟交易
capital = 1000  # 初始资金
balance = capital
positions = []
equity_curve = []
trade_results = []  # 记录每笔交易盈亏

for i, row in df.iterrows():
    if row["signal"] == 1:  # 买入
        entry_price = row["close"] + SPREAD + SLIPPAGE  # 计算买入点（含点差 & 滑点）
        # sl_price = entry_price - SL * 0.0001
        sl_price = 0
        # tp_price = entry_price + TP * 0.0001
        tp_price = 0
        positions.append({"type": "BUY", "entry": entry_price, "sl": sl_price, "tp": tp_price})

    elif row["signal"] == -1:  # 卖出
        entry_price = row["close"] - SLIPPAGE  # 计算卖出点（含滑点）
        sl_price = 0
        tp_price = 0
        # sl_price = entry_price + SL * 0.0001
        # tp_price = entry_price - TP * 0.0001
        
        positions.append({"type": "SELL", "entry": entry_price, "sl": sl_price, "tp": tp_price})

    # 检查止盈止损
    for pos in positions[:]:  # 遍历持仓
        if pos["type"] == "BUY":
            if row["low"] <= pos["sl"]:  # 触发止损
                loss = -SL * LOT_SIZE * 10 - COMMISSION_PER_LOT * LOT_SIZE  # 计算损失（含佣金）
                balance += loss
                trade_results.append(loss)
                positions.remove(pos)
            elif row["high"] >= pos["tp"]:  # 触发止盈
                profit = TP * LOT_SIZE * 10 - COMMISSION_PER_LOT * LOT_SIZE  # 计算盈利（含佣金）
                balance += profit
                trade_results.append(profit)
                positions.remove(pos)

        elif pos["type"] == "SELL":
            if row["high"] >= pos["sl"]:  # 触发止损
                loss = -SL * LOT_SIZE * 10 - COMMISSION_PER_LOT * LOT_SIZE
                balance += loss
                trade_results.append(loss)
                positions.remove(pos)
            elif row["low"] <= pos["tp"]:  # 触发止盈
                profit = TP * LOT_SIZE * 10 - COMMISSION_PER_LOT * LOT_SIZE
                balance += profit
                trade_results.append(profit)
                positions.remove(pos)

    equity_curve.append(balance)  # 记录资金曲线

# 计算回测指标
total_trades = len(trade_results)
winning_trades = len([r for r in trade_results if r > 0])
win_rate = winning_trades / total_trades if total_trades > 0 else 0
total_profit = sum([r for r in trade_results if r > 0])
total_loss = abs(sum([r for r in trade_results if r < 0]))
profit_factor = total_profit / total_loss if total_loss > 0 else 0

# 计算最大回撤
equity_series = pd.Series(equity_curve)
rolling_max = equity_series.cummax()
drawdown = equity_series - rolling_max
max_drawdown = drawdown.min()

# 结果输出
print(f"✅ 总交易数: {total_trades}")
print(f"✅ 胜率: {win_rate:.2%}")
print(f"✅ 盈亏比（Profit Factor）: {profit_factor:.2f}")
print(f"✅ 最大回撤（Max Drawdown）: {max_drawdown:.2f}")

# 绘制资金曲线
pt.figure(figsize=(15, 5))

pt.subplot(2,1,1)
pt.plot(equity_curve, label="Equity Curve", color="blue")
pt.ylabel("Balance")
pt.title("Backtest Equity Curve")
pt.legend()


pt.subplot(2,1,2)
pt.ylim(-2,2)
pt.plot(df["signal"],label='Signal',color='red')
pt.legend()

pt.show()
