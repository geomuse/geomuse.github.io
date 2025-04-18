import MetaTrader5 as mt5
import pandas as pd
import datetime

# 连接 MT5
mt5.initialize()

# 选择交易品种
symbol = "XAUUSD"

# 获取最近 500 根 K 线数据
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 500)

# 断开连接
mt5.shutdown()

# 转换为 DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# # 计算均线
# df['SMA_10'] = df['close'].rolling(window=10).mean()
# df['SMA_30'] = df['close'].rolling(window=30).mean()

# # 查看均线交叉点
# df['signal'] = df['SMA_10'] > df['SMA_30']

# 输出买卖信号
print(df)

df.to_csv('XAUUSD.csv',index=False)
