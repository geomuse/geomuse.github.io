import MetaTrader5 as mt5
import pandas as pd

LOT_SIZE = 0.1  # 交易手数
STOP_LOSS = 50  # 止损点数
TAKE_PROFIT = 100  # 止盈点数
MAGIC_NUMBER = 101  # EA 识别编号

# 初始化 MT5
if not mt5.initialize():
    print("初始化失败")
    quit()

# 获取历史数据
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M1  # 1分钟时间框架
start_pos = 0  # 从当前时间开始的第0个柱
count = 1000  # 获取1000根K线
rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

# 将数据转换为 DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

print(df.head())


import talib as ta

# 计算 EMA
df['ema50'] = ta.EMA(df['close'], timeperiod=50)
df['ema200'] = ta.EMA(df['close'], timeperiod=200)

# 生成交易信号
df['signal'] = 0
df.loc[df['ema50'] > df['ema200'], 'signal'] = 1  # 买入信号
df.loc[df['ema50'] < df['ema200'], 'signal'] = -1  # 卖出信号

print(df[['time', 'close', 'ema50', 'ema200', 'signal']].tail())

# 初始化变量
position = 0  # 0表示无持仓，1表示持有多头，-1表示持有空头
entry_price = 0
exit_price = 0
trades = []

# 模拟交易
for i in range(1, len(df)):
    if df['signal'].iloc[i] == 1 and position != 1:  # 买入信号
        if position == -1:
            # 平空头
            exit_price = df['close'].iloc[i]
            trades.append(('sell', entry_price, exit_price))
        # 开多头
        position = 1
        entry_price = df['close'].iloc[i]
    elif df['signal'].iloc[i] == -1 and position != -1:  # 卖出信号
        if position == 1:
            # 平多头
            exit_price = df['close'].iloc[i]
            trades.append(('buy', entry_price, exit_price))
        # 开空头
        position = -1
        entry_price = df['close'].iloc[i]

# 打印交易记录
for trade in trades:
    print(trade)

# 计算总盈亏
total_profit = 0
for trade in trades:
    if trade[0] == 'buy':
        profit = (trade[2] - trade[1]) * LOT_SIZE
    else:
        profit = (trade[1] - trade[2]) * LOT_SIZE
    total_profit += profit

print(f"总盈亏: {total_profit}")

# 计算胜率
win_trades = [trade for trade in trades if (trade[2] - trade[1]) * LOT_SIZE > 0]
win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
print(f"胜率: {win_rate * 100:.2f}%")

# 计算最大回撤
max_drawdown = 0
peak = df['close'].iloc[0]
for price in df['close']:
    if price > peak:
        peak = price
    drawdown = (peak - price) / peak
    if drawdown > max_drawdown:
        max_drawdown = drawdown

print(f"最大回撤: {max_drawdown * 100:.2f}%")