import MetaTrader5 as mt5
import pandas as pd
import talib as ta
import time

# 设置参数
LOT_SIZE = 0.1  # 交易手数
STOP_LOSS = 50  # 止损点数
TAKE_PROFIT = 100  # 止盈点数
MAGIC_NUMBER = 101  # EA 识别编号

symbol = "XAUUSD"
# 初始化 MT5
if not mt5.initialize():
    print("初始化失败")
    quit()

# 获取历史数据并计算 EMA
def get_ema(period, count=100):
    # 获取历史收盘价
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
    if rates is None:
        print("获取历史数据失败")
        return None
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 计算 EMA
    df['ema'] = ta.EMA(df['close'], timeperiod=period)
    return df['ema'].iloc[-1]  # 返回最新的 EMA 值

# 检查是否有持仓
def check_existing_orders():
    buy_exists = False
    sell_exists = False
    orders = mt5.orders_get()
    if orders is None:
        return False, False
    for order in orders:
        if order.magic == MAGIC_NUMBER:
            if order.type == mt5.ORDER_TYPE_BUY:
                buy_exists = True
            elif order.type == mt5.ORDER_TYPE_SELL:
                sell_exists = True
    return buy_exists, sell_exists

# 交易逻辑
def on_tick():
    ema50 = get_ema(50)
    ema200 = get_ema(200)
    price = mt5.symbol_info_tick(symbol).bid

    if ema50 is None or ema200 is None:
        return

    # 检查是否有持仓
    buy_exists, sell_exists = check_existing_orders()

    # 买入信号：50EMA 上穿 200EMA
    if ema50 > ema200 and not buy_exists:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": LOT_SIZE,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price - STOP_LOSS * mt5.symbol_info(symbol).point,
            "tp": price + TAKE_PROFIT * mt5.symbol_info(symbol).point,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": "Buy Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("买入订单失败:", result.comment)

    # 卖出信号：50EMA 下穿 200EMA
    if ema50 < ema200 and not sell_exists:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5.symbol(),
            "volume": LOT_SIZE,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price + STOP_LOSS * mt5.symbol_info(symbol).point,
            "tp": price - TAKE_PROFIT * mt5.symbol_info(symbol).point,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": "Sell Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("卖出订单失败:", result.comment)

# 主循环
while True:
    on_tick()
    print('🤖 Bot 正在运行...')
    time.sleep(1)  # 每秒检查一次