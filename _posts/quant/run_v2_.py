import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, time
import talib as ta 
import time as t 

# 连接 MT5
if not mt5.initialize():
    print("MT5 连接失败")
    quit()

# 交易参数
SYMBOL = "XAUUSD"      # 交易品种
LOT_SIZE = 0.01         # 手数
SL = 20                # 止损点（Pips）
TP = 40                # 止盈点（Pips）

# 设定交易时间（09:30 - 15:30）
TRADING_START = time(1, 30)
TRADING_END = time(23, 30)

def is_trading_time():
    now = datetime.now().time()
    return TRADING_START <= now <= TRADING_END

# 获取 K 线数据
def get_data(symbol, timeframe, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

last_signal = None  # 记录上一次的信号

def check_signal():
    global last_signal
    df = get_data(SYMBOL, mt5.TIMEFRAME_M5, 20)
    df["short"] = ta.EMA(df["close"], 5)
    df["long"] = ta.EMA(df["close"], 20)
    # df.dropna(inplace=True)

    if df["short"].iloc[-1] > df["long"].iloc[-1] and df["short"].iloc[-2] <= df["long"].iloc[-2]:
        if last_signal != "BUY":
            last_signal = "BUY"
            return "BUY"
    elif df["short"].iloc[-1] < df["long"].iloc[-1] and df["short"].iloc[-2] >= df["long"].iloc[-2]:
        if last_signal != "SELL":
            last_signal = "SELL"
            return "SELL"
    return None

# 下单函数
def place_order(action):
    price = mt5.symbol_info_tick(SYMBOL).ask if action == "BUY" else mt5.symbol_info_tick(SYMBOL).bid
    # sl = price - SL * 0.0001 if action == "BUY" else price + SL * 0.0001
    # tp = price + TP * 0.0001 if action == "BUY" else price - TP * 0.0001
    sl = 0
    tp = 0 

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 101,
        "comment": "Intraday Trading",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"{action} 订单成功！")
    else:
        print(f"{action} 订单失败: {result.comment}")

# 关闭盈利超过 5 元的持仓
def close_profitable_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
        return
    
    for pos in positions:
        profit = pos.profit  # 获取当前持仓的盈利
        if profit > 5:  # 只有盈利大于 5 才平仓
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(SYMBOL).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 10,
                "magic": 1000,
                "comment": "Closing profitable position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"盈利 {profit:.2f} 元，已成功平仓")
            else:
                print(f"平仓失败: {result.comment}")


# 关闭所有持仓（用于收盘前）
def close_all_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    for pos in positions:
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(SYMBOL).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 10,
            "magic": 1000,
            "comment": "Closing all positions",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)

# 交易循环
while True:
    now = datetime.now()

    # 检查是否在交易时间
    if is_trading_time():
        signal = check_signal()
        if signal:
            place_order(signal)

    # 检查盈利是否超过 5 元
    close_profitable_positions()

    # 收盘前 10 分钟平仓
    if now.time() >= (datetime.combine(now.date(), TRADING_END) - pd.Timedelta(minutes=10)).time():
        close_all_positions()
        print("已关闭所有持仓，等待市场重新开放。")
        t.sleep(600)  # 直接等待 10 分钟，避免多次触发

    t.sleep(1)  # 每 10 秒检测一次

