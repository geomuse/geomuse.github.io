"""
MT5连接器
用于连接MetaTrader 5并执行交易操作
"""

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import MT5Config

class MT5Connector:
    """MT5连接器"""
    
    def __init__(self, account: int = None, password: str = None, server: str = None):
        """
        初始化MT5连接器
        
        Args:
            account: MT5账户号
            password: 密码
            server: 服务器地址
        """
        self.account = account or MT5Config.ACCOUNT
        self.password = password or MT5Config.PASSWORD
        self.server = server or MT5Config.SERVER
        self.connected = False
    
    def connect(self) -> bool:
        """
        连接MT5
        
        Returns:
            是否连接成功
        """
        if not MT5_AVAILABLE:
            print("❌ 错误: MetaTrader5 库未安装。实盘功能仅支持 Windows。")
            return False
            
        # 初始化MT5
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return False
        
        # 如果提供了账户信息，则登录
        if self.account and self.password and self.server:
            if not mt5.login(self.account, self.password, self.server):
                print(f"❌ MT5登录失败: {mt5.last_error()}")
                mt5.shutdown()
                return False
            print(f"✓ 已登录MT5账户: {self.account}")
        else:
            print("✓ MT5已初始化（使用当前登录账户）")
        
        self.connected = True
        return True
    
    def disconnect(self):
        """断开连接"""
        if self.connected and MT5_AVAILABLE:
            mt5.shutdown()
            self.connected = False
            print("✓ 已断开MT5连接")
    
    def get_account_info(self) -> Dict:
        """
        获取账户信息
        
        Returns:
            账户信息字典
        """
        if not self.connected:
            raise Exception("未连接到MT5")
        
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception(f"无法获取账户信息: {mt5.last_error()}")
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'profit': account_info.profit,
            'login': account_info.login,
            'server': account_info.server,
            'leverage': account_info.leverage
        }
    
    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """
        获取当前价格
        
        Args:
            symbol: 交易品种
            
        Returns:
            (bid, ask) 元组
        """
        if not self.connected:
            raise Exception("未连接到MT5")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Exception(f"无法获取{symbol}价格: {mt5.last_error()}")
        
        return tick.bid, tick.ask
    
    def get_bars(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期 ('5min', '1H', etc.)
            count: K线数量
            
        Returns:
            K线数据DataFrame
        """
        if not self.connected:
            raise Exception("未连接到MT5")
        
        # 转换时间周期
        tf_map = {
            '1min': mt5.TIMEFRAME_M1,
            '5min': mt5.TIMEFRAME_M5,
            '15min': mt5.TIMEFRAME_M15,
            '30min': mt5.TIMEFRAME_M30,
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1,
        }
        
        mt5_timeframe = tf_map.get(timeframe)
        if mt5_timeframe is None:
            raise ValueError(f"不支持的时间周期: {timeframe}")
        
        # 获取K线数据
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            raise Exception(f"无法获取K线数据: {mt5.last_error()}")
        
        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        return df
    
    def place_order(self, symbol: str, order_type: str, volume: float,
                   price: float = None, sl: float = None, tp: float = None,
                   comment: str = "") -> Dict:
        """
        下单
        
        Args:
            symbol: 交易品种
            order_type: 订单类型 ('buy', 'sell')
            volume: 手数
            price: 价格（市价单可为None）
            sl: 止损价
            tp: 止盈价
            comment: 备注
            
        Returns:
            订单结果
        """
        if not self.connected:
            raise Exception("未连接到MT5")
        
        # 获取交易品种信息
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"无法获取{symbol}信息: {mt5.last_error()}")
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise Exception(f"无法选择{symbol}")
        
        # 准备订单请求
        point = symbol_info.point
        
        if order_type.lower() == 'buy':
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            if price is None:
                price = mt5.symbol_info_tick(symbol).ask
        elif order_type.lower() == 'sell':
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            if price is None:
                price = mt5.symbol_info_tick(symbol).bid
        else:
            raise ValueError(f"不支持的订单类型: {order_type}")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 添加止损和止盈
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # 发送订单
        result = mt5.order_send(request)
        
        if result is None:
            raise Exception(f"下单失败: {mt5.last_error()}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"下单失败: {result.comment} (code: {result.retcode})")
        
        return {
            'order': result.order,
            'volume': result.volume,
            'price': result.price,
            'retcode': result.retcode,
            'comment': result.comment
        }
    
    def close_position(self, ticket: int, volume: float = None) -> Dict:
        """
        平仓
        
        Args:
            ticket: 持仓票据号
            volume: 平仓手数（可选，默认为全部）
            
        Returns:
            平仓结果
        """
        if not self.connected:
            raise Exception("未连接到MT5")
        
        # 获取持仓信息
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            raise Exception(f"无法找到持仓: {ticket}")
        
        position = position[0]
        
        # 确定平仓量
        close_volume = volume if volume is not None else position.volume
        
        # 准备平仓请求（反向订单）
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 发送平仓请求
        result = mt5.order_send(request)
        
        if result is None:
            raise Exception(f"平仓失败: {mt5.last_error()}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"平仓失败: {result.comment} (code: {result.retcode})")
        
        return {
            'order': result.order,
            'volume': result.volume,
            'price': result.price,
            'profit': position.profit,
            'retcode': result.retcode,
            'comment': result.comment
        }
    
    def get_positions(self) -> List[Dict]:
        """
        获取当前持仓
        
        Returns:
            持仓列表
        """
        if not self.connected:
            raise Exception("未连接到MT5")
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'time': datetime.fromtimestamp(pos.time)
            })
        
        return result
