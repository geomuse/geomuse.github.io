"""
优化版EMA交叉策略
包含趋势过滤、移动止损、波动率控制
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import StrategyConfig


class EMAStrategy:
    """改进的EMA交叉策略"""
    
    def __init__(self, 
                 fast_period: int = 50,  # 改为50周期
                 slow_period: int = 200,  # 改为200周期
                 adx_period: int = 14,
                 adx_threshold: float = 25.0,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 trailing_atr_multiplier: float = 3.0):
        """
        初始化改进策略
        
        Args:
            fast_period: 快速EMA周期（50）
            slow_period: 慢速EMA周期（200）
            adx_period: ADX周期
            adx_threshold: ADX阈值（>25才开仓）
            atr_period: ATR周期
            atr_multiplier: ATR止损倍数
            trailing_atr_multiplier: 移动止损ATR倍数
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        
        # 当前持仓状态
        self.position = 0  # 0=无持仓, 1=多头, -1=空头
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.highest_price = 0.0  # 用于移动止损
        self.lowest_price = 0.0
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算EMA"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR（平均真实波幅）"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算ADX（平均趋向指数）
        ADX > 25: 趋势明显
        ADX < 20: 震荡市
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 计算+DM和-DM
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # 计算ATR
        atr = self.calculate_atr(df, period)
        
        # 计算+DI和-DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # 计算DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（带趋势过滤）
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            添加了EMA、ADX、ATR和信号列的DataFrame
        """
        # 计算快速和慢速EMA
        df['ema_fast'] = self.calculate_ema(df['close'], self.fast_period)
        df['ema_slow'] = self.calculate_ema(df['close'], self.slow_period)
        
        # 计算ADX (趋势强度)
        df['adx'] = self.calculate_adx(df, self.adx_period)
        
        # 计算ATR (波动率)
        df['atr'] = self.calculate_atr(df, self.atr_period)
        
        # 初始化信号列
        df['signal'] = 0
        
        # 检测金叉和死叉
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        # 金叉: 快线上穿慢线 AND ADX > 阈值（趋势明确）
        golden_cross = (df['ema_diff_prev'] < 0) & (df['ema_diff'] > 0) & (df['adx'] > self.adx_threshold)
        df.loc[golden_cross, 'signal'] = 1  # 买入信号
        
        # 死叉: 快线下穿慢线 AND ADX > 阈值
        death_cross = (df['ema_diff_prev'] > 0) & (df['ema_diff'] < 0) & (df['adx'] > self.adx_threshold)
        df.loc[death_cross, 'signal'] = -1  # 卖出信号
        
        # 平仓信号：ADX下降到20以下（趋势消失）
        weak_trend = df['adx'] < 20
        df.loc[weak_trend, 'signal'] = 0  # 平仓信号会在get_action中处理
        
        return df
    
    def get_action(self, current_price: float, signal: int, 
                   atr: float, adx: float) -> Dict:
        """
        根据信号和当前持仓决定交易动作（带移动止损）
        
        Args:
            current_price: 当前价格
            signal: 信号 (1=买入, -1=卖出, 0=无信号)
            atr: 当前ATR值
            adx: 当前ADX值
            
        Returns:
            交易动作字典
        """
        action_dict = {
            'action': 'hold',
            'sl': 0.0,
            'tp': 0.0,
            'reason': ''
        }
        
        # 更新最高/最低价（用于移动止损）
        if self.position == 1:  # 多头
            self.highest_price = max(self.highest_price, current_price)
            # 移动止损：价格新高后，止损跟进
            trailing_stop = self.highest_price - self.trailing_atr_multiplier * atr
            self.stop_loss = max(self.stop_loss, trailing_stop)
            
        elif self.position == -1:  # 空头
            self.lowest_price = min(self.lowest_price, current_price)
            # 移动止损
            trailing_stop = self.lowest_price + self.trailing_atr_multiplier * atr
            self.stop_loss = min(self.stop_loss, trailing_stop)
        
        # 检查移动止损
        if self.position != 0:
            if self._check_stop_loss(current_price):
                action_dict['action'] = 'close'
                action_dict['reason'] = 'Trailing SL hit'
                return action_dict
            
            # 如果ADX下降到20以下，平仓
            if adx < 20:
                action_dict['action'] = 'close'
                action_dict['reason'] = 'Weak trend (ADX < 20)'
                return action_dict
        
        # 无信号，保持当前状态
        if signal == 0:
            return action_dict
        
        # 买入信号
        if signal == 1 and adx > self.adx_threshold:
            if self.position == 0:
                # 无持仓，开多
                action_dict['action'] = 'buy'
                action_dict['sl'] = current_price - self.atr_multiplier * atr
                action_dict['tp'] = 0  # 不设固定止盈，使用移动止损
                action_dict['reason'] = f'Golden cross (ADX={adx:.1f})'
                
            elif self.position == -1:
                # 持有空头，先平仓再开多
                action_dict['action'] = 'close_and_buy'
                action_dict['sl'] = current_price - self.atr_multiplier * atr
                action_dict['tp'] = 0
                action_dict['reason'] = 'Reverse: Golden cross'
        
        # 卖出信号
        elif signal == -1 and adx > self.adx_threshold:
            if self.position == 0:
                # 无持仓，开空
                action_dict['action'] = 'sell'
                action_dict['sl'] = current_price + self.atr_multiplier * atr
                action_dict['tp'] = 0  # 不设固定止盈
                action_dict['reason'] = f'Death cross (ADX={adx:.1f})'
                
            elif self.position == 1:
                #持有多头，先平仓再开空
                action_dict['action'] = 'close_and_sell'
                action_dict['sl'] = current_price + self.atr_multiplier * atr
                action_dict['tp'] = 0
                action_dict['reason'] = 'Reverse: Death cross'
        
        return action_dict
    
    def open_position(self, position_type: int, entry_price: float, 
                      stop_loss: float, take_profit: float = 0):
        """开仓"""
        self.position = position_type
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        
        # 初始化追踪价格
        if position_type == 1:  # 多头
            self.highest_price = entry_price
        else:  # 空头
            self.lowest_price = entry_price
    
    def close_position(self):
        """平仓"""
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """检查是否触发止损"""
        if self.position == 1:  # 多头
            return current_price <= self.stop_loss
        elif self.position == -1:  # 空头
            return current_price >= self.stop_loss
        return False
    
    def get_position_pnl(self, current_price: float) -> float:
        """获取当前持仓盈亏"""
        if self.position == 0:
            return 0.0
        
        if self.position == 1:  # 多头
            return current_price - self.entry_price
        else:  # 空头
            return self.entry_price - current_price
