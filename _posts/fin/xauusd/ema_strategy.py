"""
EMA交叉策略核心模块
实现EMA计算、信号生成和交易逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import StrategyConfig


class EMAStrategy:
    """EMA交叉策略"""
    
    def __init__(self, 
                 fast_period: int = StrategyConfig.FAST_EMA,
                 slow_period: int = StrategyConfig.SLOW_EMA,
                 stop_loss_points: float = StrategyConfig.STOP_LOSS_POINTS,
                 take_profit_points: float = StrategyConfig.TAKE_PROFIT_POINTS):
        """
        初始化策略
        
        Args:
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            stop_loss_points: 止损点数
            take_profit_points: 止盈点数
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points
        
        # 当前持仓状态
        self.position = 0  # 0=无持仓, 1=多头, -1=空头
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        计算EMA（指数移动平均线）
        
        Args:
            prices: 价格序列
            period: EMA周期
            
        Returns:
            EMA值序列
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            添加了EMA和信号列的DataFrame
        """
        # 计算快速和慢速EMA
        df['ema_fast'] = self.calculate_ema(df['close'], self.fast_period)
        df['ema_slow'] = self.calculate_ema(df['close'], self.slow_period)
        
        # 初始化信号列
        df['signal'] = 0
        
        # 检测金叉和死叉
        # 金叉：快线从下方穿越慢线（买入信号）
        # 死叉：快线从上方穿越慢线（卖出信号）
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        # 金叉: 前一根K线快线<慢线，当前K线快线>慢线
        golden_cross = (df['ema_diff_prev'] < 0) & (df['ema_diff'] > 0)
        df.loc[golden_cross, 'signal'] = 1  # 买入信号
        
        # 死叉: 前一根K线快线>慢线，当前K线快线<慢线
        death_cross = (df['ema_diff_prev'] > 0) & (df['ema_diff'] < 0)
        df.loc[death_cross, 'signal'] = -1  # 卖出信号
        
        return df
    
    def get_action(self, current_price: float, signal: int, point_value: float = 0.0001) -> Dict:
        """
        根据信号和当前持仓决定交易动作
        
        Args:
            current_price: 当前价格
            signal: 信号 (1=买入, -1=卖出, 0=无信号)
            point_value: 点值（例如EURUSD为0.0001）
            
        Returns:
            交易动作字典 {'action': 'buy'/'sell'/'close'/'hold', 'sl': 止损价, 'tp': 止盈价}
        """
        action_dict = {
            'action': 'hold',
            'sl': 0.0,
            'tp': 0.0,
            'reason': ''
        }
        
        # 检查止损止盈
        if self.position != 0:
            if self._check_stop_loss(current_price) or self._check_take_profit(current_price):
                action_dict['action'] = 'close'
                action_dict['reason'] = 'SL/TP hit'
                return action_dict
        
        # 无信号，保持当前状态
        if signal == 0:
            return action_dict
        
        # 买入信号
        if signal == 1:
            if self.position == 0:
                # 无持仓，开多
                action_dict['action'] = 'buy'
                action_dict['sl'] = current_price - self.stop_loss_points * point_value
                action_dict['tp'] = current_price + self.take_profit_points * point_value
                action_dict['reason'] = 'Golden cross'
                
            elif self.position == -1:
                # 持有空头，先平仓再开多
                action_dict['action'] = 'close_and_buy'
                action_dict['sl'] = current_price - self.stop_loss_points * point_value
                action_dict['tp'] = current_price + self.take_profit_points * point_value
                action_dict['reason'] = 'Reverse: Golden cross'
        
        # 卖出信号
        elif signal == -1:
            if self.position == 0:
                # 无持仓，开空
                action_dict['action'] = 'sell'
                action_dict['sl'] = current_price + self.stop_loss_points * point_value
                action_dict['tp'] = current_price - self.take_profit_points * point_value
                action_dict['reason'] = 'Death cross'
                
            elif self.position == 1:
                # 持有多头，先平仓再开空
                action_dict['action'] = 'close_and_sell'
                action_dict['sl'] = current_price + self.stop_loss_points * point_value
                action_dict['tp'] = current_price - self.take_profit_points * point_value
                action_dict['reason'] = 'Reverse: Death cross'
        
        return action_dict
    
    def open_position(self, position_type: int, entry_price: float, 
                      stop_loss: float, take_profit: float):
        """
        开仓
        
        Args:
            position_type: 1=多头, -1=空头
            entry_price: 入场价格
            stop_loss: 止损价
            take_profit: 止盈价
        """
        self.position = position_type
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def close_position(self):
        """平仓"""
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """检查是否触发止损"""
        if self.position == 1:  # 多头
            return current_price <= self.stop_loss
        elif self.position == -1:  # 空头
            return current_price >= self.stop_loss
        return False
    
    def _check_take_profit(self, current_price: float) -> bool:
        """检查是否触发止盈"""
        if self.position == 1:  # 多头
            return current_price >= self.take_profit
        elif self.position == -1:  # 空头
            return current_price <= self.take_profit
        return False
    
    def get_position_pnl(self, current_price: float) -> float:
        """
        获取当前持仓盈亏
        
        Args:
            current_price: 当前价格
            
        Returns:
            盈亏（点数）
        """
        if self.position == 0:
            return 0.0
        
        if self.position == 1:  # 多头
            return current_price - self.entry_price
        else:  # 空头
            return self.entry_price - current_price
