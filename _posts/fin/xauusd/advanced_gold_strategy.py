"""
Multi-Filter Adaptive Gold Strategy (MFAG)
适合黄金和高波动性货币对的多维度自适应策略

核心特点：
1. 五维度信号过滤（Heikin-Ashi + RSI + BB + SuperTrend + Volume）
2. 自适应风险管理（ATR动态止损 + 三段式止盈）
3. Kelly Criterion仓位管理
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PositionTier:
    """分批止盈的仓位层级"""
    size_percent: float  # 该层级占总仓位的百分比
    entry_price: float
    take_profit: float
    is_closed: bool = False


class AdvancedGoldStrategy:
    """多维度自适应黄金策略"""
    
    def __init__(self,
                 # SuperTrend参数
                 supertrend_period: int = 10,
                 supertrend_multiplier: float = 3.0,
                 
                 # Bollinger Bands参数
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 bb_squeeze_threshold: float = 0.001,
                 
                 # RSI参数
                 rsi_period: int = 14,
                 rsi_overbought: float = 75,
                 rsi_oversold: float = 25,
                 
                 # Volume参数
                 volume_ma_period: int = 20,
                 volume_threshold: float = 1.2,
                 
                 # ATR参数
                 atr_period: int = 14,
                 atr_sl_multiplier: float = 2.0,
                 
                 # 风险管理
                 max_risk_per_trade: float = 0.015,
                 kelly_fraction: float = 0.25,
                 contract_size: float = 100.0):
        """
        初始化高级策略
        
        Args:
            supertrend_period: SuperTrend周期
            supertrend_multiplier: SuperTrend倍数
            bb_period: 布林带周期
            bb_std: 布林带标准差
            bb_squeeze_threshold: BB挤压阈值
            rsi_period: RSI周期
            rsi_overbought: RSI超买阈值
            rsi_oversold: RSI超卖阈值
            volume_ma_period: 成交量均线周期
            volume_threshold: 成交量倍数阈值
            atr_period: ATR周期
            atr_sl_multiplier: ATR止损倍数
            max_risk_per_trade: 单笔最大风险
            kelly_fraction: Kelly比例（保守）
            contract_size: 合约大小（黄金通常100）
        """
        # 指标参数
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        
        # 风险管理
        self.max_risk_per_trade = max_risk_per_trade
        self.kelly_fraction = kelly_fraction
        self.contract_size = contract_size
        
        # 持仓状态
        self.position = 0  # 0=无持仓, 1=多头, -1=空头
        self.position_tiers: List[PositionTier] = []  # 分批仓位
        self.entry_price = 0.0
        self.stop_loss = 0.0
        
        # 统计数据（用于Kelly计算）
        self.win_count = 0
        self.loss_count = 0
        self.total_win_amount = 0.0
        self.total_loss_amount = 0.0
        
    # ==================== 技术指标计算 ====================
    
    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Heikin-Ashi蜡烛图
        用于平滑价格噪音，更清晰地识别趋势
        """
        ha_df = df.copy()
        
        # HA Close = (O + H + L + C) / 4
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # HA Open = (previous HA Open + previous HA Close) / 2
        ha_df['ha_open'] = 0.0
        ha_df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2
        
        for i in range(1, len(ha_df)):
            ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2
        
        # HA High = max(H, HA Open, HA Close)
        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        
        # HA Low = min(L, HA Open, HA Close)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # HA趋势判断
        ha_df['ha_bullish'] = ha_df['ha_close'] > ha_df['ha_open']
        ha_df['ha_bearish'] = ha_df['ha_close'] < ha_df['ha_open']
        
        return ha_df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带及挤压检测
        BB挤压通常预示着即将发生突破
        """
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
        
        # BB宽度（标准化）
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB挤压检测（宽度小于阈值）
        df['bb_squeeze'] = df['bb_width'] < self.bb_squeeze_threshold
        
        # BB突破检测
        df['bb_breakout_up'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        df['bb_breakout_down'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        
        return df
    
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算SuperTrend指标
        强大的趋势跟踪指标，减少震荡市场的假信号
        """
        # 计算ATR
        df['atr'] = self.calculate_atr(df, self.atr_period)
        
        # 基本上下轨
        hl_avg = (df['high'] + df['low']) / 2
        df['basic_ub'] = hl_avg + (self.supertrend_multiplier * df['atr'])
        df['basic_lb'] = hl_avg - (self.supertrend_multiplier * df['atr'])
        
        # 最终上下轨
        df['final_ub'] = 0.0
        df['final_lb'] = 0.0
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 1  # 1=上升, -1=下降
        
        # 确保从有数据的行开始计算
        start_idx = max(self.supertrend_period, self.atr_period)
        
        # 初始化第一个有效值
        if start_idx < len(df):
            df.loc[start_idx, 'final_ub'] = df.loc[start_idx, 'basic_ub']
            df.loc[start_idx, 'final_lb'] = df.loc[start_idx, 'basic_lb']
            
        for i in range(start_idx + 1, len(df)):
            # Final Upper Band
            if df.loc[i, 'basic_ub'] < df.loc[i-1, 'final_ub'] or df.loc[i-1, 'close'] > df.loc[i-1, 'final_ub']:
                df.loc[i, 'final_ub'] = df.loc[i, 'basic_ub']
            else:
                df.loc[i, 'final_ub'] = df.loc[i-1, 'final_ub']
            
            # Final Lower Band
            if df.loc[i, 'basic_lb'] > df.loc[i-1, 'final_lb'] or df.loc[i-1, 'close'] < df.loc[i-1, 'final_lb']:
                df.loc[i, 'final_lb'] = df.loc[i, 'basic_lb']
            else:
                df.loc[i, 'final_lb'] = df.loc[i-1, 'final_lb']
            
            # SuperTrend
            if df.loc[i, 'close'] <= df.loc[i, 'final_ub']:
                df.loc[i, 'supertrend'] = df.loc[i, 'final_ub']
                df.loc[i, 'supertrend_direction'] = -1
            else:
                df.loc[i, 'supertrend'] = df.loc[i, 'final_lb']
                df.loc[i, 'supertrend_direction'] = 1
        
        return df
    
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
    
    def calculate_volume_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量动量
        高成交量确认趋势的有效性
        """
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = df['volume_ratio'] > self.volume_threshold
        
        return df
    
    # ==================== 信号生成 ====================
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        五维度信号过滤系统
        
        买入信号需满足ALL条件：
        1. Heikin-Ashi连续2根阳烛
        2. RSI < 75（非超买）
        3. BB挤压后向上突破或价格在上轨上方
        4. SuperTrend显示上升趋势
        5. 成交量 > 均线的1.2倍
        
        卖出信号类似但方向相反
        """
        # 计算所有指标
        df = self.calculate_heikin_ashi(df)
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_supertrend(df)
        df = self.calculate_volume_momentum(df)
        
        # 初始化信号
        df['signal'] = 0
        df['signal_strength'] = 0.0  # 信号强度（0-5）
        
        # Heikin-Ashi连续性检测
        df['ha_bullish_2'] = df['ha_bullish'] & df['ha_bullish'].shift(1)
        df['ha_bearish_2'] = df['ha_bearish'] & df['ha_bearish'].shift(1)
        
        # 计算信号强度
        for i in range(self.supertrend_period + 2, len(df)):
            strength = 0
            
            # 多头信号检测
            if (df.loc[i, 'ha_bullish_2'] and
                df.loc[i, 'rsi'] < self.rsi_overbought and
                df.loc[i, 'supertrend_direction'] == 1 and
                df.loc[i, 'high_volume']):
                
                # 计算信号强度
                if df.loc[i, 'ha_bullish_2']:
                    strength += 1
                if df.loc[i, 'rsi'] < self.rsi_overbought:
                    strength += 1
                if df.loc[i, 'bb_breakout_up'] or df.loc[i, 'close'] > df.loc[i, 'bb_upper']:
                    strength += 1
                if df.loc[i, 'supertrend_direction'] == 1:
                    strength += 1
                if df.loc[i, 'high_volume']:
                    strength += 1
                
                # 只有强度>=4才发出信号
                if strength >= 4:
                    df.loc[i, 'signal'] = 1
                    df.loc[i, 'signal_strength'] = strength
            
            # 空头信号检测
            elif (df.loc[i, 'ha_bearish_2'] and
                  df.loc[i, 'rsi'] > self.rsi_oversold and
                  df.loc[i, 'supertrend_direction'] == -1 and
                  df.loc[i, 'high_volume']):
                
                strength = 0
                if df.loc[i, 'ha_bearish_2']:
                    strength += 1
                if df.loc[i, 'rsi'] > self.rsi_oversold:
                    strength += 1
                if df.loc[i, 'bb_breakout_down'] or df.loc[i, 'close'] < df.loc[i, 'bb_lower']:
                    strength += 1
                if df.loc[i, 'supertrend_direction'] == -1:
                    strength += 1
                if df.loc[i, 'high_volume']:
                    strength += 1
                
                if strength >= 4:
                    df.loc[i, 'signal'] = -1
                    df.loc[i, 'signal_strength'] = strength
        
        return df
    
    # ==================== 仓位管理 ====================
    
    def calculate_kelly_risk_percent(self, win_rate: float = None, 
                                      avg_win: float = None, avg_loss: float = None) -> float:
        """
        Kelly Criterion风险比例计算
        Returns: 建议的风险百分比 (如 0.02 表示 2%)
        """
        # 如果没有足够的历史数据，使用默认值
        if win_rate is None or avg_win is None or avg_loss is None:
            if self.win_count + self.loss_count < 10:
                return self.max_risk_per_trade  # 至少10笔交易才用Kelly
            
            total_trades = self.win_count + self.loss_count
            win_rate = self.win_count / total_trades
            avg_win = self.total_win_amount / max(self.win_count, 1)
            avg_loss = abs(self.total_loss_amount) / max(self.loss_count, 1)
        
        # 计算Kelly百分比
        if avg_loss == 0:
            return self.max_risk_per_trade
        
        win_loss_ratio = avg_win / avg_loss
        kelly_pct = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        
        # 使用保守的Kelly比例
        kelly_pct = kelly_pct * self.kelly_fraction
        
        # 限制在最大风险范围内
        kelly_pct = max(0, min(kelly_pct, self.max_risk_per_trade))
        
        return kelly_pct
    
    def calculate_lot_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        """
        根据风险百分比和止损距离计算手数
        Lot Size = (Balance * Risk%) / (SL Distance * Contract Size)
        """
        risk_pct = self.calculate_kelly_risk_percent()
        risk_amount = balance * risk_pct
        
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance == 0:
            return 0.0
            
        # 计算每手波动价值
        value_per_lot_move = sl_distance * self.contract_size
        
        lot_size = risk_amount / value_per_lot_move
        
        # 简单的手数规整（0.01步长）
        lot_size = round(lot_size, 2)
        
        return max(lot_size, 0.01) # 最小0.01手
    
    def create_position_tiers(self, entry_price: float, atr: float, 
                             position_type: int) -> List[PositionTier]:
        """
        创建三段式止盈仓位
        
        第一段(30%): 1.5 ATR - 快速锁定利润
        第二段(40%): 3.0 ATR - 主要盈利部分
        第三段(30%): 移动止损 - 捕捉大趋势
        """
        tiers = []
        
        if position_type == 1:  # 多头
            tier1_tp = entry_price + 1.5 * atr
            tier2_tp = entry_price + 3.0 * atr
            tier3_tp = entry_price + 5.0 * atr  # 更激进的目标
        else:  # 空头
            tier1_tp = entry_price - 1.5 * atr
            tier2_tp = entry_price - 3.0 * atr
            tier3_tp = entry_price - 5.0 * atr
        
        tiers.append(PositionTier(0.30, entry_price, tier1_tp))
        tiers.append(PositionTier(0.40, entry_price, tier2_tp))
        tiers.append(PositionTier(0.30, entry_price, tier3_tp))
        
        return tiers
    
    # ==================== 交易逻辑 ====================
    
    def get_action(self, current_price: float, signal: int, signal_strength: float,
                   atr: float, supertrend_direction: int, balance: float) -> Dict:
        """
        根据信号决定交易动作
        
        Returns:
            action_dict包含：
            - action: 'buy', 'sell', 'close_tier_1', 'close_tier_2', 'close_tier_3', 'hold'
            - sl: 止损价
            - position_size: 仓位大小（基于Kelly）
            - tiers: 分批止盈设置
        """
        action_dict = {
            'action': 'hold',
            'sl': 0.0,
            'position_size': 0.0,
            'tiers': [],
            'reason': ''
        }
        
        # 检查分批止盈
        if self.position != 0:
            tier_closed = self._check_tier_profits(current_price)
            if tier_closed:
                action_dict['action'] = tier_closed
                action_dict['reason'] = f'Tier profit taken'
                return action_dict
            
            # 检查止损
            if self._check_stop_loss(current_price):
                action_dict['action'] = 'close'
                action_dict['reason'] = 'Stop loss hit'
                return action_dict
        
        # 无信号或信号强度不足
        if signal == 0 or signal_strength < 4:
            return action_dict
        
        # 计算实际仓位（Lots）
        
        # 买入信号
        if signal == 1:
            if self.position == 0:
                sl_price = current_price - self.atr_sl_multiplier * atr
                lot_size = self.calculate_lot_size(balance, current_price, sl_price)
                
                action_dict['action'] = 'buy'
                action_dict['sl'] = sl_price
                action_dict['position_size'] = lot_size
                action_dict['tiers'] = self.create_position_tiers(current_price, atr, 1)
                action_dict['reason'] = f'Multi-filter BUY (strength={signal_strength})'
            elif self.position == -1:
                sl_price = current_price - self.atr_sl_multiplier * atr
                lot_size = self.calculate_lot_size(balance, current_price, sl_price)
                
                action_dict['action'] = 'close_and_buy'
                action_dict['sl'] = sl_price
                action_dict['position_size'] = lot_size
                action_dict['tiers'] = self.create_position_tiers(current_price, atr, 1)
                action_dict['reason'] = 'Reverse to LONG'
        
        # 卖出信号
        elif signal == -1:
            if self.position == 0:
                sl_price = current_price + self.atr_sl_multiplier * atr
                lot_size = self.calculate_lot_size(balance, current_price, sl_price)
                
                action_dict['action'] = 'sell'
                action_dict['sl'] = sl_price
                action_dict['position_size'] = lot_size
                action_dict['tiers'] = self.create_position_tiers(current_price, atr, -1)
                action_dict['reason'] = f'Multi-filter SELL (strength={signal_strength})'
            elif self.position == 1:
                sl_price = current_price + self.atr_sl_multiplier * atr
                lot_size = self.calculate_lot_size(balance, current_price, sl_price)
                
                action_dict['action'] = 'close_and_sell'
                action_dict['sl'] = sl_price
                action_dict['position_size'] = lot_size
                action_dict['tiers'] = self.create_position_tiers(current_price, atr, -1)
                action_dict['reason'] = 'Reverse to SHORT'
        
        return action_dict
    
    def _check_tier_profits(self, current_price: float) -> Optional[str]:
        """检查分批止盈"""
        for i, tier in enumerate(self.position_tiers):
            if tier.is_closed:
                continue
            
            if self.position == 1:  # 多头
                if current_price >= tier.take_profit:
                    tier.is_closed = True
                    return f'close_tier_{i+1}'
            elif self.position == -1:  # 空头
                if current_price <= tier.take_profit:
                    tier.is_closed = True
                    return f'close_tier_{i+1}'
        
        return None
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """检查是否触发止损"""
        if self.position == 1:  # 多头
            return current_price <= self.stop_loss
        elif self.position == -1:  # 空头
            return current_price >= self.stop_loss
        return False
    
    def open_position(self, position_type: int, entry_price: float, 
                     stop_loss: float, tiers: List[PositionTier]):
        """开仓"""
        self.position = position_type
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.position_tiers = tiers
    
    def close_position(self, pnl: float):
        """平仓并更新统计"""
        if pnl > 0:
            self.win_count += 1
            self.total_win_amount += pnl
        else:
            self.loss_count += 1
            self.total_loss_amount += pnl
        
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.position_tiers = []
    
    def get_position_pnl(self, current_price: float) -> float:
        """获取当前持仓盈亏"""
        if self.position == 0:
            return 0.0
        
        if self.position == 1:  # 多头
            return current_price - self.entry_price
        else:  # 空头
            return self.entry_price - current_price
