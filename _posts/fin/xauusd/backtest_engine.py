"""
回测引擎
模拟策略在历史数据上的交易表现
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from ema_strategy import EMAStrategy
from config import BacktestConfig


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, 
                 strategy: EMAStrategy,
                 initial_balance: float = BacktestConfig.INITIAL_BALANCE,
                 commission: float = BacktestConfig.COMMISSION,
                 slippage_points: float = BacktestConfig.SLIPPAGE_POINTS,
                 point_value: float = 0.0001,
                 contract_size: float = 100000.0):
        """
        初始化回测引擎
        
        Args:
            strategy: 交易策略实例
            initial_balance: 初始资金
            commission: 手续费（每手）
            slippage_points: 滑点（点）
            point_value: 点值（如EURUSD为0.0001）
            contract_size: 合约大小（标准手单位，默认100,000）
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage_points = slippage_points
        self.point_value = point_value
        self.contract_size = contract_size
        
        # 账户状态
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin = 0.0
        
        # 交易记录
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def run(self, df: pd.DataFrame, lot_size: float = 0.01) -> Dict:
        """
        运行回测
        
        Args:
            df: 包含OHLC数据的DataFrame
            lot_size: 每次交易手数 (如果策略返回了position_size，则忽略此参数)
            
        Returns:
            回测结果字典
        """
        print("=" * 60)
        print("开始回测...")
        print(f"初始资金: ${self.initial_balance:,.2f}")
        print(f"合约大小: {self.contract_size:,.0f}")
        print(f"数据长度: {len(df)} 根K线")
        print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        print("=" * 60)
        
        # 生成交易信号
        df = self.strategy.generate_signals(df)
        
        # 初始化记录
        current_trade = None
        
        # 遍历每根K线
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # 跳过数据不足的初始阶段
            if pd.isna(row.get('ema_fast', row.get('supertrend'))): # 兼容不同策略
                continue
            
            current_price = row['close']
            signal = row.get('signal', 0)
            
            # 获取ATR和ADX（如果存在）- 用于改进策略
            atr = row.get('atr', 0.01)  # 默认0.01
            adx = row.get('adx', 25.0)  # 默认25
            
            # 获取交易动作（根据策略类型传递不同参数）
            try:
                # 尝试调用高级策略的get_action
                if hasattr(self.strategy, 'supertrend_period'): # AdvancedGoldStrategy check
                     signal_strength = row.get('signal_strength', 0)
                     supertrend_direction = row.get('supertrend_direction', 1)
                     # Advanced strategy expects balance to calculate dynamic position size
                     action_dict = self.strategy.get_action(
                        current_price, signal, signal_strength, 
                        atr, supertrend_direction, self.balance
                    )
                else:
                    # 尝试调用改进策略的get_action
                    action_dict = self.strategy.get_action(current_price, signal, atr, adx)
            except (TypeError, AttributeError):
                # 如果是旧策略，使用point_value参数
                try:
                    action_dict = self.strategy.get_action(current_price, signal, self.point_value)
                except TypeError:
                     # Fallback for very old strategy signatures
                     action_dict = self.strategy.get_action(current_price, signal)

            
            action = action_dict['action']
            
            # 确定手数
            # 如果策略返回了 clear 'position_size'，使用它，否则使用默认 lot_size
            trade_lot_size = action_dict.get('position_size', lot_size)
            if trade_lot_size <= 0: trade_lot_size = lot_size # Fallback if 0 returned

            
            # 执行交易动作
            if action == 'buy':
                # 开多仓
                entry_price = self._apply_slippage(current_price, 'buy')
                
                # Check if strategy requires tiers (Advanced Strategy)
                tiers = action_dict.get('tiers', [])
                if hasattr(self.strategy, 'open_position') and 'tiers' in action_dict:
                     self.strategy.open_position(1, entry_price, action_dict['sl'], tiers)
                else:
                     self.strategy.open_position(1, entry_price, action_dict['sl'], action_dict.get('tp', 0))

                current_trade = {
                    'entry_time': row['datetime'],
                    'entry_price': entry_price,
                    'type': 'BUY',
                    'lot_size': trade_lot_size,
                    'sl': action_dict['sl'],
                    'tp': action_dict.get('tp', tiers[-1].take_profit if tiers else 0),
                    'reason': action_dict.get('reason', '')
                }
                
            elif action == 'sell':
                # 开空仓
                entry_price = self._apply_slippage(current_price, 'sell')
                
                tiers = action_dict.get('tiers', [])
                if hasattr(self.strategy, 'open_position') and 'tiers' in action_dict:
                     self.strategy.open_position(-1, entry_price, action_dict['sl'], tiers)
                else:
                     self.strategy.open_position(-1, entry_price, action_dict['sl'], action_dict.get('tp', 0))
                
                current_trade = {
                    'entry_time': row['datetime'],
                    'entry_price': entry_price,
                    'type': 'SELL',
                    'lot_size': trade_lot_size,
                    'sl': action_dict['sl'],
                    'tp': action_dict.get('tp', tiers[-1].take_profit if tiers else 0),
                    'reason': action_dict.get('reason', '')
                }
            
            elif action in ['close', 'close_and_buy', 'close_and_sell']:
                # 平仓
                if current_trade is not None:
                    exit_price = self._apply_slippage(current_price, 
                                                     'sell' if current_trade['type'] == 'BUY' else 'buy')
                    profit = self._calculate_profit(current_trade, exit_price)
                    
                    # 记录交易
                    trade_record = {
                        **current_trade,
                        'exit_time': row['datetime'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': action_dict.get('reason', '')
                    }
                    self.trades.append(trade_record)
                    
                    # 更新账户
                    self.balance += profit
                    self.total_trades += 1
                    if profit > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    # 关闭持仓
                    if hasattr(self.strategy, 'close_position'):
                        # Check signature
                        import inspect
                        sig = inspect.signature(self.strategy.close_position)
                        if 'pnl' in sig.parameters:
                            self.strategy.close_position(profit)
                        else:
                            self.strategy.close_position()
                    
                    current_trade = None
                    
                    # 如果是反向开仓
                    if action == 'close_and_buy':
                        entry_price = self._apply_slippage(current_price, 'buy')
                        trade_lot_size = action_dict.get('position_size', lot_size)
                        
                        tiers = action_dict.get('tiers', [])
                        if hasattr(self.strategy, 'open_position') and 'tiers' in action_dict:
                             self.strategy.open_position(1, entry_price, action_dict['sl'], tiers)
                        else:
                             self.strategy.open_position(1, entry_price, action_dict['sl'], action_dict.get('tp', 0))

                        current_trade = {
                            'entry_time': row['datetime'],
                            'entry_price': entry_price,
                            'type': 'BUY',
                            'lot_size': trade_lot_size,
                            'sl': action_dict['sl'],
                            'tp': action_dict.get('tp', tiers[-1].take_profit if tiers else 0),
                            'reason': action_dict.get('reason', '')
                        }
                    elif action == 'close_and_sell':
                        entry_price = self._apply_slippage(current_price, 'sell')
                        trade_lot_size = action_dict.get('position_size', lot_size)
                        
                        tiers = action_dict.get('tiers', [])
                        if hasattr(self.strategy, 'open_position') and 'tiers' in action_dict:
                             self.strategy.open_position(-1, entry_price, action_dict['sl'], tiers)
                        else:
                             self.strategy.open_position(-1, entry_price, action_dict['sl'], action_dict.get('tp', 0))
                        
                        current_trade = {
                            'entry_time': row['datetime'],
                            'entry_price': entry_price,
                            'type': 'SELL',
                            'lot_size': trade_lot_size,
                            'sl': action_dict['sl'],
                            'tp': action_dict.get('tp', tiers[-1].take_profit if tiers else 0),
                            'reason': action_dict.get('reason', '')
                        }
            
            # 更新权益曲线
            self.equity = self.balance
            if current_trade is not None:
                unrealized_pnl = self._calculate_profit(current_trade, current_price)
                self.equity += unrealized_pnl
            
            self.equity_curve.append({
                'datetime': row['datetime'],
                'balance': self.balance,
                'equity': self.equity
            })
        
        # 如果回测结束时仍有持仓，强制平仓
        if current_trade is not None:
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            profit = self._calculate_profit(current_trade, exit_price)
            
            trade_record = {
                **current_trade,
                'exit_time': last_row['datetime'],
                'exit_price': exit_price,
                'profit': profit,
                'exit_reason': 'End of backtest'
            }
            self.trades.append(trade_record)
            self.balance += profit
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        
        # 生成回测结果
        results = self._generate_results()
        
        print("\n" + "=" * 60)
        print("回测完成!")
        print(f"总交易次数: {self.total_trades}")
        print(f"最终资金: ${self.balance:,.2f}")
        print(f"总收益: ${self.balance - self.initial_balance:,.2f} ({results['total_return']:.2f}%)")
        print("=" * 60)
        
        return results
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """应用滑点"""
        slippage = self.slippage_points * self.point_value
        if direction == 'buy':
            return price + slippage
        else:
            return price - slippage
    
    def _calculate_profit(self, trade: Dict, exit_price: float) -> float:
        """
        计算交易盈亏
        
        Args:
            trade: 交易记录
            exit_price: 平仓价格
            
        Returns:
            盈亏金额
        """
        entry_price = trade['entry_price']
        lot_size = trade['lot_size']
        
        # 计算点数差
        if trade['type'] == 'BUY':
            price_diff = exit_price - entry_price
        else:  # SELL
            price_diff = entry_price - exit_price
        
        # 计算盈亏
        # Profit = (Price Diff) * Contract Size * Lots
        profit = price_diff * self.contract_size * lot_size
        
        # 减去手续费
        profit -= self.commission * lot_size * 2  # 开仓+平仓
        
        return profit
    
    def _generate_results(self) -> Dict:
        """生成回测结果统计"""
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
        
        return results
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """获取权益曲线DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)
