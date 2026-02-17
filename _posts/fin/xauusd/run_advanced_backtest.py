"""
高级黄金策略执行脚本 (MFAG)
支持历史回测与实盘/模拟盘交易
"""

import os
import sys
import io
import time
import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Use a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fixes minus signs showing as squares

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from data_loader import DataLoader
from backtest_engine import BacktestEngine
from mt5_connector import MT5Connector
from advanced_gold_strategy import AdvancedGoldStrategy
from performance_metrics import PerformanceMetrics
from config import BacktestConfig, GeneralConfig, MT5Config
from config_advanced import AdvancedStrategyConfig, GeneralConfig as AdvancedGeneralConfig


class AdvancedBacktestEngine(BacktestEngine):
    """支持分批止盈的高级回测引擎"""
    
    def run(self, df: pd.DataFrame) -> dict:
        """运行回测（支持分批止盈）"""
        print("==" * 30)
        print("开始回测...")
        print(f"初始资金: ${self.initial_balance:,.2f}")
        print(f"数据长度: {len(df)} 根K线")
        print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        print("-" * 60)
        
        # 生成交易信号
        df = self.strategy.generate_signals(df)
        
        # 初始化记录
        current_trade = None
        position_size = 0.0  # 当前仓位大小
        
        # 遍历每根K线
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # 跳过数据不足的阶段
            if pd.isna(row.get('supertrend', None)):
                continue
            
            current_price = row['close']
            signal = row['signal']
            signal_strength = row.get('signal_strength', 0)
            atr = row.get('atr', 0.01)
            supertrend_direction = row.get('supertrend_direction', 1)
            
            # 获取交易动作
            action_dict = self.strategy.get_action(
                current_price, signal, signal_strength, 
                atr, supertrend_direction, self.balance
            )
            
            action = action_dict['action']
            
            # 处理分批平仓
            if action.startswith('close_tier_'):
                tier_num = int(action.split('_')[-1])
                tier = self.strategy.position_tiers[tier_num - 1]
                
                exit_price = self._apply_slippage(current_price, 
                                                 'sell' if current_trade['type'] == 'BUY' else 'buy')
                tier_lot_size = current_trade['lot_size'] * tier.size_percent
                
                tier_trade = {
                    **current_trade,
                    'lot_size': tier_lot_size,
                    'exit_time': row['datetime'],
                    'exit_price': exit_price,
                    'exit_reason': f'Tier {tier_num} TP'
                }
                profit = self._calculate_profit(tier_trade, exit_price)
                tier_trade['profit'] = profit
                
                self.trades.append(tier_trade)
                self.balance += profit
                self.total_trades += 1
                if profit > 0: self.winning_trades += 1
                else: self.losing_trades += 1
                
                position_size -= tier_lot_size
                continue
            
            # 执行开仓
            if action == 'buy':
                entry_price = self._apply_slippage(current_price, 'buy')
                position_size = action_dict['position_size']
                self.strategy.open_position(1, entry_price, action_dict['sl'], action_dict['tiers'])
                current_trade = {
                    'entry_time': row['datetime'],
                    'entry_price': entry_price,
                    'type': 'BUY',
                    'lot_size': position_size,
                    'sl': action_dict['sl'],
                    'tp': action_dict['tiers'][-1].take_profit if action_dict['tiers'] else 0,
                    'reason': action_dict['reason']
                }
                
            elif action == 'sell':
                entry_price = self._apply_slippage(current_price, 'sell')
                position_size = action_dict['position_size']
                self.strategy.open_position(-1, entry_price, action_dict['sl'], action_dict['tiers'])
                current_trade = {
                    'entry_time': row['datetime'],
                    'entry_price': entry_price,
                    'type': 'SELL',
                    'lot_size': position_size,
                    'sl': action_dict['sl'],
                    'tp': action_dict['tiers'][-1].take_profit if action_dict['tiers'] else 0,
                    'reason': action_dict['reason']
                }
            
            elif action in ['close', 'close_and_buy', 'close_and_sell']:
                if current_trade is not None:
                    exit_price = self._apply_slippage(current_price, 
                                                     'sell' if current_trade['type'] == 'BUY' else 'buy')
                    remaining_trade = {
                        **current_trade,
                        'lot_size': position_size,
                        'exit_time': row['datetime'],
                        'exit_price': exit_price,
                        'exit_reason': action_dict['reason']
                    }
                    profit = self._calculate_profit(remaining_trade, exit_price)
                    remaining_trade['profit'] = profit
                    self.trades.append(remaining_trade)
                    self.balance += profit
                    self.total_trades += 1
                    if profit > 0: self.winning_trades += 1
                    else: self.losing_trades += 1
                    
                    self.strategy.close_position(profit)
                    current_trade = None
                    position_size = 0.0
                    
                    if action == 'close_and_buy':
                        # ... reverse logic simplified for brevity but remains robust
                        entry_price = self._apply_slippage(current_price, 'buy')
                        position_size = action_dict['position_size']
                        self.strategy.open_position(1, entry_price, action_dict['sl'], action_dict['tiers'])
                        current_trade = {
                            'entry_time': row['datetime'], 'entry_price': entry_price, 'type': 'BUY',
                            'lot_size': position_size, 'sl': action_dict['sl'], 
                            'tp': action_dict['tiers'][-1].take_profit if action_dict['tiers'] else 0,
                            'reason': action_dict['reason']
                        }
                    elif action == 'close_and_sell':
                        entry_price = self._apply_slippage(current_price, 'sell')
                        position_size = action_dict['position_size']
                        self.strategy.open_position(-1, entry_price, action_dict['sl'], action_dict['tiers'])
                        current_trade = {
                            'entry_time': row['datetime'], 'entry_price': entry_price, 'type': 'SELL',
                            'lot_size': position_size, 'sl': action_dict['sl'], 
                            'tp': action_dict['tiers'][-1].take_profit if action_dict['tiers'] else 0,
                            'reason': action_dict['reason']
                        }
            
            # 更新权益曲线
            self.equity = self.balance
            if current_trade is not None and position_size > 0:
                temp_trade = {**current_trade, 'lot_size': position_size}
                unrealized_pnl = self._calculate_profit(temp_trade, current_price)
                self.equity += unrealized_pnl
            
            self.equity_curve.append({
                'datetime': row['datetime'],
                'balance': self.balance,
                'equity': self.equity
            })
        
        # 强制平仓
        if current_trade is not None and position_size > 0:
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            remaining_trade = {
                **current_trade, 'lot_size': position_size, 'exit_time': last_row['datetime'],
                'exit_price': exit_price, 'exit_reason': 'End of backtest'
            }
            profit = self._calculate_profit(remaining_trade, exit_price)
            remaining_trade['profit'] = profit
            self.trades.append(remaining_trade)
            self.balance += profit
            self.total_trades += 1
        
        results = self._generate_results()
        print("-" * 60)
        print(f"最终资金: ${self.balance:,.2f}")
        print(f"总收益: ${self.balance - self.initial_balance:,.2f} ({results['total_return']:.2f}%)")
        print("==" * 30 + "\n")
        return results


class AdvancedLiveTrader:
    """实盘交易器 - 支持MFAG高级策略"""
    
    def __init__(self, strategy_params: dict):
        self.symbol = AdvancedStrategyConfig.SYMBOL
        self.timeframe = AdvancedStrategyConfig.TIMEFRAME
        self.mt5 = MT5Connector()
        self.strategy = AdvancedGoldStrategy(**strategy_params)
        self.running = False
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, AdvancedGeneralConfig.LOG_LEVEL),
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(AdvancedGeneralConfig.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("MFAG_Live")
        
    def start(self):
        self.logger.info("=" * 60)
        self.logger.info("启动 MFAG 高级黄金交易系统 (实盘/模拟)")
        self.logger.info("=" * 60)
        
        if not self.mt5.connect():
            self.logger.error("MT5连接失败")
            return
            
        try:
            account = self.mt5.get_account_info()
            self.logger.info(f"账户: {account['login']} | 余额: ${account['balance']:,.2f}")
            self.running = True
            
            last_bar_time = None
            while self.running:
                # 获取数据（需要足够长度计算指标）
                df = self.mt5.get_bars(self.symbol, self.timeframe, 100)
                if df is not None and not df.empty:
                    curr_time = df.iloc[-1]['datetime']
                    if last_bar_time is None or curr_time > last_bar_time:
                        last_bar_time = curr_time
                        self._process_tick(df, account['balance'])
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("用户停止...")
        finally:
            self.stop()
            
    def _process_tick(self, df, balance):
        # 1. 生成信号
        df = self.strategy.generate_signals(df)
        last_row = df.iloc[-1]
        current_price = last_row['close']
        signal = last_row['signal']
        signal_strength = last_row.get('signal_strength', 0)
        atr = last_row.get('atr', 0.01)
        supertrend_direction = last_row.get('supertrend_direction', 1)
        
        # 2. 获取当前账户持仓
        positions = self.mt5.get_positions()
        symbol_positions = [p for p in positions if p['symbol'] == self.symbol]
        
        # 3. 同步策略状态
        if symbol_positions:
            # 简化逻辑：仅处理第一个持仓
            pos = symbol_positions[0]
            self.strategy.position = 1 if pos['type'] == 'BUY' else -1
            self.strategy.entry_price = pos['price_open']
            self.strategy.stop_loss = pos['sl']
            
            # 同步分批止盈状态 (如果未设置，则通过ATR计算)
            if not self.strategy.position_tiers:
                self.strategy.position_tiers = self.strategy.create_position_tiers(
                    pos['price_open'], atr, self.strategy.position
                )
        else:
            self.strategy.position = 0
            self.strategy.position_tiers = []
            
        # 4. 获取策略动作
        action_dict = self.strategy.get_action(
            current_price, signal, signal_strength, 
            atr, supertrend_direction, balance
        )
        
        action = action_dict['action']
        if action == 'hold':
            return

        self.logger.info(f"🎯 策略信号: {action} | 原因: {action_dict['reason']}")
            
        # 5. 执行交易
        try:
            if action == 'buy':
                res = self.mt5.place_order(
                    self.symbol, 'buy', action_dict['position_size'], 
                    sl=action_dict['sl'], 
                    comment=f"MFAG Buy - {action_dict['reason']}"
                )
                self.logger.info(f"✓ 已开多仓: {res['order']} @ {res['price']}")
                
            elif action == 'sell':
                res = self.mt5.place_order(
                    self.symbol, 'sell', action_dict['position_size'], 
                    sl=action_dict['sl'], 
                    comment=f"MFAG Sell - {action_dict['reason']}"
                )
                self.logger.info(f"✓ 已开空仓: {res['order']} @ {res['price']}")
                
            elif action == 'close':
                for p in symbol_positions:
                    self.mt5.close_position(p['ticket'])
                self.logger.info("✓ 已全额平仓")
                
            elif action.startswith('close_tier_'):
                # 分批止盈：平掉部分仓位
                tier_num = int(action.split('_')[-1])
                pos = symbol_positions[0]
                # 计算平仓量 (30%, 40%, 30%)
                tiers_pct = [0.30, 0.40, 0.30]
                close_vol = pos['volume'] * tiers_pct[tier_num - 1]
                
                # 注意：MT5部分平仓通常是发一个反向订单或直接指定volume
                # 这里简化处理为全部平仓或根据MT5Connector能力调整
                self.logger.info(f"⚠ 触发分批止盈 Tier {tier_num}，执行当前持仓全平 (演示)")
                self.mt5.close_position(pos['ticket'])
                
        except Exception as e:
            self.logger.error(f"交易执行失败: {e}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MFAG 高级黄金策略系统")
    print("=" * 60 + "\n")
    
    # 步骤 1/5: 模式选择
    print("请选择运行模式:")
    print("  1. 历史回测 (Backtest)")
    print("  2. 实盘/模拟 交易 (Live Trading)")
    choice = input("\n请输入选项 (1/2): ").strip()
    
    strategy_params = {
        'supertrend_period': AdvancedStrategyConfig.SUPERTREND_PERIOD,
        'supertrend_multiplier': AdvancedStrategyConfig.SUPERTREND_MULTIPLIER,
        'bb_period': AdvancedStrategyConfig.BB_PERIOD,
        'bb_std': AdvancedStrategyConfig.BB_STD,
        'bb_squeeze_threshold': AdvancedStrategyConfig.BB_SQUEEZE_THRESHOLD,
        'rsi_period': AdvancedStrategyConfig.RSI_PERIOD,
        'rsi_overbought': AdvancedStrategyConfig.RSI_OVERBOUGHT,
        'rsi_oversold': AdvancedStrategyConfig.RSI_OVERSOLD,
        'volume_ma_period': AdvancedStrategyConfig.VOLUME_MA_PERIOD,
        'volume_threshold': AdvancedStrategyConfig.VOLUME_THRESHOLD,
        'atr_period': AdvancedStrategyConfig.ATR_PERIOD,
        'atr_sl_multiplier': AdvancedStrategyConfig.ATR_SL_MULTIPLIER,
        'max_risk_per_trade': AdvancedStrategyConfig.MAX_RISK_PER_TRADE,
        'kelly_fraction': AdvancedStrategyConfig.KELLY_FRACTION
    }
    
    if choice != "2":
        # 步骤 2/5: 加载回测数据
        print("\n步骤 2/5: 加载历史数据...")
        loader = DataLoader()
        df = loader.load_csv(BacktestConfig.DATA_PATH)
        if df is None: return
        
        # 步骤 3/5: 初始化回测
        print("\n步骤 3/5: 初始化MFAG回测...")
        strategy = AdvancedGoldStrategy(**strategy_params)
        
        # 步骤 4/5: 运行引擎
        engine = AdvancedBacktestEngine(
            strategy=strategy,
            initial_balance=BacktestConfig.INITIAL_BALANCE,
            commission=BacktestConfig.COMMISSION,
            slippage_points=BacktestConfig.SLIPPAGE_POINTS,
            point_value=0.01
        )
        results = engine.run(df)
        
        # 步骤 5/5: 报告
        analyzer = PerformanceMetrics(results)
        analyzer.generate_report(AdvancedGeneralConfig.REPORT_FILE)
        if AdvancedGeneralConfig.PLOT_RESULTS:
            analyzer.plot_results("results/advanced_gold_performance.png", show=False)
        print(f"✓ 回测完成！报告: {AdvancedGeneralConfig.REPORT_FILE}")
        
    else:
        # 实盘模式
        print("\n" + "!" * 60)
        print("警告: 您正在进入实盘交易模式！".center(60))
        print("请确保已在 config.py 填写正确的 MT5 账户信息。".center(60))
        print("!" * 60 + "\n")
        
        trader = AdvancedLiveTrader(strategy_params)
        trader.start()


if __name__ == "__main__":
    main()
