"""
real trading main program
real-time monitoring market and executing EMA crossover strategy
"""

import time
import logging
from datetime import datetime
from typing import Optional

from mt5_connector import MT5Connector
from improved_ema_strategy import EMAStrategy
from config import StrategyConfig, MT5Config, GeneralConfig


class LiveTrader:
    """实盘交易器"""
    
    def __init__(self, 
                 symbol: str = StrategyConfig.SYMBOL,
                 timeframe: str = StrategyConfig.TIMEFRAME,
                 lot_size: float = StrategyConfig.LOT_SIZE):
        """
        初始化实盘交易器
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            lot_size: 每次交易手数
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot_size = lot_size
        
        # 初始化MT5连接器
        self.mt5 = MT5Connector()
        
        # 初始化改进策略（使用优化参数）
        self.strategy = EMAStrategy(
            fast_period=50,
            slow_period=200,
            adx_period=14,
            adx_threshold=25.0,
            atr_period=14,
            atr_multiplier=2.0,
            trailing_atr_multiplier=3.0
        )
        
        # 状态
        self.running = False
        self.current_position_ticket = None
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, GeneralConfig.LOG_LEVEL),
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(GeneralConfig.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """启动交易"""
        self.logger.info("=" * 60)
        self.logger.info("启动EMA实盘交易系统")
        self.logger.info("=" * 60)
        
        # 连接MT5
        if not self.mt5.connect():
            self.logger.error("无法连接MT5，退出")
            return
        
        try:
            # 显示账户信息
            account_info = self.mt5.get_account_info()
            self.logger.info(f"账户余额: ${account_info['balance']:,.2f}")
            self.logger.info(f"净值: ${account_info['equity']:,.2f}")
            self.logger.info(f"杠杆: 1:{account_info['leverage']}")
            
            # 显示策略参数
            self.logger.info(f"交易品种: {self.symbol}")
            self.logger.info(f"时间周期: {self.timeframe}")
            self.logger.info(f"快速EMA: {self.strategy.fast_period}")
            self.logger.info(f"慢速EMA: {self.strategy.slow_period}")
            self.logger.info(f"ADX阈值: {self.strategy.adx_threshold}")
            self.logger.info(f"ATR止损倍数: {self.strategy.atr_multiplier}")
            self.logger.info(f"移动止损倍数: {self.strategy.trailing_atr_multiplier}")
            self.logger.info("=" * 60)
            
            self.running = True
            self._trading_loop()
            
        except KeyboardInterrupt:
            self.logger.info("\n用户中断，正在退出...")
        except Exception as e:
            self.logger.error(f"发生错误: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """停止交易"""
        self.logger.info("停止交易系统...")
        self.running = False
        self.mt5.disconnect()
        self.logger.info("系统已停止")
    
    def _trading_loop(self):
        """交易主循环"""
        last_bar_time = None
        
        while self.running:
            try:
                # 获取最新K线数据
                df = self.mt5.get_bars(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    count=max(self.strategy.slow_period + 10, 50)
                )
                
                # 检查是否有新K线
                current_bar_time = df.iloc[-1]['datetime']
                
                if last_bar_time is None or current_bar_time > last_bar_time:
                    # 新K线产生，执行策略逻辑
                    last_bar_time = current_bar_time
                    self._execute_strategy(df)
                
                # 等待一段时间后再次检查
                time.sleep(1)  # 每5秒检查一次
                
            except Exception as e:
                self.logger.error(f"交易循环错误: {e}", exc_info=True)
                time.sleep(5)
    
    def _execute_strategy(self, df):
        """执行策略逻辑"""
        # 生成信号
        df = self.strategy.generate_signals(df)
        
        # 获取最新信号
        last_row = df.iloc[-1]
        current_price = last_row['close']
        signal = last_row['signal']
        
        # Debug信息
        ema_fast = last_row.get('ema_fast', None)
        ema_slow = last_row.get('ema_slow', None)
        adx = last_row.get('adx', None)
        atr = last_row.get('atr', None)
        
        # 格式化值
        ema_fast_str = f"{ema_fast:.2f}" if ema_fast is not None else "N/A"
        ema_slow_str = f"{ema_slow:.2f}" if ema_slow is not None else "N/A"
        adx_str = f"{adx:.1f}" if adx is not None else "N/A"
        atr_str = f"{atr:.4f}" if atr is not None else "N/A"
        
        self.logger.debug(f"Bar: {last_row['datetime']} | Price: {current_price:.2f} | "
                         f"EMA Fast: {ema_fast_str} | EMA Slow: {ema_slow_str} | "
                         f"ADX: {adx_str} | ATR: {atr_str} | Signal: {signal}")
        
        # 检查是否已有持仓
        positions = self.mt5.get_positions()
        current_positions = [p for p in positions if p['symbol'] == self.symbol]
        
        # 更新策略的持仓状态
        if len(current_positions) > 0:
            pos = current_positions[0]
            self.current_position_ticket = pos['ticket']
            self.strategy.position = 1 if pos['type'] == 'BUY' else -1
            self.strategy.entry_price = pos['price_open']
            self.strategy.stop_loss = pos['sl']
        else:
            self.current_position_ticket = None
            self.strategy.position = 0
        
        # 获取ATR和ADX
        atr = last_row.get('atr', 0.01)
        adx = last_row.get('adx', 25.0)
        
        # 获取交易动作（改进策略需要ATR和ADX）
        action_dict = self.strategy.get_action(current_price, signal, atr, adx)
        action = action_dict['action']
        
        # Log action
        if action != 'hold':
            self.logger.info(f"🎯 Action: {action} | Reason: {action_dict['reason']}")
        
        # 执行交易
        if action == 'buy':
            self._open_buy(current_price, action_dict)
        
        elif action == 'sell':
            self._open_sell(current_price, action_dict)
        
        elif action == 'close':
            self._close_position(action_dict['reason'])
        
        elif action == 'close_and_buy':
            self._close_position('反向开仓')
            time.sleep(1)
            self._open_buy(current_price, action_dict)
        
        elif action == 'close_and_sell':
            self._close_position('反向开仓')
            time.sleep(1)
            self._open_sell(current_price, action_dict)
    
    def _open_buy(self, price: float, action_dict: dict):
        """开多仓"""
        try:
            self.logger.info(f"📈 开多仓信号 - 价格: {price:.5f} - 原因: {action_dict['reason']}")
            
            result = self.mt5.place_order(
                symbol=self.symbol,
                order_type='buy',
                volume=self.lot_size,
                sl=action_dict['sl'],
                tp=action_dict['tp'],
                comment=f"EMA Buy - {action_dict['reason']}"
            )
            
            self.logger.info(f"✓ 开多仓成功 - 订单: {result['order']} - 价格: {result['price']:.5f}")
            
        except Exception as e:
            self.logger.error(f"❌ 开多仓失败: {e}")
    
    def _open_sell(self, price: float, action_dict: dict):
        """开空仓"""
        try:
            self.logger.info(f"📉 开空仓信号 - 价格: {price:.5f} - 原因: {action_dict['reason']}")
            
            result = self.mt5.place_order(
                symbol=self.symbol,
                order_type='sell',
                volume=self.lot_size,
                sl=action_dict['sl'],
                tp=action_dict['tp'],
                comment=f"EMA Sell - {action_dict['reason']}"
            )
            
            self.logger.info(f"✓ 开空仓成功 - 订单: {result['order']} - 价格: {result['price']:.5f}")
            
        except Exception as e:
            self.logger.error(f"❌ 开空仓失败: {e}")
    
    def _close_position(self, reason: str):
        """平仓"""
        if self.current_position_ticket is None:
            return
        
        try:
            self.logger.info(f"🔳 平仓信号 - 原因: {reason}")
            
            result = self.mt5.close_position(self.current_position_ticket)
            
            profit = result['profit']
            profit_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
            
            self.logger.info(f"✓ 平仓成功 - 盈亏: {profit_str}")
            
            self.current_position_ticket = None
            
        except Exception as e:
            self.logger.error(f"❌ 平仓失败: {e}")


def main():
    """主函数"""
    print("\n⚠️  警告: 这是实盘交易系统，请确保已在模拟账户上充分测试！\n")
    
    # 创建交易器
    trader = LiveTrader()
    
    # 启动交易
    trader.start()


if __name__ == "__main__":
    main()
