"""
运行优化后的EMA策略回测
"""

import os
import sys
from datetime import datetime

from data_loader import DataLoader
from improved_ema_strategy import EMAStrategy
from backtest_engine import BacktestEngine
from performance_metrics import PerformanceMetrics
from config import BacktestConfig, GeneralConfig


def main():
    """主函数"""
    # 设置UTF-8编码（Windows兼容）
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("\n" + "=" * 60)
    print("改进版EMA策略回测系统")
    print("=" * 60 + "\n")
    
    # 1. 加载数据
    print("步骤 1/4: 加载历史数据...")
    loader = DataLoader()
    
    try:
        df = loader.load_csv(
            filepath=BacktestConfig.DATA_PATH,
            start_date=BacktestConfig.START_DATE,
            end_date=BacktestConfig.END_DATE
        )
        
        loader.validate_data(df)
        
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}\n")
        return
    
    # 2. 初始化改进策略
    print("\n步骤 2/4: 初始化改进版EMA策略...")
    print(f"  快速EMA: 50")
    print(f"  慢速EMA: 200")
    print(f"  ADX周期: 14 (阈值: 25.0)")
    print(f"  ATR周期: 14 (止损倍数: 2.0)")
    print(f"  移动止损: 3.0倍ATR")
    
    strategy = EMAStrategy(
        fast_period=50,
        slow_period=200,
        adx_period=14,
        adx_threshold=25.0,
        atr_period=14,
        atr_multiplier=2.0,
        trailing_atr_multiplier=3.0
    )
    
    # 3. 运行回测
    print("\n步骤 3/4: 运行回测...")
    
    backtest = BacktestEngine(
        strategy=strategy,
        initial_balance=BacktestConfig.INITIAL_BALANCE,
        commission=BacktestConfig.COMMISSION,
        slippage_points=BacktestConfig.SLIPPAGE_POINTS,
        point_value=0.01  # XAUUSD点值（黄金）
    )
    
    results = backtest.run(df, lot_size=0.01)
    
    # 4. 生成报告和图表
    print("\n步骤 4/4: 生成性能报告...")
    
    metrics = PerformanceMetrics(results)
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 生成文本报告
    report = metrics.generate_report(save_path='results/improved_backtest_report.txt')
    print("\n" + report)
    
    # 保存交易记录
    if results['trades']:
        trades_df = backtest.get_trades_df()
        trades_df.to_csv('results/improved_trades.csv', index=False)
        print(f"✓ 交易记录已保存至: results/improved_trades.csv")
    
    # 生成图表
    print("\n生成图表...")
    metrics.plot_results(
        save_path='results/improved_backtest_chart.png',
        show=False
    )
    
    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
