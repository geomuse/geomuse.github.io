"""
回测执行脚本
加载数据、运行回测、生成报告
"""

import os
import sys
from datetime import datetime

from data_loader import DataLoader
from ema_strategy import EMAStrategy
from backtest_engine import BacktestEngine
from performance_metrics import PerformanceMetrics
from config import StrategyConfig, BacktestConfig, GeneralConfig


def main():
    """主函数"""
    # 设置UTF-8编码（Windows兼容）
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("\n" + "=" * 60)
    print("EMA交叉策略回测系统")
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
        
        # 验证数据
        loader.validate_data(df)
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 数据文件不存在: {BacktestConfig.DATA_PATH}")
        print(f"\n请将CSV数据文件放置在: {os.path.abspath(BacktestConfig.DATA_PATH)}")
        print(f"或修改config.py中的DATA_PATH配置")
        print("\nCSV文件格式要求:")
        print("  - 必须包含列: datetime, open, high, low, close")
        print("  - datetime格式: YYYY-MM-DD HH:MM:SS")
        print("  - 示例: 2024-01-01 09:00:00,1.0900,1.0905,1.0895,1.0900,1000\n")
        return
    
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}\n")
        return
    
    # 2. 初始化策略
    print("\n步骤 2/4: 初始化EMA策略...")
    print(f"  快速EMA: {StrategyConfig.FAST_EMA}")
    print(f"  慢速EMA: {StrategyConfig.SLOW_EMA}")
    print(f"  止损: {StrategyConfig.STOP_LOSS_POINTS} 点")
    print(f"  止盈: {StrategyConfig.TAKE_PROFIT_POINTS} 点")
    
    strategy = EMAStrategy(
        fast_period=StrategyConfig.FAST_EMA,
        slow_period=StrategyConfig.SLOW_EMA,
        stop_loss_points=StrategyConfig.STOP_LOSS_POINTS,
        take_profit_points=StrategyConfig.TAKE_PROFIT_POINTS
    )
    
    # 3. 运行回测
    print("\n步骤 3/4: 运行回测...")
    
    backtest = BacktestEngine(
        strategy=strategy,
        initial_balance=BacktestConfig.INITIAL_BALANCE,
        commission=BacktestConfig.COMMISSION,
        slippage_points=BacktestConfig.SLIPPAGE_POINTS,
        point_value=0.0001  # EURUSD点值
    )
    
    results = backtest.run(df, lot_size=StrategyConfig.LOT_SIZE)
    
    # 4. 生成报告和图表
    print("\n步骤 4/4: 生成性能报告...")
    
    metrics = PerformanceMetrics(results)
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 生成文本报告
    report = metrics.generate_report(save_path=GeneralConfig.REPORT_FILE)
    print("\n" + report)
    
    # 保存交易记录
    if GeneralConfig.SAVE_TRADES and results['trades']:
        trades_df = backtest.get_trades_df()
        trades_df.to_csv(GeneralConfig.TRADES_FILE, index=False)
        print(f"✓ 交易记录已保存至: {GeneralConfig.TRADES_FILE}")
    
    # 生成图表
    if GeneralConfig.PLOT_RESULTS:
        print("\n生成图表...")
        metrics.plot_results(
            save_path='results/backtest_chart.png',
            show=False  # 设置为True会显示图表窗口
        )
    
    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
