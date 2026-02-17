"""
调试脚本：检查策略信号生成
用于诊断为何没有产生交易
"""

import pandas as pd
from improved_ema_strategy import EMAStrategy
from data_loader import DataLoader
from config import BacktestConfig

def main():
    # Windows UTF-8 fix
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n" + "=" * 60)
    print("Strategy Signal Debug Tool")
    print("=" * 60 + "\n")
    
    # 加载数据
    print("1. 加载数据...")
    loader = DataLoader()
    df = loader.load_csv(BacktestConfig.DATA_PATH)
    print(f"   数据长度: {len(df)} 根K线")
    print(f"   时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}\n")
    
    # 初始化策略
    print("2. 初始化策略...")
    strategy = EMAStrategy(
        fast_period=50,
        slow_period=200,
        adx_period=14,
        adx_threshold=25.0,
        atr_period=14,
        atr_multiplier=2.0,
        trailing_atr_multiplier=3.0
    )
    print(f"   快速EMA: {strategy.fast_period}")
    print(f"   慢速EMA: {strategy.slow_period}")
    print(f"   ADX阈值: {strategy.adx_threshold}\n")
    
    # 生成信号
    print("3. 生成交易信号...")
    df = strategy.generate_signals(df)
    
    # 检查数据完整性
    print("\n4. 检查指标计算...")
    valid_rows = df[df['ema_fast'].notna() & df['ema_slow'].notna() & df['adx'].notna()]
    print(f"   有效数据行: {len(valid_rows)}/{len(df)}")
    
    if len(valid_rows) > 0:
        print(f"   EMA快线范围: {valid_rows['ema_fast'].min():.2f} - {valid_rows['ema_fast'].max():.2f}")
        print(f"   EMA慢线范围: {valid_rows['ema_slow'].min():.2f} - {valid_rows['ema_slow'].max():.2f}")
        print(f"   ADX范围: {valid_rows['adx'].min():.2f} - {valid_rows['adx'].max():.2f}")
        print(f"   ATR范围: {valid_rows['atr'].min():.4f} - {valid_rows['atr'].max():.4f}")
    
    # 检查信号
    print("\n5. 分析交易信号...")
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    print(f"   买入信号数量: {len(buy_signals)}")
    print(f"   卖出信号数量: {len(sell_signals)}")
    print(f"   总信号数量: {len(buy_signals) + len(sell_signals)}")
    
    # 显示最近的信号
    if len(buy_signals) > 0:
        print("\n   最近5个买入信号:")
        recent_buys = buy_signals.tail(5)[['datetime', 'close', 'ema_fast', 'ema_slow', 'adx']]
        print(recent_buys.to_string(index=False))
    
    if len(sell_signals) > 0:
        print("\n   最近5个卖出信号:")
        recent_sells = sell_signals.tail(5)[['datetime', 'close', 'ema_fast', 'ema_slow', 'adx']]
        print(recent_sells.to_string(index=False))
    
    # 检查ADX过滤效果
    print("\n6. ADX过滤分析...")
    adx_valid = valid_rows[valid_rows['adx'] > strategy.adx_threshold]
    print(f"   ADX > {strategy.adx_threshold} 的K线数量: {len(adx_valid)}/{len(valid_rows)}")
    print(f"   占比: {len(adx_valid)/len(valid_rows)*100:.1f}%")
    
    # 检查最近的数据状态
    print("\n7. 最近数据状态（最后10根K线）...")
    recent = df.tail(10)[['datetime', 'close', 'ema_fast', 'ema_slow', 'adx', 'signal']]
    print(recent.to_string(index=False))
    
    # Suggestions
    print("\n" + "=" * 60)
    print("Diagnosis & Suggestions:")
    print("=" * 60)
    
    if len(buy_signals) + len(sell_signals) == 0:
        print("WARNING: No trading signals detected!")
        print("\nPossible reasons:")
        print("1. EMA periods too long (50/200), needs more time for crossover")
        print("2. ADX threshold too high (25), market trend not strong enough")
        print("3. Insufficient data, need at least 200 bars for slow EMA")
        print("\nSuggestions:")
        print("- Try lowering ADX threshold to 20 or 15")
        print("- Or use shorter EMA periods (like 20/50)")
        print("- Ensure enough historical data available")
    elif len(adx_valid) < len(valid_rows) * 0.1:
        print(f"WARNING: ADX filter too strict, only {len(adx_valid)/len(valid_rows)*100:.1f}% of time qualifies")
        print(f"\nSuggest lowering ADX threshold from {strategy.adx_threshold} to 20 or lower")
    else:
        print("OK: Strategy normal, signals generated")
        print(f"   If live trading has no trades, check:")
        print("   1. MT5 connection is working")
        print("   2. Executing on new bar formation")
        print("   3. Review trading.log file")
    
    print()

if __name__ == "__main__":
    main()
