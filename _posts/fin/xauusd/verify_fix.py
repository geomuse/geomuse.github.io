
import pandas as pd
from advanced_gold_strategy import AdvancedGoldStrategy
from backtest_engine import BacktestEngine
from config import BacktestConfig

def verify():
    print("Verifying Advanced Gold Strategy Fixes...")
    
    # 1. Test Strategy Initialization
    strategy = AdvancedGoldStrategy(contract_size=100)
    print(f"Strategy Contract Size: {strategy.contract_size} (Expected: 100)")
    
    # 2. Test Lot Size Calculation
    balance = 10000
    entry_price = 2000.0
    sl_price = 1990.0 # 10 dollar risk
    
    # Risk = 1.5% of 10000 = 150
    # Stop Distance = 10
    # Contract Size = 100
    # Value per lot move = 10 * 100 = 1000
    # Expected Lots = 150 / 1000 = 0.15
    
    lot_size = strategy.calculate_lot_size(balance, entry_price, sl_price)
    print(f"Calculated Lot Size: {lot_size} (Expected: 0.15)")
    
    # 3. Test Backtest Engine Profit Calculation
    engine = BacktestEngine(strategy, contract_size=100)
    print(f"Engine Contract Size: {engine.contract_size} (Expected: 100)")
    
    # Simulate a trade
    trade = {
        'entry_price': 2000.0,
        'lot_size': 0.15,
        'type': 'BUY'
    }
    exit_price = 2010.0 # 10 dollar profit
    
    # Expected Profit:
    # Price Diff = 10
    # Profit = 10 * 100 * 0.15 = 150
    # Commission = 0.15 * 2 * Commission_Rate (0.00007? No, wait)
    # Commission in config is 0.00007 per unit? or per lot?
    # BacktestEngine: profit -= self.commission * lot_size * 2
    # If commission is 0.00007 (from config), it's negligible.
    
    profit = engine._calculate_profit(trade, exit_price)
    print(f"Calculated Profit: {profit} (Expected: ~150)")
    
    if abs(lot_size - 0.15) < 0.001 and abs(profit - 150) < 1:
        print("\nSUCCESS: Fixes verified!")
    else:
        print("\nFAILURE: Calculations do not match expectations.")

if __name__ == "__main__":
    verify()
