import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from strategy__ import MartingaleStrategy

class Backtester:
    def __init__(self, strategy, start_balance=100):
        self.strategy = strategy
        self.balance = start_balance
        self.equity = start_balance
        self.orders = [] # List of {'type': 'BUY/SELL', 'price': 1.0, 'lots': 0.01}
        self.history = [] # For plotting
        self.current_price = 1.1000
        self.time_step = 0
        self.price_history = []
        
    def generate_tick(self):
        # Random Walk Simulation
        change = random.uniform(-0.0005, 0.0005) # +/- 5 pips per tick (volatile)
        self.current_price += change
        self.price_history.append(self.current_price)
        
        return {
            'symbol': 'TEST',
            'bid': self.current_price,
            'ask': self.current_price + 0.0002, # 2 pip spread
            'orders': len(self.orders),
            'profit': self.calculate_open_profit(),
            'balance': self.balance,
            'closes': self.price_history,
            'highs': self.price_history, # Simulating High=Close for tick data
            'lows': self.price_history   # Simulating Low=Close for tick data
        }

    def calculate_open_profit(self):
        profit = 0.0
        for order in self.orders:
            diff = 0
            if order['type'] == 'BUY':
                diff = self.current_price - order['price']
            elif order['type'] == 'SELL':
                diff = order['price'] - self.current_price
            
            # XAUUSD Contract Size = 100, Forex = 100000
            contract_size = 100 if 'XAU' in self.orders[0].get('symbol', 'TEST') else 100000
            
            # For this simple script, we default to 100 if we are running XAU test
            if hasattr(self, 'current_symbol') and 'XAU' in self.current_symbol:
                 contract_size = 100
                 
            profit += diff * contract_size * order['lots']
        return profit

    def execute_action(self, action):
        if not action:
            return

        act = action.get('action')
        
        if act == 'BUY':
            self.orders.append({
                'type': 'BUY',
                'price': self.current_price + 0.0002,
                'lots': action.get('lots', 0.01)
            })
            # print(f"Executed BUY at {self.current_price:.5f}")
            
        elif act == 'SELL':
            self.orders.append({
                'type': 'SELL',
                'price': self.current_price,
                'lots': action.get('lots', 0.01)
            })
            # print(f"Executed SELL at {self.current_price:.5f}")
            
        elif act == 'CLOSE_ALL':
            profit = self.calculate_open_profit()
            self.balance += profit
            # Record trade result for metrics
            self.closed_trades.append(profit) 
            print(f"Closed All. Profit: {profit:.2f}. New Balance: {self.balance:.2f}")
            self.orders = []

    def run(self, ticks=1000):
        print(f"Starting Backtest with ${self.balance}")
        self.strategy.reset()
        self.closed_trades = [] # Track individual trade results
        self.max_balance = self.balance
        self.drawdowns = []
        
        for i in range(ticks):
            data = self.generate_tick()
            decision = self.strategy.on_tick(data)
            self.execute_action(decision)
            
            # Record Equity
            open_profit = self.calculate_open_profit()
            self.equity = self.balance + open_profit
            self.history.append(self.equity)
            
            # Max Drawdown Tracking
            if self.equity > self.max_balance:
                self.max_balance = self.equity
            dd = self.max_balance - self.equity
            self.drawdowns.append(dd)
            
            self.time_step += 1
            
        print(f"Backtest Finished. Final Equity: ${self.equity:.2f}")
        self.print_stats()
        
    def print_stats(self):
        import numpy as np
        
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            print("No trades closed.")
            return

        wins = [p for p in self.closed_trades if p > 0]
        losses = [p for p in self.closed_trades if p <= 0]
        
        win_rate = len(wins) / total_trades * 100
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0.0
        max_drawdown_pct = (max_drawdown / self.history[0]) * 100 if self.history else 0.0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        print("\n" + "="*30)
        print("      BACKTEST RESULTS      ")
        print("="*30)
        print(f"Final Balance:    ${self.balance:.2f}")
        print(f"Total Return:     {((self.balance - self.history[0])/self.history[0])*100:.2f}%")
        print(f"Total Trades:     {total_trades}")
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Max Drawdown:     ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
        print(f"Avg Win:          ${avg_win:.2f}")
        print(f"Avg Loss:         ${avg_loss:.2f}")
        print("="*30 + "\n")

    def plot(self):
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(self.history, label='Equity', color='blue')
        plt.title('Equity Curve')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        
        # Subplot 2: Drawdown
        plt.subplot(2, 1, 2)
        plt.plot(self.drawdowns, label='Drawdown', color='red')
        plt.title('Drawdown ($)')
        plt.ylabel('Drawdown Amount')
        plt.fill_between(range(len(self.drawdowns)), self.drawdowns, color='red', alpha=0.3)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def load_csv(self, filepath):
        import csv
        from datetime import datetime
        
        history_data = []
        print(f"Loading data from {filepath}...")
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6: continue
                # Format: 2025.12.18,23:40,4331.17,4331.48,4330.83,4330.93,204
                try:
                    date_str = row[0]
                    time_str = row[1]
                    open_price = float(row[2])
                    high = float(row[3])
                    low = float(row[4])
                    close = float(row[5])
                    
                    history_data.append({
                        'time': f"{date_str} {time_str}",
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close
                    })
                except ValueError:
                    continue # Skip header or bad lines
                    
        print(f"Loaded {len(history_data)} bars.")
        
        # Run Backtest with loaded data
        self.run_with_data(history_data)

    def run_with_data(self, history_data):
        print("Starting Backtest on Historical Data...")
        self.strategy.reset()
        self.closed_trades = []
        self.max_balance = self.balance
        self.drawdowns = []
        self.history = []
        self.orders = []
        self.time_step = 0
        
        # Pre-calculate lists for efficiency
        opens = [d['open'] for d in history_data]
        highs = [d['high'] for d in history_data]
        lows = [d['low'] for d in history_data]
        closes = [d['close'] for d in history_data]
        
        print(f"Total Bars: {len(history_data)}")
        
        for i in range(len(history_data)):
            # Construct tick data from bar
            # We use 'Close' as the current market price for this simple simulation
            current_close = closes[i]
            self.current_price = current_close
            self.current_symbol = 'XAUUSD'
            
            tick = {
                'symbol': 'XAUUSD',
                'bid': current_close,
                'ask': current_close + 0.30, # 30 cents spread
                'orders': len(self.orders),
                'profit': self.calculate_open_profit(),
                'balance': self.balance,
                'closes': closes[:i+1],
                'highs': highs[:i+1],
                'lows': lows[:i+1],
                'bar_index': i
            }
            
            decision = self.strategy.on_tick(tick)
            self.execute_action(decision)
            
            # Record Equity
            open_profit = self.calculate_open_profit()
            self.equity = self.balance + open_profit
            self.history.append(self.equity)
            
            # Max Drawdown Tracking
            if self.equity > self.max_balance:
                self.max_balance = self.equity
            dd = self.max_balance - self.equity
            self.drawdowns.append(dd)
            
        print(f"Backtest Finished. Final Equity: ${self.equity:.2f}")
        self.print_stats()
        self.plot()

if __name__ == "__main__":
    import sys
    strategy = MartingaleStrategy()
    backtester = Backtester(strategy)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--zmq':
            backtester.start_zmq_server()
        elif sys.argv[1] == '--csv':
            csv_path = sys.argv[2] if len(sys.argv) > 2 else "XAUUSD5.csv"
            backtester.load_csv(csv_path)
    else:
        # Default Random Walk
        backtester.run(ticks=5000)
        backtester.plot()
