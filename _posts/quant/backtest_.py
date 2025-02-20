import backtrader as bt

class EMACrossoverStrategy(bt.Strategy):
    params = (
        ('ema_short', 50),
        ('ema_long', 200),
        ('stop_loss', 50),
        ('take_profit', 100),
        ('lot_size', 0.1),
    )

    def __init__(self):
        self.ema_short = bt.indicators.EMA(period=self.params.ema_short)
        self.ema_long = bt.indicators.EMA(period=self.params.ema_long)
        self.order = None

    def next(self):
        if self.order:
            return
        
        if not self.position:  # 如果没有持仓
            if self.ema_short > self.ema_long:
                self.order = self.buy()
            elif self.ema_short < self.ema_long:
                self.order = self.sell()

        else:  # 处理止盈止损
            if self.position.size > 0:  # 持有多头
                if self.data.close[0] >= self.buy_price + self.params.take_profit:
                    self.close()
                elif self.data.close[0] <= self.buy_price - self.params.stop_loss:
                    self.close()

            elif self.position.size < 0:  # 持有空头
                if self.data.close[0] <= self.sell_price - self.params.take_profit:
                    self.close()
                elif self.data.close[0] >= self.sell_price + self.params.stop_loss:
                    self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
            else:
                self.sell_price = order.executed.price
            self.order = None
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

if __name__ == '__main__':

    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(1000)
    cerebro.addstrategy(EMACrossoverStrategy)
    data = bt.feeds.GenericCSVData(
        dataname='XAUUSD.csv',
        # dtformat='%Y-%m-%d %H:%M:%S ',
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    cerebro.adddata(data)
    cerebro.run()
    cerebro.plot()
