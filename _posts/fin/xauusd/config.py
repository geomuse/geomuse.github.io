"""
配置文件 - EMA交叉策略
包含策略参数、MT5连接设置、回测配置
"""

# ============================================
# 策略参数
# ============================================
class StrategyConfig:
    # EMA周期
    FAST_EMA = 12          # 快速EMA周期
    SLOW_EMA = 26          # 慢速EMA周期
    
    # 风险管理
    STOP_LOSS_POINTS = 50      # 止损点数
    TAKE_PROFIT_POINTS = 100   # 止盈点数
    RISK_PER_TRADE = 0.01      # 每次交易风险（账户的1%）
    
    # 交易设置
    SYMBOL = "XAUUSD"          # 交易品种
    TIMEFRAME = "5min"         # 时间周期（5分钟）
    LOT_SIZE = 0.01            # 默认手数（如果不使用风险管理）

# ============================================
# MT5连接设置
# ============================================
class MT5Config:
    # MT5账户信息（实盘交易时填写）
    ACCOUNT = 60018521             # 账户号
    PASSWORD = "Thuctive963,"            # 密码
    SERVER = "TradeMaxGlobal-Demo"              # 服务器地址
    
    # MT5路径（如果需要指定）
    MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"            # 例如: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    
    # 重连设置
    MAX_RETRIES = 5            # 最大重连次数
    RETRY_DELAY = 3            # 重连延迟（秒）

# ============================================
# 回测设置
# ============================================
class BacktestConfig:
    # 初始资金
    INITIAL_BALANCE = 1000   # 1,000 USD
    
    # 交易成本
    COMMISSION = 0.00007       # 每手手续费（7美元/标准手）
    SLIPPAGE_POINTS = 2        # 滑点（点）
    
    # 数据设置
    DATA_PATH = "data/XAUUSD_.csv"  # CSV数据文件路径
    
    # 日期范围（如果需要限制回测时间段）
    START_DATE = None          # 例如: "2023-01-01"
    END_DATE = None            # 例如: "2024-01-01"

# ============================================
# 其他设置
# ============================================
# class GeneralConfig:
#     # 日志设置
#     LOG_LEVEL = "INFO"         # DEBUG, INFO, WARNING, ERROR
#     LOG_FILE = "logs/trading.log"
    
#     # 输出设置
#     PLOT_RESULTS = True        # 是否生成图表
#     SAVE_TRADES = True         # 是否保存交易记录
#     TRADES_FILE = "results/trades.csv"
    
#     # 性能报告
#     REPORT_FILE = "results/backtest_report.txt"
