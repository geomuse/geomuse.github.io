"""
高级黄金策略配置文件
针对黄金和高波动性货币对优化的参数
"""

# ============================================
# 高级策略参数
# ============================================
class AdvancedStrategyConfig:
    # SuperTrend指标
    SUPERTREND_PERIOD = 10
    SUPERTREND_MULTIPLIER = 3.0
    
    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD = 2.0
    BB_SQUEEZE_THRESHOLD = 0.001  # 挤压阈值（相对宽度）
    
    # RSI
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 75  # 黄金波动大，提高阈值
    RSI_OVERSOLD = 25
    
    # Volume动量
    VOLUME_MA_PERIOD = 20
    VOLUME_THRESHOLD = 1.2  # 成交量需超过MA的120%
    
    # ATR
    ATR_PERIOD = 14
    ATR_SL_MULTIPLIER = 2.0  # 止损倍数
    
    # 风险管理
    MAX_RISK_PER_TRADE = 0.015  # 单笔最大风险1.5%
    KELLY_FRACTION = 0.25  # Kelly criterion的25%（保守）
    
    # 分批止盈设置
    TIER_1_SIZE = 0.30  # 第一段30%
    TIER_1_TARGET = 1.5  # 1.5倍ATR
    
    TIER_2_SIZE = 0.40  # 第二段40%
    TIER_2_TARGET = 3.0  # 3.0倍ATR
    
    TIER_3_SIZE = 0.30  # 第三段30%
    TIER_3_TARGET = 5.0  # 5.0倍ATR（移动止损）
    
    # 信号过滤
    MIN_SIGNAL_STRENGTH = 4  # 最低信号强度（5个指标中至少4个确认）
    
    # 交易设置
    SYMBOL = "XAUUSD"
    TIMEFRAME = "5min"  # 5分钟周期

# ============================================
# MT5连接设置（继承原配置）
# ============================================
class MT5Config:
    ACCOUNT = 60018521
    PASSWORD = "Thuctive963,"
    SERVER = "TradeMaxGlobal-Demo"
    MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    MAX_RETRIES = 5
    RETRY_DELAY = 3

# ============================================
# 回测设置
# ============================================
class BacktestConfig:
    INITIAL_BALANCE = 10000  # 10,000 USD
    COMMISSION = 0.00007  # 每手手续费
    SLIPPAGE_POINTS = 2
    DATA_PATH = "data/XAUUSD.csv"
    START_DATE = None
    END_DATE = None

# ============================================
# 输出设置
# ============================================
class GeneralConfig:
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/advanced_trading.log"
    PLOT_RESULTS = True
    SAVE_TRADES = True
    TRADES_FILE = "results/advanced_trades.csv"
    REPORT_FILE = "results/advanced_backtest_report.txt"
