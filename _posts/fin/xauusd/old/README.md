# MT5 EMA交叉策略系统

完整的EMA交叉策略交易系统，支持回测和MT5实盘交易。

## 功能特性

- ✅ **EMA交叉策略**: 基于快速和慢速EMA的金叉/死叉信号
- ✅ **完整风险管理**: 止损、止盈、自动计算手数
- ✅ **回测引擎**: 使用历史数据验证策略性能
- ✅ **性能分析**: 夏普比率、最大回撤、胜率等指标
- ✅ **可视化图表**: 权益曲线、回撤曲线、交易分析
- ✅ **MT5实盘交易**: 实时监控市场并自动执行交易
- ✅ **日志记录**: 完整的交易日志和错误追踪

## 目录结构

```
quant/
├── config.py                  # 配置文件
├── ema_strategy.py           # EMA策略核心
├── data_loader.py            # 数据加载器
├── backtest_engine.py        # 回测引擎
├── performance_metrics.py    # 性能指标
├── run_backtest.py           # 回测执行脚本
├── mt5_connector.py          # MT5连接器
├── live_trader.py            # 实盘交易主程序
├── requirements.txt          # Python依赖
├── data/                     # 数据目录
│   └── EURUSD_5min.csv      # 历史数据（需自行准备）
├── results/                  # 回测结果
│   ├── backtest_report.txt
│   ├── backtest_chart.png
│   └── trades.csv
└── logs/                     # 交易日志
    └── trading.log
```

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 准备历史数据

将CSV格式的历史数据放在 `data/EURUSD_5min.csv`

**CSV格式要求**:
- 必须包含列: `datetime, open, high, low, close`
- datetime格式: `YYYY-MM-DD HH:MM:SS`
- 示例:

```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,1.0900,1.0905,1.0895,1.0900,1000
2024-01-01 00:05:00,1.0900,1.0910,1.0898,1.0905,1200
```

### 3. 配置策略参数

编辑 `config.py` 文件，调整以下参数：

```python
# 策略参数
FAST_EMA = 12              # 快速EMA周期
SLOW_EMA = 26              # 慢速EMA周期
STOP_LOSS_POINTS = 50      # 止损点数
TAKE_PROFIT_POINTS = 100   # 止盈点数

# 交易设置
SYMBOL = "EURUSD"          # 交易品种
TIMEFRAME = "5min"         # 时间周期
LOT_SIZE = 0.01            # 默认手数

# 回测设置
INITIAL_BALANCE = 10000    # 初始资金
DATA_PATH = "data/EURUSD_5min.csv"
```

## 使用方法

### 回测模式

运行回测并查看策略表现：

```bash
python run_backtest.py
```

回测结果将保存在 `results/` 目录：
- `backtest_report.txt`: 文本报告
- `backtest_chart.png`: 可视化图表
- `trades.csv`: 交易记录明细

### 实盘交易模式

**⚠️ 警告**: 实盘交易涉及真实资金风险，请先在模拟账户上充分测试！

#### 1. 配置MT5账户信息

编辑 `config.py` 中的MT5设置：

```python
class MT5Config:
    ACCOUNT = 12345678          # MT5账户号
    PASSWORD = "your_password"   # 密码
    SERVER = "MetaQuotes-Demo"   # 服务器
```

#### 2. 启动实盘交易

```bash
python live_trader.py
```

程序将：
1. 连接到MT5终端
2. 实时监控市场数据
3. 根据EMA交叉信号自动开平仓
4. 记录所有交易到日志文件

#### 3. 停止交易

按 `Ctrl+C` 安全停止交易系统。

## 策略说明

### EMA交叉逻辑

- **金叉（买入信号）**: 快速EMA从下方上穿慢速EMA
- **死叉（卖出信号）**: 快速EMA从上方下穿慢速EMA

### 风险管理

- 每笔交易设置固定止损点数
- 每笔交易设置固定止盈点数
- 当触及止损或止盈时自动平仓
- 反向信号出现时先平仓再反向开仓

## 性能指标说明

回测报告包含以下关键指标：

- **Total Return**: 总收益率
- **Win Rate**: 胜率（盈利交易占比）
- **Max Drawdown**: 最大回撤
- **Sharpe Ratio**: 夏普比率（风险调整后收益）
- **Sortino Ratio**: 索提诺比率（仅考虑下行风险）
- **Profit Factor**: 盈亏比（平均盈利/平均亏损）

## 常见问题

### Q: 回测时提示找不到数据文件？

A: 确保CSV文件路径正确，默认为 `data/EURUSD_5min.csv`。可在 `config.py` 中修改 `DATA_PATH`。

### Q: MT5无法连接？

A: 请确保：
1. MT5终端已安装并运行
2. 账户信息在 `config.py` 中配置正确
3. 可以手动在MT5中登录该账户

### Q: 如何获取历史数据？

A: 有两种方式：
1. 使用 `data_loader.py` 中的 `load_from_mt5()` 从MT5下载
2. 从其他数据源下载CSV格式的历史数据

示例代码（从MT5下载）：

```python
from data_loader import DataLoader

loader = DataLoader()
df = loader.load_from_mt5(
    symbol="EURUSD",
    timeframe="5min",
    start_date="2023-01-01",
    end_date="2024-01-01",
    save_path="data/EURUSD_5min.csv"
)
```

### Q: 如何调整策略参数？

A: 编辑 `config.py` 文件中的 `StrategyConfig` 类即可。修改后重新运行回测或实盘交易。

## 风险提示

⚠️ **重要提示**:

1. 本系统仅供学习和研究使用
2. 实盘交易前请务必在模拟账户上充分测试
3. 过去的回测表现不代表未来收益
4. 任何交易都存在风险，请谨慎使用
5. 建议从小资金开始，逐步验证策略稳定性

## 技术支持

如有问题或建议，请检查：

1. 日志文件: `logs/trading.log`
2. 确保所有依赖正确安装
3. 确认MT5终端版本兼容

## 许可证

MIT License
