# MF5 实盘交易部署指南 (MFAG 策略)

本指南将指导您如何将 **Multi-Filter Adaptive Gold Strategy (MFAG)** 部署到 MetaTrader 5 (MT5) 进行实盘或模拟盘交易。

## 1. 环境准备

### 软件要求
- **操作系统**: Windows (MT5 官方库仅支持 Windows)
- **Python**: 3.8+
- **依赖库**: 
  ```bash
  pip install MetaTrader5 pandas numpy matplotlib
  ```

### MT5 终端设置
1. 打开 MT5 终端。
2. 导航至 `工具` -> `选项` -> `智能交易系统` (Expert Advisors)。
3. 勾选 **"允许算法交易"** (Allow Algorithmic trading)。
4. (可选) 勾选 "允许 DLL 导入"。

## 2. 账号配置

打开项目根目录下的 [config.py](file:///home/geo/Downloads/geo/_posts/fin/xauusd/config.py)，在 `MT5Config` 类中填写您的交易账户信息：

```python
class MT5Config:
    ACCOUNT = 12345678          # 您的账户号
    PASSWORD = "您的密码"        # 您的账户密码
    SERVER = "您的服务器地址"      # 例如: "TradeMaxGlobal-Demo"
```

## 3. 启动策略

在 Windows 终端（PowerShell 或 CMD）中运行以下命令：

```bash
python run_advanced_backtest.py
```

在程序启动后，系统会询问运行模式：
1. 输入 `2` 并按回车进入 **实盘交易模式**。
2. 程序会尝试连接 MT5 终端并开始实时市场监控。

## 4. 注意事项 (重要)

> [!WARNING]
> **风险提示**: 实盘交易存在风险。在投入真实资金前，强烈建议在 **模拟账户 (Demo Account)** 上运行至少 1-2 周，以观察策略在实时环境下的表现。

- **断网处理**: 如果互联网连接断开，脚本会自动尝试重连（取决于 `mt5_connector` 的实现）。
- **同步状态**: 脚本启动时会自动检测该品种已有的持仓，并接管止损管理。
- **日志监控**: 交易日志会保存在 `logs/advanced_trading.log` 中。

## 5. 如何停止

在运行脚本的窗口按下 `Ctrl + C`，脚本会安全断开与 MT5 的连接并退出。
