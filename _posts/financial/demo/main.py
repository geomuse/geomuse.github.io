import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set(style='whitegrid')

# 模拟股票代码列表
np.random.seed(42)
stock_list = ['Stock_' + str(i) for i in range(1, 101)]  # 100只股票

# 模拟日期范围
dates = pd.date_range(start='2020-01-01', periods=24, freq='M')  # 24个月

# 创建空的DataFrame来存储数据
price_data = pd.DataFrame(index=dates, columns=stock_list)
fundamental_data = pd.DataFrame(index=dates, columns=stock_list)

# 模拟价格数据和基本面数据
for stock in stock_list:
    price_data[stock] = np.random.lognormal(mean=0.001, sigma=0.05, size=len(dates)).cumprod()
    fundamental_data[stock] = np.random.rand(len(dates))

# 计算月度收益率
returns = price_data.pct_change().dropna()

# 初始化因子DataFrame
factors = pd.DataFrame(index=returns.index, columns=stock_list)

# 动量因子（过去12个月的累计收益率）
momentum = price_data.pct_change(12).shift(1).loc[returns.index]
factors_momentum = momentum.copy()

# 市盈率因子（用基本面数据的倒数模拟）
pe_ratio = 1 / fundamental_data.loc[returns.index]
factors_pe = pe_ratio.copy()

# 市净率因子（用基本面数据模拟）
pb_ratio = fundamental_data.loc[returns.index]
factors_pb = pb_ratio.copy()

# 规模因子（模拟市值，这里用价格的对数模拟）
size = np.log(price_data.loc[returns.index])
factors_size = size.copy()

# 因子列表
factor_dict = {
    'Momentum': factors_momentum,
    'PE': factors_pe,
    'PB': factors_pb,
    'Size': factors_size
}

# 因子过滤：计算每个因子的缺失值比例，过滤掉缺失值超过阈值的因子
filtered_factors = {}
threshold = 0.05  # 缺失值阈值
for name, factor in factor_dict.items():
    missing_ratio = factor.isnull().sum().sum() / (factor.shape[0] * factor.shape[1])
    if missing_ratio < threshold:
        filtered_factors[name] = factor.fillna(factor.mean())
        print(f'因子 {name} 被保留，缺失值比例为 {missing_ratio:.2%}')
    else:
        print(f'因子 {name} 被过滤，缺失值比例为 {missing_ratio:.2%}')

# 计算每个因子的IC（Spearman Rank Correlation）
ic_table = pd.DataFrame(index=filtered_factors.keys(), columns=['IC Mean', 'IC Std', 'IC IR'])

for name, factor in filtered_factors.items():
    ic_list = []
    for date in returns.index:
        factor_values = factor.loc[date]
        return_values = returns.loc[date]
        if factor_values.isnull().all() or return_values.isnull().all():
            continue
        # 计算Spearman相关系数
        ic, _ = spearmanr(factor_values, return_values)
        ic_list.append(ic)
    ic_series = pd.Series(ic_list)
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    ic_table.loc[name] = [ic_mean, ic_std, ic_ir]

print('因子IC统计表：')
print(ic_table)

# 设置IC均值和IR的阈值
ic_mean_threshold = 0.01
ic_ir_threshold = 0.05

selected_factors = ic_table[
    (ic_table['IC Mean'].abs() > ic_mean_threshold) &
    (ic_table['IC IR'].abs() > ic_ir_threshold)
]

print('选取的有效因子：')
print(selected_factors)

# 获取选取的因子名称
selected_factor_names = selected_factors.index.tolist()

# 准备回归数据
Y = returns.stack()
X_list = []

for name in selected_factor_names:
    factor = filtered_factors[name]
    X_list.append(factor.stack())

X = pd.concat(X_list, axis=1)
X.columns = selected_factor_names

# 对齐因变量和自变量
reg_data = pd.concat([Y, X], axis=1).dropna()
reg_data.columns = ['Return'] + selected_factor_names

# 添加截距项
reg_data = sm.add_constant(reg_data)

# 回归分析
model = sm.OLS(reg_data['Return'], reg_data[['const'] + selected_factor_names]).fit()
print(model.summary())

# 对每个日期，选取因子得分最高的前20只股票构建组合
portfolio_returns = []

for date in returns.index:
    # 获取因子值
    factor_values = filtered_factors[selected_factor_names[0]].loc[date]
    # 排序并选取前20只股票
    top_stocks = factor_values.sort_values(ascending=False).head(20).index
    # 计算当期组合收益率（等权重）
    portfolio_return = returns.loc[date, top_stocks].mean()
    portfolio_returns.append(portfolio_return)

# 将组合收益率转换为Series
portfolio_returns = pd.Series(portfolio_returns, index=returns.index)

# 绘制组合收益曲线
cumulative_returns = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns)
plt.title('Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()

# 计算年化收益率和年化波动率
annual_return = (cumulative_returns[-1]) ** (12 / len(cumulative_returns)) - 1
annual_volatility = portfolio_returns.std() * np.sqrt(12)

# 计算夏普比率（假设无风险利率为0）
sharpe_ratio = annual_return / annual_volatility

print(f'组合年化收益率：{annual_return:.2%}')
print(f'组合年化波动率：{annual_volatility:.2%}')
print(f'组合夏普比率：{sharpe_ratio:.2f}')
