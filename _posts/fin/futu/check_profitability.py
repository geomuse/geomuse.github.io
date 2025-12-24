import yfinance as yf
import numpy as np
import pandas as pd

TICKERS = pd.read_csv('/home/geo/Downloads/geo/_posts/fin/futu/all_tickers.csv')


# TICKERS = ["AAPL","MSFT"]

def safe_get(df, possible_keys):
    for key in possible_keys:
        if key in df.index:
            return df.loc[key].iloc[0]
    return None

results = []

print('loading...')

for ticker in TICKERS.Ticker:

        stock = yf.Ticker(ticker)
        print(f'{stock}...')
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
        info = stock.info

        revenue = safe_get(fin, [
            "Total Revenue",
            "Revenue",
            "Total Revenues",
            "Net Revenue"
        ])

        gross_profit = safe_get(fin, ["Gross Profit", "GrossProfit"])
        operating_income = safe_get(fin, ["Operating Income", "OperatingIncome"])
        net_income = safe_get(fin, ["Net Income", "NetIncome"])

        total_assets = safe_get(bs, [
            "Total Assets",
            "Assets",
            "TotalAssets",
            "Total Assets Gross"
        ])

        equity = safe_get(bs, [
            "Total Stockholder Equity",
            "Total Shareholder Equity",
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Total Equity"
        ])

        current_liabilities = safe_get(bs, [
            "Total Current Liabilities",
            "Current Liabilities"
        ])

        operating_cf = safe_get(cf, [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Net Cash Provided by Operating Activities",
            "Cash Flow from Operations"
        ])

        capex = safe_get(cf, [
            "Capital Expenditures",
            "Purchase of Property, Plant and Equipment",
            "Additions to Property, Plant and Equipment"
        ])

        # if None in [revenue, gross_profit, operating_income, net_income,
        #             total_assets, equity, current_liabilities, operating_cf, capex]:
        #     continue

        # if operating_cf is not None and capex is not None:
        #     fcf = operating_cf + capex
        # else:
        #     fcf = None

        liab = safe_get(bs,[
            "Total Current Liabilities",
            "Current Liabilities",
            "Total Liab"
        ])
        try :
            invested_capital = total_assets - liab

            roic = operating_income / invested_capital
            # roe = net_income / equity
            # gross_margin = gross_profit / revenue
            # operating_margin = operating_income / revenue
            # fcf_margin = fcf / revenue

            results.append({
                "Ticker": ticker,
                "ROIC": roic,
            })

            print(f'{ticker} procssing...')
        except : 
            continue

df = pd.DataFrame(results).dropna()

df = df[df["ROIC"] > 0.10]
# print(df)

# Winsorize 防极值
# for col in df.columns[1:]:
#     df[col] = np.clip(df[col], df[col].quantile(0.05), df[col].quantile(0.95))

# # 标准化
# for col in df.columns[1:]:
#     df[col] = (df[col] - df[col].mean()) / df[col].std()

# 综合评分（权重可调）
# df["Profitability_Score"] = (
#     df["ROIC"] * 0.30 +
#     df["Operating_Margin"] * 0.20 +
#     df["FCF_Margin"] * 0.20 +
#     df["Gross_Margin"] * 0.15 +
#     df["ROE"] * 0.15
# )

df = df.sort_values("ROIC", ascending=False)

print("\n🏆 实战选股 Top 10(S&P500 + NASDAQ100)")
print(df.head(10))

df.to_csv("/home/geo/Downloads/geo/_posts/fin/futu/profitability_stock_selection.csv", index=False)