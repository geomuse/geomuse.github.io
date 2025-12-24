import yfinance as yf
import pandas as pd
TICKERS = pd.read_csv('/home/geo/Downloads/geo/_posts/fin/futu/profitability_stock_selection.csv')
TICKERS = TICKERS['Ticker']

results = []

for t in TICKERS:
    data = yf.Tickers(t)
    info = data.tickers[t].info
    price = info["currentPrice"]
    dividend = info.get("dividendRate", 0)
    expensive, reasonable, cheap = dividend * 40 , dividend * 20 , dividend * 10

    results.append({
                "ticker": t,
                "price": price,
                "expensive" : expensive,
                "reasonable" : reasonable ,
                "cheap" : cheap , 
                "dividend" : dividend
            })

df = pd.DataFrame(results).dropna()

print("\n🏆 实战选股 Top 10")
print(df.head(10))

df.to_csv("/home/geo/Downloads/geo/_posts/fin/futu/estimate_stock_selection.csv", index=False)