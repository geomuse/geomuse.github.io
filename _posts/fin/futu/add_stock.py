#%%
import pandas as pd
import requests

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html"
}

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
respone = requests.get(url,headers=headers)
df_sp500 = pd.read_html(respone.text)[0]

sp500_tickers = df_sp500["Symbol"].tolist()

# Yahoo Finance 使用 - 号
sp500_tickers = [t.replace(".", "-") for t in sp500_tickers]

print(f"S&P 500 股票数量: {len(sp500_tickers)}")
print(sp500_tickers[:20])

#%%

url = "https://zh.wikipedia.org/wiki/納斯達克100指數"
respone = requests.get(url,headers=headers)
df = pd.read_html(respone.text)[2]
# nasdaq_all = df[df["ETF"] == "N"]["Symbol"].tolist()
nasdaq_100 = [str(t).replace(".", "-") for t in df['股票代號']]

print(f"NASDAQ 全市场股票数: {len(nasdaq_100)}")
print(nasdaq_100[:20])

all_tickers = sorted(set(sp500_tickers + nasdaq_100))

print(f"合并后股票数量: {len(all_tickers)}")

pd.DataFrame(all_tickers, columns=["Ticker"]).to_csv(
    "/home/geo/Downloads/geo/_posts/fin/futu/all_tickers.csv", index=False
)