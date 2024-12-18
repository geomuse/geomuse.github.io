import yfinance as yf
import pandas as pd

# 从标普500成分股提取美股代码
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    symbols = sp500_table['Symbol'].tolist()
    return symbols

# 显示结果
symbols = get_sp500_symbols()[0:1]
print(symbols)

download_data = yf.download(symbols,start="2024-12-01",interval="1mo",progress=False)
print(df := pd.DataFrame(download_data))

df.to_csv(f'{symbols[0]}.csv',index=True)

# 保存为CSV文件
# sp500_df = pd.DataFrame(symbols, columns=["Ticker"])
# sp500_df.to_csv("sp500_tickers.csv", index=False)
# print("股票代码已保存至 sp500_tickers.csv")
