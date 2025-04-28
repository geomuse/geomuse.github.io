import yfinance as yf
from pymongo import MongoClient
import pandas as pd

# 获取美股数据 (以AAPL为例)
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data['Date'] = data.index
    return data

# 连接到本地 MongoDB 数据库
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_data"]  # 数据库名称

# 将 Pandas DataFrame 转换为字典并存储到 MongoDB
def store_data_to_mongodb(collection, dataframe):
    records = dataframe.to_dict("records")  # 转换为字典列表
    collection.insert_many(records)
    print(f"成功插入 {len(records)} 条记录！")

# 示例：存储数据
# store_data_to_mongodb(collection, data)

# 从 MongoDB 查询数据
# 可以通过 pymongo 或 MongoDB 客户端工具验证数据存储是否成功
# for record in collection.find().limit(5):
#     print(record)

def batch_store_stocks(tickers, start_date, end_date):
    for ticker in tickers:
        print(f"正在获取 {ticker} 数据...")
        data = get_stock_data(ticker, start_date, end_date)
        collection = db[ticker]  # 每只股票创建一个集合
        store_data_to_mongodb(collection, data)

# 示例：批量存储多只股票数据
batch_store_stocks(["AAPL", "MSFT", "GOOGL"], "2022-01-01", "2025-12-31")
