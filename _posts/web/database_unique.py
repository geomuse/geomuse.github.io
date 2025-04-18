from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
db = client["stock_data"]  # 数据库名称
print(collection:=db['AAPL'])

def unique(collection):
    date = []
    open = []
    high = []
    low = []
    close = []
    volume = []

    for record in collection.find():
        if record['Date'] not in date : 
            date.append(record['Date'])
            open.append(record['Open'])
            high.append(record['High'])
            low.append(record['Low']) 
            close.append(record['Close'])
            volume.append(record['Volume'])

    data = pd.DataFrame({
        'Date' : date , 
        'Open' : open , 
        'High' : high , 
        'Low'  : low , 
        'Close' : close , 
        'Volume' : volume
    })

    return data

print(unique(collection))