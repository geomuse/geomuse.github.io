from pymongo import MongoClient 
url = "mongodb://127.0.0.1:27017/"
client = MongoClient(url)

db = client.db

acccount_collection = db.account

new_account = {
    "account_holder" : "Linu" , 
    "balance" : 50352434
}

if __name__ == '__main__' : 

    for db_name in client.list_database_names():
        print(db_name)
    
    result = acccount_collection.insert_one(new_account)
    print(result.inserted_id)

    client.close()