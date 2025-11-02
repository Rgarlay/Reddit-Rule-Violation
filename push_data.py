from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from dotenv import load_dotenv
import pymongo
import pandas as pd
import json



load_dotenv()

MONGODB_URI = os.getenv('uri')

class push_data:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
    
    def convert_to_json(self,file_path):
        try:
            df = pd.read_csv(file_path)
            df = df.reset_index()
            json_data = list(json.loads(df.T.to_json()).values())
            return json_data
        
        except Exception as e:
            raise CustomException(e,sys)
    def push_to_mongodb(self, collection, database, records):
        
        try:
            self.collection = collection
            self.records = records
            self.database = database


            mongo_db_connection = pymongo.MongoClient(MONGODB_URI)
            
            collection_name = mongo_db_connection[self.collection]

            database_name = collection_name[self.database]

            database_name.insert_many(self.records)

            return print(len(self.records))
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    database_name = 'Reddit_database'
    collection_name = "Session_3"
    new_insertion = push_data()
    file_path = r'archieve\reddit_dataset.csv'
    data_to_mongodb = new_insertion.convert_to_json(file_path)
    new_insertion.push_to_mongodb(database=database_name,collection=collection_name, records=data_to_mongodb)
