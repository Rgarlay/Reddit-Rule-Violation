from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
from reddit_data.entity.entity_config import DataIngestionConfig
import os,sys

import pandas as pd
import numpy as np
import pymongo
import json
from sklearn.model_selection import train_test_split
from reddit_data.constants.training_pipeline import TRAIN_TEST_SPLIT_RATIO
from reddit_data.entity.artifact_config import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('uri')


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        
    ##method to import data from the mongodb
    def pulling_from_mongodb(self):
        try:
            self.collection_name = self.data_ingestion_config.collection_name
            self.database_name = self.data_ingestion_config.database_name

            mongo_db = pymongo.MongoClient(MONGO_DB_URL)

            collection = mongo_db[self.database_name][self.collection_name]

            records = list(collection.find())

            df = pd.read_csv(records)

            cols_to_drop = ['row_id']

            df.drop(columns=cols_to_drop, inplace=True)

            if '_id' in df.columns:
                df.drop(columns = ['_id'], inplace=True)

            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def data_export_to_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_name = os.path.join(feature_store_file_path)

            os.makedirs(dir_name,exist_ok=True)

            dataframe.to_csv(feature_store_file_path, header=True, index=False)
        except Exception as e:
            raise CustomException(e,sys)
        
    def performing_train_test_split(self, dataframe:pd.DataFrame):
        try:
            train_file, test_file = train_test_split(dataframe,
                                                    train_size=TRAIN_TEST_SPLIT_RATIO,
                                                    random_state=42)
            

            train_file_path = self.data_ingestion_config.train_file_path

            dir_name = os.path.join(train_file_path)

            os.makedirs(dir_name, exist_ok= True)

            train_file.to_csv(self.data_ingestion_config.train_file_path, header=True, index=False)

            test_file.to_csv(self.data_ingestion_config.test_file_path, header=True, index=False)

            return train_file, test_file
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            feature_file = self.pulling_from_mongodb()
            self.data_export_to_feature_store(feature_file)
            train_file, test_file = self.performing_train_test_split(feature_file)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.train_file_path,
                                                            test_file_path=self.data_ingestion_config.test_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
        

 
        
