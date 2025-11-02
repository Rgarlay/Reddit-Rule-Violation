from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from reddit_data.constants import training_pipeline
from datetime import datetime


class TrainingPipelineConfig:
    def __init__(self, timestamp = datetime.now()):
        timestamp = timestamp.strftime(format='%m-%d-%y-%H-%M-%S')
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)


class DataIngestionConfig:
    def __init__(self,training_config: TrainingPipelineConfig):
        self.database_name = training_pipeline.DATABASE_NAME
        self.collection_name = training_pipeline.COLLECTION_NAME
        self.train_test_split_ratio = training_pipeline.TRAIN_TEST_SPLIT_RATIO

        self.data_ingestion_dir = os.path.join(training_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR)


        self.feature_store_file_path = os.path.join( self.data_ingestion_dir,
                                         training_pipeline.DATA_INGESTION_FEATURE_STORE_FILE_PATH,
                                         training_pipeline.FILE_NAME)

        self.train_file_path = os.path.join( self.data_ingestion_dir,
                                            training_pipeline.DATA_INGESTION_TRAIN_FILE_PATH,
                                            training_pipeline.TRAIN_FILE_NAME)
        
        self.test_file_path = os.path.join( self.data_ingestion_dir,
                                           training_pipeline.DATA_INGESTION_TRAIN_FILE_PATH,
                                           training_pipeline.TEST_FILE_NAME)
        

