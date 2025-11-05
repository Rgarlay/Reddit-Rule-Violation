from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
from reddit_data.entity.entity_config import DataValidationConfig
from reddit_data.entity.artifact_config import DataIngestionArtifact
import os,sys

import pandas as pd
import json

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    ### Let's import our data first

    def importing_data_from_dir(self,file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise CustomException(e,sys)
