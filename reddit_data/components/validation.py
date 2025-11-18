from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
from reddit_data.entity.entity_config import DataValidationConfig
from reddit_data.entity.artifact_config import DataIngestionArtifact, DataValidationArtifact
import os,sys

import pandas as pd
import json

from reddit_data.utils.main_utils.utils import load_yaml_file

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

    def validate_no_of_cols(self, current_df):
        try:
            current_df = self.importing_data_from_dir(self.data_ingestion_artifact.test_file_path)
            file_path=r'C:\Users\rgarlay\Desktop\DS\Reddit_Project\Reddit-Rule-Violation\data_schema\schema.yml'
            schema_file = load_yaml_file(file_path)
            list_of_list_of_column_names = list(schema_file.values()) 
            column_names = list_of_list_of_column_names[0]
            if column_names == list(current_df.columns):
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validation(self):
        try:

            train_path = self.data_ingestion_artifact.train_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_file_column_validation = self.validate_no_of_cols(train_df)
            test_file_column_validation = self.validate_no_of_cols(test_df)

            logging.info(f'Are training columns same as desired columns? : - {train_file_column_validation} ')
            logging.info(f'Are testing columns same as desired columns? : - {test_file_column_validation} ')

            dir_name = os.path.dirname(self.data_validation_config.valid_train_file_path)

            os.makedirs(dir_name,exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, header=True, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, header=True, index=False)


            data_validattion_artifact = DataValidationArtifact(
                valid_train_file_path=train_path,
                valid_test_file_path=test_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path
            )

            return data_validattion_artifact
        
        except Exception as e:
            raise CustomException(e, sys)