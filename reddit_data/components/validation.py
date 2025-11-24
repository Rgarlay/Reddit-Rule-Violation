from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
from reddit_data.entity.entity_config import DataValidationConfig
from reddit_data.entity.artifact_config import DataIngestionArtifact, DataValidationArtifact
import os, sys

import pandas as pd
from reddit_data.utils.main_utils.utils import load_yaml_file


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info("Initializing DataValidation class")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            logging.info("DataValidation initialization successful")
        except Exception as e:
            raise CustomException(e, sys)

    def importing_data_from_dir(self, file_path):
        try:
            logging.info(f"Reading dataset from: {file_path}")
            df = pd.read_csv(file_path)
            logging.info("File read successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def validate_no_of_cols(self, current_df):
        try:
            logging.info("Starting column validation")
            
            current_df = self.importing_data_from_dir(self.data_ingestion_artifact.test_file_path)
            
            file_path = r"data_schema\schema.yml"
            logging.info(f"Loading schema file: {file_path}")
            schema_file = load_yaml_file(file_path)

            expected_cols = list(schema_file.values())[0]
            actual_cols = list(current_df.columns)

            logging.info(f"Expected columns: {expected_cols}")
            logging.info(f"Current columns : {actual_cols}")

            if expected_cols == actual_cols:
                logging.info("Column validation successful: Schema matched")
                return True
            
            logging.info("Column validation failed: Schema mismatch")
            return False
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        try:
            logging.info("=== Data Validation Started ===")
            
            train_path = self.data_ingestion_artifact.train_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            logging.info(f"Train dataset path: {train_path}")
            logging.info(f"Test  dataset path: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Validating training dataset schema")
            train_valid = self.validate_no_of_cols(train_df)

            logging.info("Validating testing dataset schema")
            test_valid = self.validate_no_of_cols(test_df)

            logging.info(f"Training data schema valid? : {train_valid}")
            logging.info(f"Testing data schema valid?  : {test_valid}")

            output_dir = os.path.dirname(self.data_validation_config.valid_train_file_path)
            logging.info(f"Ensuring output directory exists: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            logging.info("Writing validated datasets to output directory")
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            logging.info("Creating DataValidationArtifact object")

            data_validation_artifact = DataValidationArtifact(
                valid_train_file_path=train_path,
                valid_test_file_path=test_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path
            )

            logging.info("=== Data Validation Successful ===")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
