from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from reddit_data.utils.main_utils.utils import  save_pickle_file
from reddit_data.utils.main_utils.transformation_utils import CleaningEmbed

import pandas as pd
from reddit_data.entity.artifact_config import DataValidationArtifact, DataTransformationArtifact
from reddit_data.entity.entity_config import DataTransformationConfig

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        try:
            logging.info("Initializing DataTransformation")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def taking_apart_and_concatinating(self, df:pd.DataFrame):
        try:
            logging.info("Starting data concatenation")
            x_positive_1_df = df[['rule', 'positive_example_1']].rename(columns={'positive_example_1': 'body'})
            x_positive_2_df = df[['rule', 'positive_example_2']].rename(columns={'positive_example_2': 'body'})
            x_negative_1_df = df[['rule', 'negative_example_1']].rename(columns={'negative_example_1': 'body'})
            x_negative_2_df = df[['rule', 'negative_example_2']].rename(columns={'negative_example_2': 'body'})
            x_body_df = df[['rule','body','rule_violation']]

            x_positive_1_df['rule_violation'] = 1
            x_positive_2_df['rule_violation'] = 1
            x_negative_1_df['rule_violation'] = 0
            x_negative_2_df['rule_violation'] = 0

            df_concatinated = pd.concat([x_positive_1_df, x_positive_2_df, 
                                         x_negative_1_df, x_negative_2_df, 
                                         x_body_df], axis = 0).reset_index(drop=True)

            logging.info("Data concatenation completed")
            return df_concatinated

        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self):
        try:

            logging.info("Starting data transformation process")

            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            logging.info("Reading train and test CSV files")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            train_df = train_df.head(10)
            test_df = test_df.head(5)

            logging.info("Concatenating train data")
            train_df_concatinated = self.taking_apart_and_concatinating(train_df)
            logging.info("Concatenating test data")
            test_df_concatinated = self.taking_apart_and_concatinating(test_df)

            ## Body Text
            text_column = 'body'
            
            logging.info("Cleaning train body text")
            train_text_file = list(train_df_concatinated[text_column])
            logging.info("Cleaning test body text")
            test_text_file = list(test_df_concatinated[text_column])

            clean_and_embed = CleaningEmbed()

            logging.info("Embedding train body text")
            train_text_cleaned = clean_and_embed.clean_text(train_text_file)
            train_text_embedded = clean_and_embed.embed_text(train_text_cleaned,padding = 'max_length')


            logging.info("Embedding test body text")
            test_text_cleaned = clean_and_embed.clean_text(test_text_file)
            test_text_embedded = clean_and_embed.embed_text(test_text_cleaned, padding = 'max_length')
            
            ##Rule part
            rule_column = 'rule'
            train_rule_file = list(train_df_concatinated[rule_column])
            test_rule_file = list(test_df_concatinated[rule_column])

            logging.info("Embedding train rule text")

            train_rule_embedded = clean_and_embed.embed_text(train_rule_file)

            logging.info("Embedding test rule text")
            test_rule_embedded = clean_and_embed.embed_text(test_rule_file)
        
            train_df_concatinated[text_column] = train_text_embedded
            test_df_concatinated[text_column] = test_text_embedded

            train_df_concatinated[rule_column] = train_rule_embedded
            test_df_concatinated[rule_column] = test_rule_embedded

            dirname = os.path.dirname(self.data_transformation_config.data_transformed_train_file_path)

            logging.info("Creating directories if missing")
            os.makedirs(dirname, exist_ok=True)

            logging.info("Saving transformed train and test pickle files")
            save_pickle_file(file_to_save=train_df_concatinated,
                            file_path = self.data_transformation_config.data_transformed_train_file_path)
            
            save_pickle_file(file_to_save = test_df_concatinated,
                             file_path=self.data_transformation_config.data_transformed_test_file_path)
            
            logging.info("Creating final DataTransformationArtifact")
            data_transformation_artifact = DataTransformationArtifact(
                train_obj_file_path = self.data_transformation_config.data_transformed_train_file_path,
                test_obj_file_path = self.data_transformation_config.data_transformed_test_file_path
            )
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)