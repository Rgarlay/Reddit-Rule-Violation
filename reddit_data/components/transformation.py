from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from reddit_data.utils.main_utils.utils import load_pickle_file, save_pickle_file

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re
from reddit_data.entity.artifact_config import DataValidationArtifact, DataTransformationArtifact
from reddit_data.entity.entity_config import DataTransformationConfig
import torch
from reddit_data.utils.main_utils.transformation_utils import CleaningEmbedTransformer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def importing_tokenizer_and_model(self):
        try:
            model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise CustomException(e,sys)
    
    def taking_apart_and_concatinating(self, dataframe:pd.DataFrame):
        try:
            df = dataframe
            
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

            return df_concatinated

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            train_df = train_df.head(20)
            test_df = test_df.head(10)

            train_df_concatinated = self.taking_apart_and_concatinating(train_df)
            test_df_concatinated = self.taking_apart_and_concatinating(test_df)

            text_column = 'body'
            rule_column = 'rule'

            train_text_file = list(train_df_concatinated[text_column])
            test_text_file = list(test_df_concatinated[text_column])

            train_rule_file = list(train_df_concatinated[rule_column])
            test_rule_file = list(test_df_concatinated[rule_column])

            body_cleaning_embed = CleaningEmbedTransformer(padding='max_length')
            rule_cleaning_embed = CleaningEmbedTransformer(padding=True)

            train_text_cleaned = body_cleaning_embed.clean_text(train_text_file)
            test_text_cleaned = body_cleaning_embed.clean_text(test_text_file)


            ## For text
            train_text_embedded = body_cleaning_embed.embed_text(train_text_cleaned)
            test_text_embedded = body_cleaning_embed.embed_text(test_text_cleaned)

            ## For rule

            train_rule_embedded = rule_cleaning_embed.embed_text(train_rule_file)
            test_rule_embedded = rule_cleaning_embed.embed_text(test_rule_file)
        

            train_df_concatinated[text_column] = train_text_embedded
            test_df_concatinated[text_column] = test_text_embedded

            train_df_concatinated[rule_column] = train_rule_embedded
            test_df_concatinated[rule_column] = test_rule_embedded

            dirname = os.path.dirname(self.data_transformation_config.data_transformed_train_file_path)

            os.makedirs(dirname, exist_ok=True)

            save_pickle_file(file_to_save=train_df_concatinated,
                            file_path = self.data_transformation_config.data_transformed_train_file_path)
            
            save_pickle_file(file_to_save = test_df_concatinated,
                             file_path=self.data_transformation_config.data_transformed_test_file_path)
            
            save_pickle_file(file_to_save = body_cleaning_embed, 
                             file_path = self.data_transformation_config.transformation_obj_file_path_for_body)
            save_pickle_file(file_to_save = rule_cleaning_embed, 
                             file_path = self.data_transformation_config.transformation_obj_file_path_for_rule)
            

            data_transformation_artifact = DataTransformationArtifact(
                train_obj_file_path = self.data_transformation_config.data_transformed_train_file_path,
                test_obj_file_path = self.data_transformation_config.data_transformed_test_file_path,
                transformed_obj_file_path_for_body_text=self.data_transformation_config.transformation_obj_file_path_for_body,
                transformed_obj_file_path_for_rule_text=self.data_transformation_config.transformation_obj_file_path_for_rule
            )

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)