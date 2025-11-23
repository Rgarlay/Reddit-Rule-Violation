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
        
    def clean_text(self, text_list):
        try:
            cleaned_list = []
            pattern1 = re.compile(r'https?:\/\/\S+|www\.\S+|Https?:\/\/\S+|\S+\.com\S+|\S+\.com|\[.*?\]|\S+ \. com.*')
            pattern2 = re.compile(r'<.*?>')
            pattern3 =  re.compile(r'#\S+|@\S+|\S+\@\S+|\S+@')
            pattern4 = re.compile(r'u\/\S+|r\/\S+')
            pattern5 = re.compile(r"["
                            u"\U0001F600-\U0001F64F"  
                            u"\U0001F300-\U0001F5FF"  
                            u"\U0001F680-\U0001F6FF"  
                            u"\U0001F1E0-\U0001F1FF"  
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
            pattern6 = re.compile(r'\d|\\n')


            for text in text_list:
                text = pattern1.sub('', text)
                text = pattern2.sub('', text)       ##Removing HTML rags
                text = pattern3.sub('', text)       ## Removing Emails and Hashtags
                text = pattern4.sub('', text)       ### Removing username and subreddit mentions
                text = pattern5.sub('', text)       #emotions, symbols, pictographs, transport and map symbols, flags etx.
                text = pattern6.sub('', text)       ###Removing Numbers & \n spaces
                cleaned_list.append(text)

            return cleaned_list
        except Exception as e:
            raise CustomException(e,sys)
        

    def embed_text(self, text_list, padding=True):

        try:
            if padding == 'max_length':
                tokenized = tokenizer(text_list, padding ='max_length',
                                                    max_length = 193, truncation = True, 
                                                    return_tensors = 'pt')
            else:
                tokenized = tokenizer(text_list, padding =True,
                                        truncation = True, return_tensors = 'pt')

            embeddings = []
            self.batch_size = 32
            with torch.no_grad():
                for i in range(0, len(tokenized['input_ids']),self.batch_size ):
                    batch = {k: v[i:i+self.batch_size] for k, v in tokenized.items()}
                    output = model(**batch)
                    embeddings.append(output.last_hidden_state[:, 0, :].cpu())

            embedded_tensor = torch.cat(embeddings, dim=0)

            return embedded_tensor.tolist()
        
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

            ## Body Text
            text_column = 'body'
            
            train_text_file = list(train_df_concatinated[text_column])
            test_text_file = list(test_df_concatinated[text_column])

            train_text_cleaned = self.clean_text(train_text_file)
            test_text_cleaned = self.clean_text(test_text_file)

            train_text_embedded = self.embed_text(text_list = train_text_cleaned,padding='max_length')
            test_text_embedded = self.embed_text(text_list = test_text_cleaned,padding='max_length')

            ##Rule part
            rule_column = 'rule'
            train_rule_file = list(train_df_concatinated[rule_column])
            test_rule_file = list(test_df_concatinated[rule_column])

            train_rule_embedded = self.embed_text(train_rule_file)
            test_rule_embedded = self.embed_text(test_rule_file)
        

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
            

            data_transformation_artifact = DataTransformationArtifact(
                train_obj_file_path = self.data_transformation_config.data_transformed_train_file_path,
                test_obj_file_path = self.data_transformation_config.data_transformed_test_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)