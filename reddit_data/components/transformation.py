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
        
    def cleaning_text(self, dataframe_text_feature_as_list):
        try:
            text_file = dataframe_text_feature_as_list
            pattern = re.compile('https?:\/\/\S+|www\.\S+|Https?:\/\/\S+|\S+\.com\S+|\S+\.com|\[.*?\]|\S+ \. com.*')
            for i in range(len(text_file)):
                text_file[i] = pattern.sub(r'',text_file[i])

            ##Removing HTML rags
            pattern = re.compile('<.*?>')
            for i in range(len(text_file)):
                text_file[i] = pattern.sub(r'',text_file[i])

            ## Removing Emails and Hashtags
            pattern = re.compile('#\S+|@\S+|\S+\@\S+|\S+@')
            for i in range(len(text_file)):
                text_file[i] = pattern.sub(r'',text_file[i])

            ### Removing username and subreddit mentions
            pattern = re.compile('u\/\S+|r\/\S+')
            for i in range(len(text_file)):
                text_file[i] = pattern.sub(r'',text_file[i])

            #emotions, symbols, pictographs, transport and map symbols, flags etx.
            pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    "]+", flags=re.UNICODE)
            for i in range(len(text_file)):
                text_file[i] = pattern.sub(r'',text_file[i])

            ##Removing Numbers & \n spaces
            pattern = re.compile('\d|\\n')
            for i in range(len(text_file)):
                text_file[i] = pattern.sub(r'',text_file[i])

            return text_file
        except Exception as e:
            raise CustomException(e,sys)
    
    def embedded_text(text_feature_file, model, tokenizer, padding ,batch_size = 32):
        try:
            if padding == 'max_length':
                tokenized_text = tokenizer(text_feature_file, padding ='max_length',
                                        max_length = 193, truncation = True, return_tensors = 'pt')
            else:
                tokenized_text = tokenizer(text_feature_file, padding =True,
                                        truncation = True, return_tensors = 'pt')
            embeddings = []
            for i in range(0, len(tokenized_text['input_ids']), batch_size):
                batch = {k: v[i:i+batch_size].to(model.device) for k, v in tokenized_text.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                    # Use the embeddings of the first token (CLS token) as the sentence embedding
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
            troch_concatinated = torch.cat(embeddings, dim=0)
            embedded_text = [emb.tolist() for emb in troch_concatinated]

            return embedded_text
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self):
        try:
            train_file_path = self.data_transformation_config.data_transformed_train_file_path
            test_file_path = self.data_transformation_config.data_transformed_test_file_path

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            train_df_concatinated = self.taking_apart_and_concatinating(train_df)
            test_df_concatinated = self.taking_apart_and_concatinating(test_df)

            text_column = 'body'
            rule_column = 'rule'

            train_text_file = list(train_df_concatinated[text_column])
            train_rule_file = list(train_df_concatinated[rule_column])

            test_text_file = list(test_df_concatinated[text_column])
            test_rule_file = list(test_df_concatinated[rule_column])


            train_text_cleaned = self.cleaning_text(train_text_file)
            test_text_cleaned = self.cleaning_text(test_text_file)


            ## For text
            train_text_embedded = self.embedded_text(text_feature_file = train_text_cleaned,
                                                                 model = model, padding = 'max_length',tokenizer=tokenizer )
            test_text_embedded = self.embedded_text(text_feature_file = test_text_cleaned,
                                                                 model = model, padding = 'max_length',tokenizer=tokenizer )
        

            ## For rule
            train_text_embedded = self.embedded_text(text_feature_file = train_text_cleaned,
                                                                 model = model, padding = True,tokenizer=tokenizer)
            test_text_embedded = self.embedded_text(text_feature_file = test_text_cleaned,
                                                                 model = model, padding = True,tokenizer=tokenizer)
        

            

            train_df_concatinated[text_column] = train_text_embedded
            test_df_concatinated[text_column] = test_text_embedded

            train_df_concatinated[text_column] = train_rule_file
            test_df_concatinated[rule_column] = test_rule_file

            dirname = os.path.dirname(self.data_transformation_config.data_transformed_train_file_path)

            os.makedirs(dirname, exist_ok=True)

            save_pickle_file(file_to_save=train_df_concatinated,
                            file_path = self.data_transformation_config.data_transformed_train_file_path)
            
            save_pickle_file(file_to_save = test_df_concatinated,
                             file_path=self.data_transformation_config.data_transformed_test_file_path)
            

            data_transformation_artifact = DataTransformationArtifact(
                train_file_path = self.data_transformation_config.data_transformed_train_file_path,
                test_file_path = self.data_transformation_config.data_transformed_test_file_path,
            )

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)