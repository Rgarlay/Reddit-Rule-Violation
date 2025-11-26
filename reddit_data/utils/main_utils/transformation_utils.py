
from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys

from sklearn.base import BaseEstimator, TransformerMixin
import torch

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

class CleaningEmbedTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, tokenizer, model, padding='max_length', batch_size=32):
        self.padding = padding
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model = model

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            self.load_model()

            cleaned = self.clean_text(X)
            embedded = self.embed_text(cleaned)

            return embedded
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
            
    def embed_text(self, text_list):
        try:
            if self.padding == 'max_length':
                tokenized = self.tokenizer(text_list, padding ='max_length',
                                                    max_length = 193, truncation = True, return_tensors = 'pt')
            else:
                tokenized = self.tokenizer(text_list, padding =True,
                                        truncation = True, return_tensors = 'pt')

            embeddings = []

            with torch.no_grad():
                for i in range(0, len(tokenized['input_ids']), self.batch_size):
                    batch = {k: v[i:i+self.batch_size] for k, v in tokenized.items()}
                    output = self.model(**batch)
                    embeddings.append(output.last_hidden_state[:, 0, :].cpu())

            embedded_tensor = torch.cat(embeddings, dim=0)

            return embedded_tensor.tolist()
        
        except Exception as e:
            raise CustomException(e,sys)


class CleaningEmbed:
    def __init__(self):
        try: 
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            raise CustomException(e,sys)

    def clean_text(self, text_list:list):
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
                logging.info(f"Starting text embedding for {len(text_list)} items")
                if padding == 'max_length':
                    tokenized = self.tokenizer(text_list, padding ='max_length',
                                                        max_length = 193, truncation = True, 
                                                        return_tensors = 'pt')
                else:
                    tokenized = self.tokenizer(text_list, padding =True,
                                            truncation = True, return_tensors = 'pt')

                embeddings = []
                self.batch_size = 32
                with torch.no_grad():
                    for i in range(0, len(tokenized['input_ids']),self.batch_size ):
                        batch = {k: v[i:i+self.batch_size] for k, v in tokenized.items()}
                        output = self.model(**batch)
                        embeddings.append(output.last_hidden_state[:, 0, :].cpu())

                embedded_tensor = torch.cat(embeddings, dim=0)

                logging.info("Embedding completed")
                return embedded_tensor.tolist()
            
            except Exception as e:
                raise CustomException(e,sys)