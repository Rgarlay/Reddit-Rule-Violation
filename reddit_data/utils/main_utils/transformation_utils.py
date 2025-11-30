from reddit_data.exception.exception import CustomException
import sys
import torch
from transformers import AutoTokenizer, AutoModel
import re


class CleaningEmbed:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            raise CustomException(e, sys)

    def clean_text(self, text_list: list):
        try:
            cleaned_list = []

            pattern1 = re.compile(
                r'https?:\/\/\S+|www\.\S+|Https?:\/\/\S+|\S+\.com\S+|\S+\.com|\[.*?\]|\S+ \. com.*')
            pattern2 = re.compile(r'<.*?>')  # HTML tags
            pattern3 = re.compile(r'#\S+|@\S+|\S+\@\S+|\S+@')  # emails, hashtags
            pattern4 = re.compile(r'u\/\S+|r\/\S+')  # usernames & subreddit mentions
            pattern5 = re.compile(
                r"["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols
                u"\U0001F680-\U0001F6FF"  # transport/map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE
            )
            pattern6 = re.compile(r'\d|\\n')  # digits and literal '\n'

            for text in text_list:
                text = pattern1.sub('', text)
                text = pattern2.sub('', text)
                text = pattern3.sub('', text)
                text = pattern4.sub('', text)
                text = pattern5.sub('', text)
                text = pattern6.sub('', text)
                cleaned_list.append(text)

            return cleaned_list

        except Exception as e:
            raise CustomException(e, sys)

    def embed_text(self, text_list, padding=True):
        try:

            if padding == 'max_length':
                tokenized = self.tokenizer(
                    text_list,
                    padding='max_length',
                    max_length=193,
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                tokenized = self.tokenizer(
                    text_list,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

            embeddings = []
            self.batch_size = 32

            with torch.no_grad():
                for i in range(0, len(tokenized['input_ids']), self.batch_size):
                    batch = {k: v[i:i + self.batch_size] for k, v in tokenized.items()}
                    output = self.model(**batch)
                    embeddings.append(output.last_hidden_state[:, 0, :].cpu())

            embedded_tensor = torch.cat(embeddings, dim=0)

            return embedded_tensor.tolist()

        except Exception as e:
            raise CustomException(e, sys)
