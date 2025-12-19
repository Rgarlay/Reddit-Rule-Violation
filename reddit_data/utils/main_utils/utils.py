from reddit_data.exception.exception import CustomException
import os,sys
import pickle
import re
import sentencepiece as spm
import spacy

def tokenization_of_text(tokenizer, file_name: list, pad_id) -> list:
        '''
        Here we actually tokenize our text
        
        :tokenizer: tokenizer defined in above step
        :param file_name: list file. It's the one that's cleaned from regex
        :param pad_id: padding token, Necessary for evevning out the padding
        :return: {list: padded, integer: max_length}
        '''
        try:
            tokenized_stored = [tokenizer.encode(i,add_bos = True, add_eos = True , out_type = int)
            for i in file_name]
            max_length = max(len(x) for x in tokenized_stored)
            text_tokenized = [x + [pad_id] * (max_length - len(x)) for x in tokenized_stored]

            return text_tokenized
        
        except Exception as e:
            raise CustomException(e,sys)


def load_yaml_file(file_path):
    try:
        import yaml
        with open(file_path, 'rb') as file:
            lines = yaml.safe_load(file)
        return lines
    except Exception as e:
        raise CustomException(e,sys)

def save_yaml_file(file_path, content, replace):
    try:
        import yaml
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.safe_dump(content,file)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_pickle_file(file_to_save, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(file_to_save, file)
    except Exception as e:
        raise CustomException(e,sys)

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            lines = pickle.load(file)
            return lines
    except Exception as e:
        raise CustomException(e,sys)

def clean_text(text_list: list):

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
    

def save_txt_file(list_text_file: list, file_path, replace):
    '''
    Saving text file.
    '''
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding = 'utf-8') as f:
            for line in list_text_file:
                f.write(line.strip() + '\n')
    except Exception as e:
        raise CustomException(e,sys)


def get_vocab_size(file_name: list) -> int:
    '''
    This code will give us the vocab size
    '''
    try:
        file_to_str = ' '.join(map(str, file_name))

        eng_lang = spacy.blank('en')
        eng_lang.max_length = 10e7

        doc = eng_lang(file_to_str)
        tokenized_word_file = [i.text for i in doc]

        text_vocab = list(set(tokenized_word_file))
        return  len(text_vocab)

    except Exception as e:
        raise CustomException(e,sys)
    








