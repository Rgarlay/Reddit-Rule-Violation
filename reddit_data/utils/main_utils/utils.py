from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import yaml
import os,sys
import pickle


def load_yaml_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            lines = yaml.safe_load(file)
        return lines
    except Exception as e:
        raise CustomException(e,sys)

def save_yaml_file(file_path, content, replace):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.safe_dump(content,file)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_pickle_file(file_to_save, file_path, replace):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
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
