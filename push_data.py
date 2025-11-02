from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from dotenv import load_dotenv
import pymongo

load_dotenv()

MONGODB_URI = os.getenv('uri')

