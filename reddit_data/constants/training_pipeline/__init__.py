

'''
GENERAL CONSTANTS
'''

DATABASE_NAME: str = 'Reddit_database'
COLLECTION_NAME: str = 'Session_3'
TRAIN_TEST_SPLIT_RATIO: float = 0.80
PIPELINE_NAME: str = 'reddit_violation'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
FILE_NAME: str = 'feature.csv'
TARGET_COL: str = 'rule_violation'

AWS_BUCKET_NAME: str = 'aws-reddit-rule-violation-classification'

'''
data ingestion constants will begin with DATA_INGESTION
'''

ARTIFACT_DIR: str = 'archieve'
DATA_INGESTION_DIR: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_FILE_PATH: str = 'feature_store'
DATA_INGESTION_TRAIN_FILE_PATH: str = 'ingested'

'''
Data validation constants will begin with DATA_VALIDATION
'''

DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_VALID_DIR: str = 'valid'
DATA_VALIDATION_INVALID_DIR: str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR: str = 'drift report' 
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = 'drift_report.yml' 

'''
Data Transformation constants will begin with DATA_TRANSFORMATION
'''

DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test.npy"

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJ_DIR_FOR_BODY: str = "transformed_object"


DATA_TRANSFORMATION_CLEANED_FILE_DIR: str = 'cleaned_text'
DATA_TRANSFORMATION_BODY_CLEANED_FILE: str = 'body_cleaned.txt'
DATA_TRANSFORMATION_RULE_CLEANED_FILE: str = 'rule_cleaned.txt'


'''
Model Training constants will begiin with MODEL_TRAINER
'''
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME: str = 'trained_model'
MODEL_TRAINER_MODEL_NAME: str = 'model.pkl'


'''
Tokenization and vocabulary artifact will begin with TOKEN_AND_VOCAB
'''
TOKEN_AND_VOCAB_BODY_TOKENIZER: str = 'body_tokenizer'
TOKEN_AND_VOCAB_RULE_TOKENIZER: str = 'rule_tokenizer'





