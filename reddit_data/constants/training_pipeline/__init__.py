

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

