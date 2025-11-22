from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
from reddit_data.entity.entity_config import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
import os,sys

from reddit_data.entity.artifact_config import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from reddit_data.components.ingestion import DataIngestion
from reddit_data.components.validation import DataValidation 
from reddit_data.components.transformation import DataTransformation 

if __name__ == "__main__":
    try:
        training_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_config)
        data_ingestion_initiate = DataIngestion(data_ingestion_config)

        data_ingestion_artiact = data_ingestion_initiate.initiate_data_ingestion()

        data_validation_config = DataValidationConfig(training_config)
        data_validatoin_initiate = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artiact)

        data_validatoin_artifact = data_validatoin_initiate.initiate_data_validation()

        data_transformation_config = DataTransformationConfig(training_config)

        data_transformation_initiate = DataTransformation(data_validation_artifact=data_validatoin_artifact,
                                                          data_transformation_config=data_transformation_config)
        
        data_transformation_artifact = data_transformation_initiate.initiate_data_transformation()


        print(data_transformation_artifact)
    except Exception as e:
        raise CustomException(e,sys)