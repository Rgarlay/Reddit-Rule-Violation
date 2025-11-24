from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys

from reddit_data.entity.entity_config import (DataIngestionConfig, 
                                              TrainingPipelineConfig, 
                                              DataValidationConfig, 
                                              DataTransformationConfig,
                                              ModelTrainerConfig,
                                              TrainingPipelineConfig)

from reddit_data.components.ingestion import DataIngestion
from reddit_data.components.validation import DataValidation 
from reddit_data.components.transformation import DataTransformation 
from reddit_data.components.traininig import ModelTraining 

from reddit_data.entity.artifact_config import (DataIngestionArtifact,
                                                DataTransformationArtifact,
                                                DataValidationArtifact,
                                                ModelTrainerArtifact)
from reddit_data.constants.training_pipeline import AWS_BUCKET_NAME

from reddit_data.cloud.s3_sync import s3Sync

class TrainingPipeline:
    def __init__ (self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            self.sync_to_s3 = s3Sync()
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingest(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion_initiate = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion_initiate.initiate_data_ingestion()
                
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validate(self,data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validatoin_initiate = DataValidation(data_validation_config=self.data_validation_config, 
                                                  data_ingestion_artifact=data_ingestion_artifact)
            data_validatoin_artifact = data_validatoin_initiate.initiate_data_validation()

            return data_validatoin_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transform(self, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation_initiate = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                            data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation_initiate.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_train(self, data_transformation_artifact: DataTransformationArtifact):
        try:

            self.model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            
            model_trainer_initiate = ModelTraining(model_trainer_config=self.model_trainer_config,
                                                data_transformation_artifact = data_transformation_artifact)
            
            model_trainer_artifact = model_trainer_initiate.initiate_model_training()

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def s3_artifact_sync(self):
        try:
            
            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/artifact/"
            self.sync_to_s3.sync_folder_to_s3(aws_bucket_url=aws_bucket_url, 
                                              folder=self.training_pipeline_config.artifact_dir)
        except Exception as e:
            raise CustomException(e,sys)
        
    def s3_model_sync(self):
        try:
            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/model/"
            self.sync_to_s3.sync_folder_to_s3(aws_bucket_url=aws_bucket_url, 
                                              folder=self.training_pipeline_config.model_dir)
        except Exception as e:
            raise CustomException(e,sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.initiate_data_ingest()
            data_validation_artifact = self.initiate_data_validate(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.initiate_data_transform(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.initiate_model_train(data_transformation_artifact=data_transformation_artifact)
            self.s3_artifact_sync()
            self.s3_model_sync()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)