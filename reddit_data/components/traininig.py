from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import sys
from reddit_data.entity.entity_config import ModelTrainerConfig
from reddit_data.entity.artifact_config import ModelTrainerArtifact, DataTransformationArtifact
from reddit_data.utils.main_utils.utils import load_pickle_file, save_pickle_file
from reddit_data.utils.ml_utils.main_ml_utils import prepare_rule_body_tensors

from reddit_data.utils.ml_utils.main_ml_utils import CustomDataset, ViolationClassifier
from reddit_data.utils.ml_utils.metrics import evaluate_result
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


class ModelTraining:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)  

    def train_and_evaluate(self, model, train_loader, test_loader):
        try:
            logging.info("Starting train_and_evaluate method...")
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=0.001)
            epochs = 25 
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for body_batch, rule_batch, y_batch in train_loader:
                    # Move data to the same device as the model
                    body_batch, rule_batch, y_batch = body_batch.to(self.device), rule_batch.to(self.device), y_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(body_batch, rule_batch)
                    loss = criterion(outputs, y_batch.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                logging.info("Evaluating on test data...")            
                
                model.eval()
                test_pred_labels, test_original_labels = [], []
                with torch.no_grad():
                    for body_batch, rule_batch, y_batch in test_loader:
                        # Move data to the same device as the model
                        body_batch, rule_batch, y_batch = body_batch.to(self.device), rule_batch.to(self.device), y_batch.to(self.device)
                        outputs = model(body_batch, rule_batch)
                        preds = torch.argmax(outputs, dim=1)
                        test_pred_labels.extend(preds.cpu())
                        test_original_labels.extend(y_batch.cpu())

            train_pred_labels = []
            train_original_labels = []
            logging.info("Evaluating on train data...")
            with torch.no_grad():
                for body_batch, rule_batch, y_batch in train_loader:
                    # Move data to the same device as the model
                    body_batch, rule_batch, y_batch = body_batch.to(self.device), rule_batch.to(self.device), y_batch.to(self.device)

                    outputs = model(body_batch, rule_batch)
                    preds = torch.argmax(outputs, dim=1)
                    train_pred_labels.extend(preds.cpu())
                    train_original_labels.extend(y_batch.cpu())
    
                    # Move data to the same device as the model
                    
            
            evaluate_results = evaluate_result(train_labels=train_original_labels, train_pred_label=train_pred_labels,
                                               test_labels=test_original_labels, test_pred_label=test_pred_labels)

            logging.info(f'The metrics for the train parts are{evaluate_results['TRAIN METRICS']}')
            logging.info(f'The metrics for the train parts are{evaluate_results['TEST METRICS']}')

        except Exception as e:
            raise CustomException(e,sys)  
    
    def initiate_model_training(self):
        try:
            logging.info("Loading transformed training and test data...")

            train_data = load_pickle_file(self.data_transformation_artifact.train_obj_file_path)
            test_data = load_pickle_file(self.data_transformation_artifact.test_obj_file_path)
            logging.info("Pickle files loaded successfully.")

            ##Let's define the model_architecture

            data = prepare_rule_body_tensors(train_data, test_data)

            rule_train = data["rule_train"]
            body_train = data["body_train"]
            rule_test  = data["rule_test"]
            body_test  = data["body_test"]
            y_train    = data["y_train"]
            y_test     = data["y_test"]


            logging.info("Creating ViolationDataset objects...")
            vioation_train_dataset = CustomDataset(body_text=body_train,
                                                        rule_text=rule_train,
                                                        labels=y_train)
            
            vioation_test_dataset = CustomDataset(body_text=body_test,
                                                        rule_text=rule_test,
                                                        labels=y_test)
            
            logging.info("Building DataLoaders...")     
            train_loader_data = DataLoader(vioation_train_dataset, batch_size=32, shuffle=True)
            test_loader_data = DataLoader(vioation_test_dataset, batch_size=32)

            logging.info("Initializing ViolationClassifier model...")
            
            

            model = ViolationClassifier(pad_id=self.data_transformation_artifact.pad_id_token).to(self.device)

            self.train_and_evaluate(model=model, train_loader=train_loader_data, test_loader=test_loader_data)
            
            logging.info("Saving trained model...")

            torch.save(model.state_dict(),self.model_trainer_config.model_trained_file_path_second
)
            
            logging.info("Creating ModelTrainerArtifact...")
            model_trainer_artifact = ModelTrainerArtifact(train_model_artifact=self.model_trainer_config.model_trained_file_path)
            
            logging.info("Model training completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
                 
