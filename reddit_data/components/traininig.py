from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from reddit_data.entity.entity_config import ModelTrainerConfig
from reddit_data.entity.artifact_config import ModelTrainerArtifact, DataTransformationArtifact
from reddit_data.utils.main_utils.utils import load_pickle_file, save_pickle_file
from sklearn.model_selection import train_test_split


from reddit_data.utils.ml_utils.main_ml_utils import ViolationDataset, ViolationClassifier
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class ModelTraining:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)  

    def train_and_evaluate(self, model, train_loader, test_loader):
        try:
            logging.info("Starting train_and_evaluate method...")
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=0.0001)
            epochs = 25 
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for body_batch, rule_batch, y_batch in train_loader:
                # Move data to the same device as the model
                #body_batch, rule_batch, y_batch = body_batch.to(device), rule_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(body_batch, rule_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            logging.info("Evaluating on test data...")
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for body_batch, rule_batch, y_batch in test_loader:
                    # Move data to the same device as the model
#                    body_batch, rule_batch, y_batch = body_batch.to(device), rule_batch.to(device), y_batch.to(device)

                    outputs = model(body_batch, rule_batch)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
            test_acc = correct / total
            logging.info(f"Test Accuracy: {test_acc}")


            correct, total = 0, 0
            logging.info("Evaluating on train data...")
            with torch.no_grad():
                for body_batch, rule_batch, y_batch in train_loader:
                    
                    # Move data to the same device as the model
#                    body_batch, rule_batch, y_batch = body_batch.to(device), rule_batch.to(device), y_batch.to(device)

                    outputs = model(body_batch, rule_batch)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
            train_acc = correct / total

            logging.info(f"Train Accuracy: {train_acc}")

            return {'Train Accuracy': train_acc,
                       'Test Accuracy': test_acc}

        except Exception as e:
            raise CustomException(e,sys)  
    
    def initiate_model_training(self):
        try:
            logging.info("Loading transformed training and test data...")

            train_data = load_pickle_file(self.data_transformation_artifact.train_obj_file_path)
            test_data = load_pickle_file(self.data_transformation_artifact.test_obj_file_path)
            logging.info("Pickle files loaded successfully.")

            ##Let's define the model_architecture

            train_data_rule = train_data['rule']
            train_data_body = train_data['body']
            train_data_violation = train_data[['rule_violation']]

            test_data_rule = test_data['rule']
            test_data_body = test_data['body']
            test_data_violation = test_data[['rule_violation']]

            logging.info("Formatting label data...")
            train_data_violation = train_data_violation.squeeze().astype(int)
            test_data_violation = test_data_violation.squeeze().astype(int)

            logging.info("Creating ViolationDataset objects...")
            vioation_train_dataset = ViolationDataset(body_emb=train_data_body,
                                                        rule_emb=train_data_rule,
                                                        labels=train_data_violation)
            
            vioation_test_dataset = ViolationDataset(body_emb=test_data_body,
                                                        rule_emb=test_data_rule,
                                                        labels=test_data_violation)
            logging.info("Building DataLoaders...")     
            train_loader_data = DataLoader(vioation_train_dataset, batch_size=32, shuffle=True)
            test_loader_data = DataLoader(vioation_test_dataset, batch_size=32)

            logging.info("Initializing ViolationClassifier model...")
            model = ViolationClassifier(hidden_dim=256, dropout=0.5)

            train_test_metrics = self.train_and_evaluate(model=model, train_loader=train_loader_data, test_loader=test_loader_data)
            
            logging.info("Saving trained model...")
            save_pickle_file(file_to_save=model,
                             file_path=self.model_trainer_config.model_trained_file_path)
            
            logging.info("Creating ModelTrainerArtifact...")
            model_trainer_artifact = ModelTrainerArtifact(train_model_artifact=self.model_trainer_config.model_trained_file_path,
                                                          train_artifact_metric=train_test_metrics['Train Accuracy'],
                                                          test_artifact_metric=train_test_metrics['Test Accuracy'])
            
            logging.info("Model training completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
                 
