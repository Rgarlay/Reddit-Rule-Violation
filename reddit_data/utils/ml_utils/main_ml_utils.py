import torch
from torch.utils.data import  Dataset
import torch.nn as nn
import numpy as np
from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
import torch.nn.functional as F

def prepare_rule_body_tensors(train_data, test_data):
    
    try:
        # -------- Extract columns --------
        train_rule = train_data['rule']
        train_body = train_data['body']
        y_train    = train_data['rule_violation']

        test_rule  = test_data['rule']
        test_body  = test_data['body']
        y_test     = test_data['rule_violation']

        # -------- Stack token sequences --------
        rule_train_np = np.stack(train_rule).astype(np.int64)
        body_train_np = np.stack(train_body).astype(np.int64)

        rule_test_np  = np.stack(test_rule).astype(np.int64)
        body_test_np  = np.stack(test_body).astype(np.int64)

        # -------- NumPy → Torch --------
        rule_train = torch.from_numpy(rule_train_np)
        body_train = torch.from_numpy(body_train_np)

        rule_test  = torch.from_numpy(rule_test_np)
        body_test  = torch.from_numpy(body_test_np)

        # -------- Labels --------
        y_train_np = np.asarray(y_train, dtype=np.int64)
        y_test_np  = np.asarray(y_test, dtype=np.int64)

    except Exception as e:
        raise CustomException(e,sys)

    return {
        "rule_train": rule_train,
        "body_train": body_train,
        "rule_test": rule_test,
        "body_test": body_test,
        "y_train": y_train_np,
        "y_test": y_test_np
    }



class CustomDataset(Dataset):
  def __init__(self, body_text, rule_text, labels):
    super().__init__()

    self.body_text = body_text
    self.rule_text = rule_text
    self.labels = labels

  def __len__(self):
    return  len(self.labels)

  def __getitem__(self, idx):
    return  self.body_text[idx],self.rule_text[idx],self.labels[idx]




class ViolationClassifier(nn.Module):
    def __init__(self,
                 pad_id,
                 num_embedding=10000,
                 embedding_dim=768,
                 hidden_dim=256,
                 dropout=0.2):
        super().__init__()

        # First fully connected layer

        self.embed1 = nn.Embedding(num_embeddings=num_embedding,
                                   embedding_dim=embedding_dim,
                                   padding_idx = pad_id)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim // 2, 2)  # Binary classification output

    def forward(self, rule_text, body_text):  # Concatenate embeddings

        x = torch.cat([rule_text, body_text], dim = 1)

        x = self.embed1(x)

        x = x.mean(dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        out = self.fc_out(x)
        return out

