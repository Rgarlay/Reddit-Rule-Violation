import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam



class ViolationDataset(Dataset):
    def __init__(self, body_emb, rule_emb, labels):
        self.body_emb = body_emb
        self.rule_emb = rule_emb
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve embeddings
        body_embedding = self.body_emb.iloc[idx]
        rule_embedding = self.rule_emb.iloc[idx]

        # Handle body_embedding: ensure it's a non-empty list, otherwise use a zero tensor
        if isinstance(body_embedding, list) and len(body_embedding) > 0:
            body_tensor = torch.tensor(body_embedding, dtype=torch.float32)
        else:
            # Covers [], None, np.nan, or other non-list/empty-list types
            body_tensor = torch.zeros(768, dtype=torch.float32)

        # Handle rule_embedding: ensure it's a non-empty list, otherwise use a zero tensor
        if isinstance(rule_embedding, list) and len(rule_embedding) > 0:
            rule_tensor = torch.tensor(rule_embedding, dtype=torch.float32)
        else:
            # Covers [], None, np.nan, or other non-list/empty-list types
            rule_tensor = torch.zeros(768, dtype=torch.float32)

        label_tensor = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        return body_tensor, rule_tensor, label_tensor



class ViolationClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.2):
        super(ViolationClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim // 2, 2)

        # Define activation once
        self.relu = nn.ReLU()

    def forward(self, body_emb, rule_emb):
        x = torch.cat((body_emb, rule_emb), dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        out = self.fc_out(x)
        return out
