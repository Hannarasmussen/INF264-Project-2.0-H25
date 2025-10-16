import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Any


class GiftCNN(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 15)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN:
    def __init__(self, num_epochs=20, batch_size=32, learning_rate=0.001, dropout_rate=0.25, weight_decay=1e-4, verbose=False, random_state=42):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GiftCNN(dropout_rate=self.dropout_rate).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def compute_accuracy(self, loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
    
    def fit(self, X, y):

        train_loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)

        self.train_loss_history = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            self.train_loss_history.append(epoch_loss)

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")

        return self

    def predict(self, X):
        self.model.eval()
        dummy_labels = torch.zeros(X.size(0), dtype=torch.long)
        loader = DataLoader(TensorDataset(X, dummy_labels), batch_size=self.batch_size, shuffle=False)
        preds = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
        return np.array(preds)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = GiftCNN(dropout_rate=self.dropout_rate).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return self
