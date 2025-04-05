import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lstm import YahooLSTM
import json
import os


class Trainer:
    def __init__(self, epochs, batch_size, device):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, epochs):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(
                    self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_loss = self.evaluate_model(model, val_loader, criterion)
            print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss /
                  len(train_loader)}, Validation Loss: {val_loss}")

        return model

    def evaluate_model(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(
                    self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def hyperparameter_tuning(self, ticker, hidden_sizes, num_layers, dropouts):
        X_train = np.load(f"..\\data\\training_data\\{ticker}_X_train.npy")
        y_train = np.load(f"..\\data\\training_data\\{ticker}_y_train.npy")
        X_test = np.load(f"..\\data\\training_data\\{ticker}_X_test.npy")
        y_test = np.load(f"..\\data\\training_data\\{ticker}_y_test.npy")

        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

        train_loader = DataLoader(TensorDataset(
            X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            X_test, y_test), batch_size=self.batch_size, shuffle=False)

        input_size = X_train.shape[2]
        output_size = y_train.shape[1]

        best_model = None
        best_val_loss = float('inf')
        best_params = None

        for hidden_size in hidden_sizes:
            for layers in num_layers:
                for dropout in dropouts:
                    print(f"Training {ticker} model with hidden_size={
                          hidden_size}, layers={layers}, dropout={dropout}...")
                    model = YahooLSTM(
                        input_size, hidden_size, layers, output_size, dropout).to(self.device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                    trained_model = self.train_model(
                        model, train_loader, val_loader, criterion, optimizer, self.epochs)

                    val_loss = self.evaluate_model(
                        trained_model, val_loader, criterion)
                    print(f"Validation Loss for {ticker}: {val_loss}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = trained_model
                        best_params = {
                            'hidden_size': hidden_size,
                            'num_layers': layers,
                            'dropout': dropout,
                            'validation_loss': best_val_loss
                        }

        print(f"Best model for {ticker} - hidden_size: {best_params['hidden_size']}, layers: {
              best_params['num_layers']}, dropout: {best_params['dropout']} with Validation Loss: {best_val_loss}")

        self.save_best_params(ticker, best_params)

    def save_best_params(self, ticker, best_params):
        with open(f'best_models/{ticker}_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
