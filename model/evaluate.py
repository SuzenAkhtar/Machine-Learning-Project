import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from lstm import YahooLSTM
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tickers = ['AAPL', 'AMZN', 'META', 'GOOGL', 'NFLX', 'MSFT']


def load_test_data(ticker):
    X_test = np.load(f"../data/training_data/{ticker}_X_test.npy")
    y_test = np.load(f"../data/training_data/{ticker}_y_test.npy")
    return torch.tensor(X_test, dtype=torch.float32).to(device), y_test


def load_scaler(ticker):
    return joblib.load(f"../data/scalers/{ticker}_scaler.pkl")


def inverse_transform(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)


def load_best_params(ticker):
    with open(f"best_models/{ticker}_best_params.json", "r") as f:
        params = json.load(f)
    return params


def initialize_model(ticker):
    X_train = np.load(f"..\\data\\training_data\\{ticker}_X_train.npy")
    y_train = np.load(f"..\\data\\training_data\\{ticker}_y_train.npy")
    X_test = np.load(f"..\\data\\training_data\\{ticker}_X_test.npy")
    y_test = np.load(f"..\\data\\training_data\\{ticker}_y_test.npy")

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

    input_size = X_train.shape[2]
    output_size = y_train.shape[1]
    params = load_best_params(ticker)
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    dropout = params['dropout']

    model = YahooLSTM(input_size, hidden_size, num_layers,
                      output_size, dropout).to(device)
    model.eval()
    return model


def evaluate_model(ticker):
    X_test, y_test = load_test_data(ticker)
    model = initialize_model(ticker)
    scaler = load_scaler(ticker)

    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()

    predictions_rescaled = inverse_transform(predictions, scaler)
    y_test_rescaled = inverse_transform(y_test, scaler)

    columns = scaler.feature_names_in_

    close_index = np.where(columns == 'Close')[0][0]
    y_test_close = y_test_rescaled[:, close_index]
    predictions_close = predictions_rescaled[:, close_index]

    rmse = np.sqrt(mean_squared_error(y_test_close, predictions_close))
    mae = mean_absolute_error(y_test_close, predictions_close)

    print(f"{ticker} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_close, label='Actual Close Prices')
    plt.plot(predictions_close, label='Predicted Close Prices',
             linestyle='dashed')
    plt.title(f"{ticker} - Actual vs. Predicted Close Prices")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


for ticker in tickers:
    print(f"Evaluating model for {ticker}...")
    evaluate_model(ticker)
