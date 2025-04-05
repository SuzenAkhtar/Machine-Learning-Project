import torch
import joblib
import numpy as np
import json
from lstm import YahooLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(ticker):
    with open(f"best_models/{ticker}_best_params.json", "r") as f:
        params = json.load(f)

    X_train = np.load(f"../data/training_data/{ticker}_X_train.npy")
    y_train = np.load(f"../data/training_data/{ticker}_y_train.npy")
    input_size = X_train.shape[2]
    output_size = y_train.shape[1]

    if params['num_layers'] == 1 and params['dropout'] > 0:
        params['dropout'] = 0

    model = YahooLSTM(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        output_size=output_size,
        dropout=params['dropout']
    ).to(device)
    model.eval()
    return model


def load_scaler(ticker):
    return joblib.load(f"../data/scalers/{ticker}_scaler.pkl")


def prepare_input_data(ticker, n_steps):
    X_train = np.load(f"../data/training_data/{ticker}_X_train.npy")
    last_n_steps = X_train[-1]
    return last_n_steps


def predict_stock_price(ticker, days_ahead=5, n_steps=60):
    model = initialize_model(ticker)
    scaler = load_scaler(ticker)

    last_n_steps = prepare_input_data(ticker, n_steps)
    num_features = last_n_steps.shape[1]

    last_n_steps_scaled = scaler.transform(last_n_steps)
    input_tensor = torch.tensor(last_n_steps_scaled.reshape(
        1, n_steps, num_features), dtype=torch.float32).to(device)

    future_predictions = []
    with torch.no_grad():
        for _ in range(days_ahead):
            predicted_price = model(input_tensor)
            predicted_value = predicted_price.cpu().numpy()[0]
            future_predictions.append(predicted_value)

            new_input = np.zeros((1, 1, num_features))
            new_input[0, 0, :len(predicted_value)] = predicted_value
            input_tensor = torch.cat((input_tensor[:, 1:, :], torch.tensor(
                new_input, dtype=torch.float32).to(device)), dim=1)

    future_predictions = np.array(future_predictions)
    padding = np.zeros(
        (future_predictions.shape[0], num_features - future_predictions.shape[1]))
    future_predictions_padded = np.hstack((future_predictions, padding))
    future_predictions_rescaled = scaler.inverse_transform(
        future_predictions_padded)

    return future_predictions_rescaled[:, 0]


# ticker = 'AMZN'
# days_ahead = 15
# predictions = predict_stock_price(ticker, days_ahead=days_ahead, n_steps=60)
# print(f"Predicted closing prices for the next {
#       days_ahead} days: {predictions}")
