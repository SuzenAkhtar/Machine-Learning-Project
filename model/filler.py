import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from predict_stock_price import predict_stock_price
import joblib

tickers = ['AAPL', 'AMZN', 'META', 'GOOGL', 'NFLX', 'MSFT']
today_date = datetime.now().date()

for ticker in tickers:
    historical_data = pd.read_csv(
        f'../data/preprocessed/{ticker}_preprocessed_data.csv', parse_dates=['Date'])
    last_date = historical_data['Date'].max()
    gap_days = (today_date - last_date.date()).days

    if gap_days > 0:
        predicted_prices = predict_stock_price(
            ticker, days_ahead=gap_days, n_steps=60)

        scaler = joblib.load(f"../data/scalers/{ticker}_scaler.pkl")

        num_features = scaler.n_features_in_
        predicted_prices_full = np.zeros((gap_days, num_features))
        predicted_prices_full[:, 3] = predicted_prices

        predicted_prices_scaled = scaler.transform(predicted_prices_full)

        scaled_close_prices = predicted_prices_scaled[:, 0]

        prediction_dates = [last_date +
                            timedelta(days=i) for i in range(1, gap_days + 1)]

        predicted_data = pd.DataFrame({
            'Date': prediction_dates,
            'Close': scaled_close_prices
        })

        predicted_data['Open'] = np.nan
        predicted_data['High'] = np.nan
        predicted_data['Low'] = np.nan
        predicted_data['Adj Close'] = np.nan
        predicted_data['Volume'] = np.nan

        updated_data = pd.concat(
            [historical_data, predicted_data], ignore_index=True)

        updated_data.to_csv(f'updated_data/{ticker}_updated.csv', index=False)
