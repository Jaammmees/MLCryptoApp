import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from tensorflow.keras.models import load_model
import joblib

N_TRAIN = 200

def run_backtest(data_path, model_path, scaler_path, scaler_columns_path):
    # Load data
    data = pd.read_csv(data_path, parse_dates=True, index_col='open_time')
    
    # Load pre-trained model
    model = load_model(model_path)

    seq_length = model.input_shape[1]

    # Generate sequences for LSTM input
    def create_sequences(data, seq_length, pred_length):
        X = []
        y = []
        for i in range(len(data) - seq_length - pred_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length + pred_length])
        return np.array(X).reshape(-1, seq_length, 1), np.array(y).reshape(-1, 1)

    class LSTMStrategy(Strategy):
        price_delta = 0.004  # 0.4%

        def init(self):
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            self.scaler_columns = joblib.load(scaler_columns_path)
            # Scale the data
            self.scaled_data = self.scaler.transform(self.data.df)
            
            # Generate sequences
            self.sequences, self.targets = create_sequences(self.scaled_data[:, self.scaler_columns.index('Close')], seq_length, 1)
            
            # Prepare empty, all-NaN forecast indicator
            self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

        def next(self):
            # Skip the training, in-sample data
            if len(self.data) < N_TRAIN + seq_length:
                return
            
            # Prepare some variables
            high, low, close = self.data.High, self.data.Low, self.data.Close
            current_time = self.data.index[-1]
            
            # Forecast the next movement using the LSTM model
            X = self.sequences[-1].reshape((1, seq_length, 1))
            forecast_scaled = model.predict(X)[0, 0]

            # Prepare a row of zeros and set the forecast value for the 'Close' column
            forecast_row = np.zeros((1, len(self.scaler_columns)))
            forecast_row[0, self.scaler_columns.index('Close')] = forecast_scaled
            
            # Inverse transform the scaled forecast
            forecast = self.scaler.inverse_transform(forecast_row)[0, self.scaler_columns.index('Close')]
            
            # Print current prediction and last closing price
            print(f"Current Prediction: {forecast}, Last Closing Price: {close[-1]}")

            # Convert forecast to binary signal
            forecast = 1 if forecast > close[-1] else -1

            # Update the plotted "forecast" indicator
            self.forecasts[-1] = forecast

            # If our forecast is upwards and we don't already hold a long position
            # place a long order for 20% of available account equity. Vice versa for short.
            # Also set target take-profit and stop-loss prices to be one price_delta
            # away from the current closing price.
            upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)

            if forecast == 1 and not self.position.is_long:
                self.buy(size=.2, tp=upper, sl=lower)
            elif forecast == -1 and not self.position.is_short:
                self.sell(size=.2, tp=lower, sl=upper)

            # Additionally, set aggressive stop-loss on trades that have been open 
            # for more than two days
            for trade in self.trades:
                if current_time - trade.entry_time > pd.Timedelta('2 days'):
                    if trade.is_long:
                        trade.sl = max(trade.sl, low)
                    else:
                        trade.sl = min(trade.sl, high)

    # Running the backtest
    bt = Backtest(data, LSTMStrategy, commission=.0002, margin=.05)
    print(bt.run())
    bt.plot()

