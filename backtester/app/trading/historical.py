import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import EURUSD
import tensorflow as tf
import os

def create_sequences(data, sequence_length=30, prediction_steps=5):
    X, y = [], []
    data = data.dropna().reset_index(drop=True)
    for i in range(sequence_length, len(data) - prediction_steps + 1):
        X.append(data[['Close']].iloc[i-sequence_length:i].values)
        y.append(data['Close'].pct_change(prediction_steps).iloc[i + prediction_steps - 1])
    return np.array(X), np.array(y)

class LSTMStrategy(Strategy):
    n_train = 300

    def init(self):
        print("Initializing strategy...")

        self.model = tf.keras.models.load_model("C:/Users/LimJ/Documents/GitHub/MLCrypto/backtester/app/trading/lstmTest.h5")
        print("Model loaded.")

    def next(self):
        print(f"Processing next step for data at index {len(self.data)}...")
        if len(self.data) < self.n_train + 30:
            print("Not enough data to predict. Skipping...")
            return

        # Get the latest data sequence for prediction
        X_last = self.data.df['Close'][-30:].values.reshape(1, 30, 1)
        prediction = self.model.predict(X_last)[0][0]
        print(f"Prediction: {prediction}")

        # Basic trading logic based on the prediction
        if prediction > 0:
            print("Predicted an upward movement. Placing a buy order.")
            self.buy()
        elif prediction < 0:
            print("Predicted a downward movement. Placing a sell order.")
            self.sell()

data = EURUSD.copy() # Make sure to load your data properly

def run_backtest_and_save_report(data):
    bt = Backtest(data, LSTMStrategy, commission=0.0002, margin=0.05, exclusive_orders=True)
    stats = bt.run()
    report_filename = 'backtest_report.html'
    full_path = os.path.abspath(report_filename)  # Get absolute path
    bt.plot(filename=full_path)  # Save the plot to the specified absolute path
    return stats, full_path


