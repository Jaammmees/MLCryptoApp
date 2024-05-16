import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from tensorflow.keras.models import load_model
import customtkinter as ctk
from tkinter import *
import threading
import joblib
import os

N_TRAIN = 200

def run_backtest(data_path, model_path, scaler_path, scaler_columns_path, progress_bar, progress_label, predictions_label, font, results_frame, cash):
    # Load data
    data = pd.read_csv(data_path, parse_dates=True, index_col='open_time')

    # Check if required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = []
    for col in required_columns:
        if col not in data.columns:
            lower_col = col.lower()
            if lower_col in data.columns:
                data.rename(columns={lower_col: col}, inplace=True)
            else:
                missing_columns.append(col)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Load pre-trained model
    model = load_model(model_path)

    # Load scaler and scaler columns
    scaler = joblib.load(scaler_path)
    scaler_columns = joblib.load(scaler_columns_path)

    # Ensure all scaler columns are present in the data
    missing_scaler_columns = [col for col in scaler_columns if col not in data.columns]
    if missing_scaler_columns:
        raise ValueError(f"Missing scaler columns in data: {missing_scaler_columns}")

    # Scale the data
    try:
        scaled_data = scaler.transform(data[scaler_columns])
    except Exception as e:
        raise ValueError(f"Error applying scaler to data: {e}")

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
            self.scaler = scaler
            self.scaler_columns = scaler_columns
            self.scaled_data = scaled_data
            
            # Generate sequences
            self.sequences, self.targets = create_sequences(self.scaled_data[:, self.scaler_columns.index('Close')], seq_length, 1)
            
            # Prepare empty, all-NaN forecast indicator
            self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

            self.total_steps = len(self.data) - N_TRAIN - seq_length
            self.current_step = 0

        def next(self):
            """
            Main trading logic pattern.

            This function is called at each step of the backtest. It performs the following tasks:
            1. Skip the training, in-sample data.
            2. Prepare variables like high, low, and close prices, and the current time.
            3. Forecast the next movement using the LSTM model.
            4. Inverse transform the scaled forecast to get the actual forecast value.
            5. Print the current prediction and the last closing price.
            6. Convert the forecast to a binary signal (1 for up, -1 for down).
            7. Update the plotted "forecast" indicator.
            8. Place long or short orders based on the forecast if not already in a position.
            9. Set aggressive stop-loss on trades that have been open for more than two days.

            For more detailed documentation on the backtesting library and trading logic, visit:
            https://kernc.github.io/backtesting.py/doc/backtesting/#gsc.tab=0
            """
            
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
            
            # Update GUI with current prediction and last closing price
            predictions_label.configure(text=f"Current Prediction: {forecast}, Last Closing Price: {close[-1]}")
            
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

            # Update progress bar
            self.current_step += 1
            progress = self.current_step / self.total_steps
            progress_bar.set(progress)
            progress_label.configure(text=f"Backtesting: {self.current_step}/{self.total_steps}")


    # Running the backtest
    bt = Backtest(data, LSTMStrategy, commission=.0002, margin=.05, cash=cash)
    results = bt.run()

    save_dataframes_to_csv(results)

    display_backtest_results(results, results_frame, font)

    bt.plot()

    # Update progress bar to 100% at the end of backtest
    progress_bar.set(1)
    progress_label.configure(text="Backtest Complete")

def save_dataframes_to_csv(results):
    """
    Save the _equity_curve and _trades DataFrames to CSV files.
    """
    output_dir = "backtest_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    equity_curve = results['_equity_curve']
    trades = results['_trades']

    equity_curve_path = os.path.join(output_dir, "equity_curve.csv")
    trades_path = os.path.join(output_dir, "trades.csv")

    equity_curve.to_csv(equity_curve_path, index=False)
    trades.to_csv(trades_path, index=False)


def start_backtest(root, path_container, progress_bar, progress_label, predictions_label, font, results_frame, cash):
    threading.Thread(
        target=run_backtest,
        args=(path_container['data_path'], path_container['model_path'], path_container['scaler_path'], path_container['columns_path'], progress_bar, progress_label, predictions_label, font, results_frame, float(cash)),
        daemon=True  # Ensures the thread will exit when the main program does
    ).start()

def display_backtest_results(results, display_frame, font):
    """
    Displays the backtest results in the provided customtkinter frame.
    """
    # Clear previous widgets from the frame
    for widget in display_frame.winfo_children():
        widget.destroy()

    # Configure grid layout for the display_frame
    display_frame.grid_rowconfigure(0, weight=1)
    display_frame.grid_columnconfigure(0, weight=1)
    display_frame.grid_columnconfigure(1, weight=3)  # Give more weight to the right frame

    # Create left and right frames
    left_frame = ctk.CTkFrame(display_frame, corner_radius=10)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    right_frame = ctk.CTkFrame(display_frame, corner_radius=10)
    right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    # Extract key-value pairs from the results for single-value keys
    single_value_keys = [key for key in results.keys() if key not in ['_equity_curve', '_trades']]
    single_value_lines = [f"{key}: {value}" for key, value in results.items() if key in single_value_keys]

    # Display the single-value results in the left frame
    for line in single_value_lines:
        label = ctk.CTkLabel(master=left_frame, text=line, text_color='#000000', font=font)
        label.pack(pady=2, padx=15, anchor="w")

    # Save the _equity_curve and _trades to CSV files
    save_dataframes_to_csv(results)

    # Display the file paths in the right frame
    equity_curve_path = "backtest_results/equity_curve.csv"
    trades_path = "backtest_results/trades.csv"

    equity_curve_label = ctk.CTkLabel(
        master=right_frame, 
        text=f"Equity Results available at : {equity_curve_path}", 
        text_color='#000000', 
        font=font
    )
    equity_curve_label.pack(pady=5, padx=10)

    trades_label = ctk.CTkLabel(
        master=right_frame, 
        text=f"Trade Results available at : {trades_path}", 
        text_color='#000000', 
        font=font
    )
    trades_label.pack(pady=5, padx=10)

    display_frame.pack(padx=15, pady=15, expand=True, fill='both')