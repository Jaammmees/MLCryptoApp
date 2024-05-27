# trading_utils.py
import datetime
import requests
import pandas as pd
import mplfinance as mpf
import numpy as np
import joblib
from keras.models import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def fetch_data(hours_back=1.5, symbol='BTCUSDT', interval='1m'):
    API_URL = 'https://api.binance.com/api/v3/klines'
    now = datetime.datetime.now()
    start_time = now - datetime.timedelta(hours=hours_back)
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(now.timestamp() * 1000),
    }
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.status_code} - {response.text}")

    data = response.json()

    if isinstance(data, dict) and 'code' in data:
        raise ValueError(f"Error from API: {data['msg']}")

    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Australia/Adelaide')
    df.set_index('Open Time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['Prediction'] = float('nan')  # Initialize the prediction column
    return df

def fetch_current_minute_data(symbol='BTCUSDT', interval='1m'):
    API_URL = 'https://api.binance.com/api/v3/klines'
    now = datetime.datetime.now()
    start_time = now - datetime.timedelta(minutes=1)
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime' : int(now.timestamp() * 1000)
    }
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.status_code} - {response.text}")

    data = response.json()

    if isinstance(data, dict) and 'code' in data:
        raise ValueError(f"Error from API: {data['msg']}")

    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Australia/Adelaide')
    df.set_index('Open Time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['Prediction'] = float('nan')  # Initialize the prediction column
    return df

def plot_data(df, axes, canvas, prediction_time, prediction_price, minutes_ahead):
    df.index = pd.to_datetime(df.index)
    axes[0].clear()
    axes[1].clear()
    current_df = df

    if prediction_time is not None and prediction_price is not None:
        # Only make one prediction for that time index
        if np.isnan(df.loc[prediction_time, 'Prediction']):
            df.loc[prediction_time, 'Prediction'] = prediction_price

    most_recent_close = current_df.iloc[-minutes_ahead - 1]['Close']
    most_recent_time = current_df.index[-minutes_ahead - 1]

    # Plot candlestick chart on the first axis
    apd = mpf.make_addplot(current_df['Prediction'], ax=axes[0], type='scatter', markersize=50, color='blue')
    hlines = [most_recent_close]
    vlines = [most_recent_time]
    
    mpf.plot(
        current_df,
        ax=axes[0],
        type='candle',
        style='yahoo',
        addplot=apd,
        hlines=dict(hlines=hlines, linestyle='--', linewidths=1, alpha=0.7, colors='red'),
        vlines=dict(vlines=vlines, linestyle='--', linewidths=1, alpha=0.7, colors='blue'),
        tight_layout=True,
    )

    ymin, ymax = current_df[['Low', 'High']].min().min(), current_df[['Low', 'High']].max().max()
    axes[0].set_ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)

    canvas.draw()


def extend_future_data(df, future_minutes):
    last_time = df.index[-1]
    future_times = [last_time + datetime.timedelta(minutes=i) for i in range(1, future_minutes + 1)]
    future_data = pd.DataFrame(index=future_times, columns=df.columns)
    df = df._append(future_data)
    return df

def stop_trading(is_running):
    return False

def prepare_data_for_prediction(df, sequence_length_in, minutes_ahead, scaler, scaler_columns):
    data = df.iloc[-(sequence_length_in + minutes_ahead):-minutes_ahead].copy()
    data = data[scaler_columns]
    data_scaled = scaler.transform(data)
    data_scaled_df = pd.DataFrame(data_scaled, columns=scaler_columns)
    X = data_scaled_df.values.reshape((1, sequence_length_in, len(scaler_columns)))
    return X

def update_plot(df_full, minutes_ahead, selected_resolution, selected_crypto, labels, model, scaler, scaler_columns, canvas, ax1, ax2, is_running, root):
    if not is_running:
        return

    try:
        # Fetch new data
        new_data = fetch_current_minute_data(symbol=selected_crypto, interval=selected_resolution)
        print(new_data)
        if new_data.empty:
            print("No new data fetched.")
        else:
            last_actual_index = -minutes_ahead - 1
            new_timestamp = new_data.index[0]
            if new_timestamp > df_full.index[last_actual_index]:
                last_index = df_full.index[-1]
                next_index = last_index + pd.Timedelta(minutes=1)
                buffer_row = pd.DataFrame([[np.nan] * len(df_full.columns)], columns=df_full.columns, index=[next_index])
                df_full = pd.concat([df_full, buffer_row])
            
            df_full.loc[df_full.index[last_actual_index], ['Open', 'High', 'Low', 'Close', 'Volume']] = new_data.iloc[0]

            prediction_time = df_full.index[-1]
            prediction_price = None

            # Check if a prediction is already made for this time
            if np.isnan(df_full.loc[prediction_time, 'Prediction']):
                if model is not None and scaler is not None and scaler_columns is not None:
                    # Get the model input shape
                    _, sequence_length_in, num_features = model.input_shape
                    
                    # Prepare data for prediction
                    X = prepare_data_for_prediction(df_full, sequence_length_in - minutes_ahead, minutes_ahead, scaler, scaler_columns)
                    if len(X) > 0:
                        # Make prediction
                        prediction = model.predict(X)
                        inverse_transform_array = np.zeros((1, len(scaler_columns)))
                        inverse_transform_array[0, 0] = prediction
                        prediction_price = scaler.inverse_transform(inverse_transform_array)[0, 0]

            """
            Trading Strategy Implementation Section
            ---------------------------------------
            This is the section where you implement your trading strategy.
            Based on the model's prediction, you can decide whether to buy, sell, or hold.
            Example:
            if prediction_price > current_price + threshold:
                execute_buy_order()
            elif prediction_price < current_price - threshold:
                execute_sell_order()
            else:
                hold_position()

            Replace the following line with your trading logic.
            """
            # For testing: doesn't actually represent real prediction
            prediction_price = df_full.iloc[-minutes_ahead - 1]['Close']

            # Update plot with new data and prediction
            plot_data(df_full, [ax1, ax2], canvas, prediction_time, prediction_price, minutes_ahead)

            # Update UI labels
            labels['current_time_label'].configure(text=f"Current Time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
            labels['selected_crypto_label'].configure(text=f"Selected Crypto: {selected_crypto}")
            labels['selected_resolution_label'].configure(text=f"Selected Resolution: {selected_resolution}")

            candlestick_data = df_full.iloc[last_actual_index]
            candlestick_text = (
                f"Open: {candlestick_data['Open']}\n"
                f"High: {candlestick_data['High']}\n"
                f"Low: {candlestick_data['Low']}\n"
                f"Close: {candlestick_data['Close']}\n"
                f"Volume: {candlestick_data['Volume']}"
            )
            labels['most_recent_candlestick_label'].configure(text=f"Most Recent Candlestick:\n{candlestick_text}")

            if prediction_price is not None:
                labels['most_recent_prediction_label'].configure(text=f"Most Recent Prediction: {prediction_price:.3f}")

    except Exception as e:
        print(f"Error in update_plot: {e}")

    root.after(1000, lambda: update_plot(df_full, minutes_ahead, selected_resolution, selected_crypto, labels, model, scaler, scaler_columns, canvas, ax1, ax2, is_running, root))

def initialise_trading(frame, selected_resolution, selected_crypto, path_container, minutes_ahead, labels, is_running, root):
    # Load the model
    model = load_model(path_container['model_path'])
    # Load the scaler and columns
    scaler = joblib.load(path_container['scaler_path'])
    scaler_columns = joblib.load(path_container['columns_path'])

    # Initialize the data and plot
    is_running = True
    df_full = fetch_data(symbol=selected_crypto, interval=selected_resolution)
    df_full = extend_future_data(df_full, minutes_ahead)

    fig, axes = mpf.plot(
        df_full,
        type='candle',
        style='yahoo',
        returnfig=True,
        figscale=1,
        figratio=(16, 9),
        title="Live Trading",
        scale_padding={'left': 0.1, 'right': 0.70, 'top': 0.4, 'bottom': 0.7},
    )
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    update_plot(df_full, minutes_ahead, selected_resolution, selected_crypto, labels, model, scaler, scaler_columns, canvas, axes[0], axes[1], is_running, root)

    return model, scaler, scaler_columns, df_full, fig, axes, canvas, is_running