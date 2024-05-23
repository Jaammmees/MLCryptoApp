# trading_utils.py
import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import mplcursors

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
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
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
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df.set_index('Open Time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['Prediction'] = float('nan')  # Initialize the prediction column
    return df

def plot_data(df, axes, canvas, window_size, prediction_time, prediction_price, minutes_ahead):
    axes[0].clear()
    axes[1].clear()
    current_df = df.iloc[-window_size:]

    # Add annotation for the latest OHLCV data
    latest = current_df.iloc[-minutes_ahead-1]
    latest_prediction = current_df.iloc[-1]
    legend_text = (
        f"Open: {latest['Open']:.2f}\n"
        f"High: {latest['High']:.2f}\n"
        f"Low: {latest['Low']:.2f}\n"
        f"Close: {latest['Close']:.2f}\n"
        f"Volume: {latest['Volume']:.2f}\n"
        f"Latest Prediction : {latest_prediction['Prediction']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
    axes[0].text(0.90, 0.97, legend_text, transform=axes[0].transAxes, fontsize='small',
                 verticalalignment='top', bbox=props)

    if prediction_time is not None and prediction_price is not None:
        df.loc[prediction_time, 'Prediction'] = prediction_price

    apd = mpf.make_addplot(current_df['Prediction'], ax = axes[0], type='scatter', markersize=50, color='green')
    mpf.plot(current_df, ax=axes[0], volume=axes[1], type='candle', style='yahoo', addplot=apd)
    ymin, ymax = current_df[['Low', 'High']].min().min(), current_df[['Low', 'High']].max().max()
    axes[0].set_ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)

    canvas.draw()

def extend_future_data(df, future_minutes):
    last_time = df.index[-1]
    future_times = [last_time + datetime.timedelta(minutes=i) for i in range(1, future_minutes + 1)]
    future_data = pd.DataFrame(index=future_times, columns=df.columns)
    df = df._append(future_data)
    return df

