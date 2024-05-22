# trading_utils.py
import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf

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
    return df

def plot_data(df, axes, canvas, window_size, current_index):
    if current_index < len(df):
        axes[0].clear()
        axes[1].clear()
        current_df = df.iloc[current_index - window_size:current_index]
        mpf.plot(current_df, ax=axes[0], volume=axes[1], type='candle', style='yahoo')
        canvas.draw()
        current_index += 1
    return current_index
