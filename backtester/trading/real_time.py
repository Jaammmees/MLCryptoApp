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
    return df

def plot_data(df, axes, canvas, window_size):
    axes[0].clear()
    axes[1].clear()
    current_df = df.iloc[-window_size:]

    # Add annotation for the latest OHLCV data
    latest = current_df.iloc[-1]
    legend_text = (
        f"Open: {latest['Open']:.2f}\n"
        f"High: {latest['High']:.2f}\n"
        f"Low: {latest['Low']:.2f}\n"
        f"Close: {latest['Close']:.2f}\n"
        f"Volume: {latest['Volume']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
    axes[0].text(0.90, 0.97, legend_text, transform=axes[0].transAxes, fontsize='small',
                 verticalalignment='top', bbox=props)

    mpf.plot(current_df, ax=axes[0], volume=axes[1], type='candle', style='yahoo')
    canvas.draw()


