import json
import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
from flask import Flask, request

app = Flask(__name__)

# Initialize IQ Option API
iq_api = IQ_Option('Allan.traderksa@gmail.com', '%$iqualab%')
iq_api.connect()

# Verify API connection
if not iq_api.check_connect():
    print("Error connecting")
    exit()


# Calculate Exponential Moving Average (EMA)
def calculate_ema(data, period, column='close'):
    return data[column].ewm(span=period, adjust=False).mean()


# Calculate Weighted Moving Average (WMA)
def calculate_wma(data, period, column='buffer1'):
    weights = np.arange(1, period + 1)
    return data[column].rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


# Parse win data based on direction
def parse_trade_outcome(row, direction):
    is_win = (row['open'] > row['close']) == direction
    return {
        "from": row['from'],
        "to": row['to'],
        'at': row['at'],
        'open': row['open'],
        'close': row['close'],
        'volume': row['volume'],
        'min': row['min'],
        'max': row['max'],
        "win": is_win,
    }


# Calculate win conditions for trades
def evaluate_trade_outcomes(row, df):
    upcoming_trades = df.iloc[row.name + 2:row.name + 5]
    trade_direction = row['open'] - row['close'] > 0
    outcomes = [parse_trade_outcome(upcoming_trades.iloc[i], trade_direction) for i in range(len(upcoming_trades))]
    return outcomes


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/signals')
def trading_signals():
    asset = request.args.get('asset', 'EURUSD')
    interval = int(request.args.get('interval', 86400))
    count = int(request.args.get('count', 1000))
    endtime = float(request.args.get('endtime', time.time()))

    # Fetch historical data
    candles = iq_api.get_candles(asset, interval, count, endtime)
    df = pd.DataFrame(candles)

    # Calculate EMAs and their difference's WMA
    df['fast_ema'] = calculate_ema(df, 7)
    df['slow_ema'] = calculate_ema(df, 25)
    df['ema_diff'] = df['fast_ema'] - df['slow_ema']
    df['diff_wma'] = calculate_wma(df, 5, 'ema_diff')

    # Define buy and sell conditions
    df['buy_signal'] = (df['ema_diff'] > df['diff_wma']) & (df['ema_diff'].shift(1) < df['diff_wma'].shift(1))
    df['sell_signal'] = (df['ema_diff'] < df['diff_wma']) & (df['ema_diff'].shift(1) > df['diff_wma'].shift(1))

    signals = df[df['buy_signal'] | df['sell_signal']]
    return signals.to_json(orient='records')


@app.route('/wins')
def trade_wins():
    asset = request.args.get('asset', 'EURUSD')
    interval = int(request.args.get('interval', 86400))
    count = int(request.args.get('count', 1000))
    endtime = float(request.args.get('endtime', time.time()))

    # Retrieve historical data
    candles = iq_api.get_candles(asset, interval, count, endtime)
    df = pd.DataFrame(candles)

    # Calculate EMAs and their difference's WMA
    df['fast_ema'] = calculate_ema(df, 7)
    df['slow_ema'] = calculate_ema(df, 25)
    df['ema_diff'] = df['fast_ema'] - df['slow_ema']
    df['diff_wma'] = calculate_wma(df, 5, 'ema_diff')

    # Define buy and sell signals
    df['buy_signal'] = (df['ema_diff'] > df['diff_wma']) & (df['ema_diff'].shift(1) < df['diff_wma'].shift(1))
    df['sell_signal'] = (df['ema_diff'] < df['diff_wma']) & (df['ema_diff'].shift(1) > df['diff_wma'].shift(1))

    # Initialize win column
    df['win'] = None

    # Calculate trade outcomes
    outcomes = df[df['buy_signal'] | df['sell_signal']].apply(lambda row: evaluate_trade_outcomes(row, df), axis=1)
    result = pd.DataFrame([outcome for sublist in outcomes for outcome in sublist])
    
    return result.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
