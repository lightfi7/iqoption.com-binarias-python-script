import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time

# Log in to IQ Option
api = IQ_Option('Allan.traderksa@gmail.com', '%$iqualab%')
api.connect()

# Check if connected
if api.check_connect() == False:
    print("Error connecting")
    exit()

# Function to calculate EMA
def ema(data, period, column='close'):
    return data[column].ewm(span=period, adjust=False).mean()

# Function to calculate WMA
def wma(data, period, column='buffer1'):
    weights = np.arange(1, period + 1)
    return data[column].rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

# Replace 'asset' with the actual asset name, 'interval' with the candle interval in seconds, and 'amount' with the number of candles
asset = 'EURUSD'
interval = 60 # 1 minute in seconds
amount = 1000 # Number of candles

# Fetch historical candles
candles = api.get_candles(asset, interval, amount, time.time())

# Convert to DataFrame
df = pd.DataFrame(candles)

# Calculate EMAs and WMA of their difference
df['smaFast'] = ema(df, 7) # Fast EMA with a period of 7
df['smaSlow'] = ema(df, 25) # Slow EMA with a period of 25
df['buffer1'] = df['smaFast'] - df['smaSlow']
df['buffer2'] = wma(df, 5) # WMA of the difference with a period of 5

# Calculate buy and sell conditions
df['buyCondition'] = (df['buffer1'] > df['buffer2']) & (df['buffer1'].shift(1) < df['buffer2'].shift(1))
df['sellCondition'] = (df['buffer1'] < df['buffer2']) & (df['buffer1'].shift(1) > df['buffer2'].shift(1))

# Print buy and sell signals
print(df['buyCondition'].sum())
print(df['sellCondition'].sum())
# print(df[['buyCondition', 'sellCondition']])