import ta

def add_technical_indicators(df):
    df['MACD'] = ta.trend.macd(df['Close'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=14)
    return df