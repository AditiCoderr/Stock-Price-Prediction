import numpy as np                          # numpy → numerical calculations
import pandas as pd                         # pandas → data handling (DataFrames)
from ta.momentum import RSIIndicator        # RSIIndicator → Relative Strength Index (trading indicator)
from ta.trend import MACD                   #  MACD → Moving Average Convergence Divergence
from ta.volatility import BollingerBands    # BollingerBands → volatility indicator


WINDOW_SIZE = 60

def add_features(df):

    df = df.copy()

    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20']= df['Daily_Return'].rolling(20).std()

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    df.dropna(inplace=True)
    return df


FEATURE_BUFFER = 250    # enough history for MA50, Bollinger, RSI etc.

def prepare_input(df, scaler, feature_cols, window=WINDOW_SIZE):
    """Scale and window — expects df already has all feature columns computed."""
    # Drop only rows where feature_cols have NaN
    df_clean = df.dropna(subset=feature_cols)

    if len(df_clean) == 0:
        raise ValueError("No valid rows after dropna — not enough history.")

    data       = df_clean[feature_cols].values
    data_scaled = scaler.transform(data)

    # Pad if still under window size
    if len(data_scaled) < window:
        pad_size    = window - len(data_scaled)
        padding     = np.tile(data_scaled[0], (pad_size, 1))
        data_scaled = np.vstack([padding, data_scaled])

    X = data_scaled[-window:]
    return X.reshape(1, window, len(feature_cols))


def predict_next_n_days(model, df, scaler, close_scaler,
                        feature_cols, n_days=30):
    """
    Autoregressively predict next n_days prices.
    Keeps a rolling buffer of FEATURE_BUFFER rows so rolling
    windows never run out of history.
    """
    predictions = []

    # Start with last FEATURE_BUFFER rows of raw OHLCV only
    df_buffer = df[['Open','High','Low','Close','Volume']].copy()
    df_buffer = df_buffer.tail(FEATURE_BUFFER).copy()

    for day in range(n_days):
        # Recompute all features on the current buffer
        df_feat = add_features(df_buffer.copy())

        if len(df_feat.dropna(subset=feature_cols)) == 0:
            print(f"Warning: ran out of history at day {day}, stopping.")
            break

        # Predict
        X           = prepare_input(df_feat, scaler, feature_cols)
        pred_scaled = model.predict(X, verbose=0)
        pred_price  = float(close_scaler.inverse_transform(
                          pred_scaled.reshape(-1, 1))[0][0])
        predictions.append(pred_price)

        # Append predicted row to buffer as next business day
        last_date = df_buffer.index[-1]
        next_date = pd.bdate_range(last_date, periods=2)[1]

        new_row = pd.DataFrame(
            [[pred_price, pred_price, pred_price,
              pred_price, float(df_buffer['Volume'].mean())]],
            columns=['Open','High','Low','Close','Volume'],
            index=[next_date]
        )

        # Append and keep only last FEATURE_BUFFER rows
        df_buffer = pd.concat([df_buffer, new_row])
        df_buffer = df_buffer.tail(FEATURE_BUFFER)

    return predictions

def compute_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mae = np.mean(np.abs(actual-predicted))
    mape = np.mean(np.abs((actual-predicted)/ actual))*100
    return round(rmse, 2), round(mae, 2), round(mape, 2)