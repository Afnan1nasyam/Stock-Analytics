# utils.py
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 60

def download_stock_data(symbol, start, end):
    sequence_length = 60
    adjusted_start = (pd.to_datetime(start) - pd.Timedelta(days=sequence_length)).strftime('%Y-%m-%d')
    print(f"[INFO] Downloading data for {symbol} from {adjusted_start} to {end}")
    
    try:
        df = yf.download(symbol, start=adjusted_start, end=end)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        return df[['Close']].dropna()
    except Exception as e:
        print(f"[ERROR] Failed to download data for {symbol}: {e}")
        raise

def prepare_input(data, scaler):
    scaled = scaler.transform(data)
    X = []
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i - WINDOW_SIZE:i, 0])
    return np.array(X).reshape(-1, WINDOW_SIZE, 1)

def make_predictions(symbol, start, end):
    print(f"\n[INFO] Running prediction for {symbol} | Start: {start}, End: {end}")
    
    try:
        df = download_stock_data(symbol, start, end)
        print("[INFO] Data downloaded successfully")
        print(df.head())

        # Load model and scaler
        try:
            model = load_model(f"saved_models/{symbol}_model.h5")
            scaler = joblib.load(f"scalers/{symbol}_scaler.pkl")
            print("[INFO] Model and scaler loaded successfully")
        except FileNotFoundError as e:
            print(f"[ERROR] Model or scaler not found: {e}")
            return None, None, f"❌ Missing model or scaler for {symbol}"

        if len(df) <= WINDOW_SIZE:
            print("[ERROR] Not enough data to make predictions")
            return None, None, "❌ Not enough data to predict"

        X = prepare_input(df, scaler)
        print(f"[INFO] Input shape: {X.shape}")

        preds_scaled = model.predict(X)
        preds = scaler.inverse_transform(preds_scaled)
        actual = df.values[WINDOW_SIZE:]

        result_df = pd.DataFrame({
            'Date': df.index[WINDOW_SIZE:].strftime('%Y-%m-%d'),
            'Actual': actual.flatten(),
            'Predicted': preds.flatten()
        })

        print("[INFO] Predictions generated successfully")
        return result_df, df, None

    except Exception as e:
        print(f"[FATAL] Prediction failed: {e}")
        return None, None, f"❌ Internal Error: {str(e)}"
