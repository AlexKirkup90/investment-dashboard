# ==============================================================================
# V6 - BACKEND ENGINE
# ==============================================================================
# This file contains all the data processing, feature engineering, and modeling
# logic. The Streamlit app will call the functions in this file to get live
# predictions.
# VERSION 6.1: Fixed __getstate__() error by separating st.status updates
# from @memory.cache decorated functions.
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pandas_datareader.data as web
import warnings

warnings.filterwarnings('ignore')

# --- Setup Caching ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUNDAMENTALS_CACHE_FILE = os.path.join(CACHE_DIR, 'fundamentals_cache.json')

# --- Caching Functions ---
def load_fundamentals_cache():
    if os.path.exists(FUNDAMENTALS_CACHE_FILE):
        with open(FUNDAMENTALS_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_fundamentals_cache(cache):
    with open(FUNDAMENTALS_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def get_fundamentals(ticker, cache, st_status):
    """Fetches fundamentals with progress updates for Streamlit."""
    if ticker in cache: return cache[ticker]
    try:
        # This function is NOT cached by joblib, so it's safe to pass st_status
        st_status.text(f"Fetching fundamentals for {ticker}...")
        info = yf.Ticker(ticker).info
        fundamentals = {'PE': info.get('trailingPE'), 'PB': info.get('priceToBook'), 'ROE': info.get('returnOnEquity')}
        cache[ticker] = fundamentals
        return fundamentals
    except Exception:
        cache[ticker] = {'PE': np.nan, 'PB': np.nan, 'ROE': np.nan}
        return cache[ticker]

# --- Data Fetching ---
@memory.cache
def fetch_sp500_constituents():
    # This function IS cached, so we DO NOT pass the st_status object here.
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df_sp500 = pd.read_html(requests.get(url).text)[0]
    tickers = [t.replace('.', '-') for t in df_sp500['Symbol'].tolist()]
    return tickers

@memory.cache
def fetch_market_data(tickers, start, end):
    # This function IS cached, so we DO NOT pass the st_status object here.
    all_tickers = tickers + ['SPY', '^VIX']
    prices = yf.download(all_tickers, start=start, end=end, auto_adjust=True, timeout=30)['Close']
    yc_slope = (web.DataReader('DGS10', 'fred', start, end)['DGS10'] - 
                web.DataReader('DGS2', 'fred', start, end)['DGS2']).dropna()
    return prices, yc_slope

# --- Feature Engineering ---
def engineer_features(prices, yc_slope, fundamentals_cache, st_status):
    """Engineers features for all historical data for model training."""
    st_status.text("Engineering features for all tickers...")
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    monthly_spy_returns = monthly_prices['SPY'].pct_change()
    
    features_dict = {}
    tickers = [t for t in prices.columns if t not in ['SPY', '^VIX']]
    
    for i, ticker in enumerate(tickers):
        st_status.text(f"Engineering features for {ticker} ({i+1}/{len(tickers)})...")
        try:
            ret, prc = monthly_returns[ticker], monthly_prices[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M1'] = ret.shift(1)
            df['M3'] = ret.rolling(3).mean().shift(1)
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['Beta'] = ret.rolling(12).cov(monthly_spy_returns).shift(1) / monthly_spy_returns.rolling(12).var().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            df['YC_slope'] = yc_slope.resample('M').last().shift(1)
            
            fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
            df['PE'], df['PB'], df['ROE'] = fundamentals['PE'], fundamentals['PB'], fundamentals['ROE']
            
            df['Target'] = ret.shift(-1)
            df.dropna(inplace=True)
            if not df.empty:
                features_dict[ticker] = df
        except Exception:
            continue
            
    return features_dict

# --- Live Prediction Model ---
def generate_live_predictions(features_dict, st_status):
    """Trains one final model on all data and predicts for the next month."""
    st_status.text("Training final models...")
    
    X_train_list, y_train_list = [], []
    X_pred_list, pred_tickers = [], []

    for ticker, df in features_dict.items():
        X_train_list.append(df.drop(columns='Target').values)
        y_train_list.append(df['Target'].values)
        X_pred_list.append(df.drop(columns='Target').iloc[-1].values)
        pred_tickers.append(ticker)
    
    if not X_train_list or not X_pred_list:
        return pd.DataFrame()

    X_train, y_train = np.vstack(X_train_list), np.hstack(y_train_list)
    X_pred = np.vstack(X_pred_list)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_pred_s = scaler.transform(X_train), scaler.transform(X_pred)

    st_status.text("Generating quantile predictions...")
    p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(X_train_s, y_train).predict(X_pred_s)
    
    predictions_df = pd.DataFrame({
        'Ticker': pred_tickers,
        'Predicted_Return_P50': p50,
    }).sort_values('Predicted_Return_P50', ascending=False).reset_index(drop=True)
    
    return predictions_df

# --- Main Orchestration Function ---
def run_prediction_pipeline(st_status):
    """The main function called by the Streamlit app to run the pipeline."""
    START_DATE, END_DATE = '2013-01-01', datetime.today().strftime('%Y-%m-%d')
    
    # Update status outside of the cached functions
    st_status.text("Fetching S&P 500 constituents...")
    tickers = fetch_sp500_constituents()
    
    st_status.text("Downloading market data...")
    prices, yc_slope = fetch_market_data(tickers, START_DATE, END_DATE)
    
    fundamentals_cache = load_fundamentals_cache()
    
    features = engineer_features(prices, yc_slope, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    predictions = generate_live_predictions(features, st_status)
    
    return predictions

