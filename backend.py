# ==============================================================================
# V15 - LIVE PREDICTION ENGINE (BACKEND)
# ==============================================================================
# This is a lightweight backend designed ONLY to generate a live portfolio.
# The heavy backtesting logic has been moved to the Colab Research Notebook.
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
import warnings

warnings.filterwarnings('ignore')

# --- Setup Caching ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)

# --- Data Fetching ---
@memory.cache
def fetch_sp500_constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df_sp500 = pd.read_html(requests.get(url).text)[0]
    tickers = [t.replace('.', '-') for t in df_sp500['Symbol'].tolist()]
    return tickers

@memory.cache
def fetch_market_data(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, timeout=30)['Close']
    return prices

# --- Feature Engineering ---
def engineer_live_features(prices, st_status=None):
    if st_status: st_status.text("Engineering features for live prediction...")
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    
    features_dict = {}
    tickers = prices.columns.tolist()
    
    for i, ticker in enumerate(tickers):
        if st_status: st_status.text(f"Engineering features for {ticker} ({i+1}/{len(tickers)})...")
        try:
            ret = monthly_returns[ticker]
            df = pd.DataFrame(index=monthly_returns.tail(1).index) # Only for the last month
            df['M12'] = ret.rolling(12).mean().shift(1).iloc[-1]
            df['Vol3'] = ret.rolling(3).std().shift(1).iloc[-1]
            df.dropna(inplace=True)
            if not df.empty:
                features_dict[ticker] = df
        except Exception:
            continue
    return features_dict

# --- Main Orchestration Function ---
def run_live_prediction_pipeline(st_status):
    START_DATE, END_DATE = '2013-01-01', datetime.today().strftime('%Y-%m-%d')
    
    st_status.text("Fetching S&P 500 constituents...")
    tickers = fetch_sp500_constituents()
    
    st_status.text("Downloading market data...")
    prices = fetch_market_data(tickers, START_DATE, END_DATE)
    
    features = engineer_live_features(prices, st_status)
    
    st_status.text("Training final model...")
    X_train_list, y_train_list = [], []
    X_pred_list, pred_tickers = [], []

    # Simplified historical feature generation for training
    full_monthly_returns = prices.resample('M').last().pct_change()
    for ticker, df in features.items():
        hist_ret = full_monthly_returns[ticker]
        hist_features = pd.DataFrame(index=hist_ret.index)
        hist_features['M12'] = hist_ret.rolling(12).mean().shift(1)
        hist_features['Vol3'] = hist_ret.rolling(3).std().shift(1)
        hist_features['Target'] = hist_ret.shift(-1)
        hist_features.dropna(inplace=True)
        
        if not hist_features.empty:
            X_train_list.append(hist_features.drop(columns='Target').values)
            y_train_list.append(hist_features['Target'].values)

        X_pred_list.append(df.values)
        pred_tickers.append(ticker)

    if not X_train_list or not X_pred_list:
        return pd.DataFrame()

    X_train, y_train, X_pred = np.vstack(X_train_list), np.hstack(y_train_list), np.vstack(X_pred_list)
    scaler = StandardScaler().fit(X_train)
    p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_pred))
    
    predictions_df = pd.DataFrame({'Ticker': pred_tickers, 'P50': p50}).set_index('Ticker')
    watchlist = predictions_df[predictions_df['P50'] > 0.005].sort_values('P50', ascending=False).head(15)
    
    # Simple equal weight for live portfolio
    if not watchlist.empty:
        watchlist['Weight'] = 1 / len(watchlist)
        return watchlist[['Weight']]
        
    return pd.DataFrame()
