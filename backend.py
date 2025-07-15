# ==============================================================================
# V16 - LIVE ENGINE WITH INTERACTION FEATURES
# ==============================================================================
# This is the definitive backend, upgraded with the validated interaction
# features that achieved a Sharpe Ratio > 1.0 in backtesting.
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from fredapi import Fred
from scipy.optimize import minimize
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# --- Setup Caching ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUNDAMENTALS_CACHE_FILE = os.path.join(CACHE_DIR, 'fundamentals.json')

# --- Caching Functions ---
def load_fundamentals_cache():
    if os.path.exists(FUNDAMENTALS_CACHE_FILE):
        with open(FUNDAMENTALS_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_fundamentals_cache(cache):
    with open(FUNDAMENTALS_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def get_fundamentals(ticker, cache, st_status=None):
    if ticker in cache: return cache[ticker]
    try:
        if st_status: st_status.text(f"Fetching fundamentals for {ticker}...")
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
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df_sp500 = pd.read_html(requests.get(url).text)[0]
    tickers = [t.replace('.', '-') for t in df_sp500['Symbol'].tolist()]
    sector_map = dict(zip(tickers, df_sp500['GICS Sector']))
    return tickers, sector_map

@memory.cache
def fetch_market_data(tickers, start, end):
    all_tickers = tickers + ['SPY', '^VIX']
    raw_data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, timeout=30)
    prices, highs, lows = raw_data['Close'], raw_data['High'], raw_data['Low']
    try:
        fred_api_key = st.secrets.get("FRED", {}).get("API_KEY")
        if not fred_api_key:
            raise ValueError("FRED API key not found in secrets.toml.")
        fred = Fred(api_key=fred_api_key)
        macro_data = fred.get_series_latest_release('CPIAUCSL').pct_change(12) * 100
        macro_data = pd.DataFrame(macro_data, columns=['CPI_YoY'])
    except Exception as e:
        print(f"Could not fetch FRED data. Error: {e}")
        macro_data = pd.DataFrame(columns=['CPI_YoY'])
    return prices, highs, lows, macro_data

# --- Feature Engineering ---
def engineer_features(prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status=None):
    if st_status: st_status.text("Engineering features...")
    monthly_prices, monthly_highs, monthly_lows = prices.resample('M').last(), highs.resample('M').max(), lows.resample('M').min()
    monthly_returns, monthly_spy_returns = monthly_prices.pct_change(), monthly_prices['SPY'].pct_change()
    
    ticker_to_sector = pd.Series(sector_map)
    sector_monthly_returns = monthly_returns.groupby(ticker_to_sector, axis=1).mean()
    sector_mom_1m = sector_monthly_returns.rolling(1).mean()

    features_dict = {}
    tickers = [t for t in prices.columns if t not in ['SPY', '^VIX']]
    for i, ticker in enumerate(tickers):
        if st_status: st_status.text(f"Engineering features for {ticker} ({i+1}/{len(tickers)})...")
        try:
            ret, prc = monthly_returns[ticker], monthly_prices[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            
            # --- Base Features ---
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            
            ticker_sector = sector_map.get(ticker)
            if ticker_sector in sector_mom_1m.columns:
                df['SectorMom_1M'] = sector_mom_1m[ticker_sector].shift(1)
            else:
                df['SectorMom_1M'] = 0

            fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
            df['PE'] = fundamentals['PE']
            
            if not macro_data.empty:
                df = df.join(macro_data[['CPI_YoY']].resample('M').last().shift(1))
                df.fillna(method='ffill', inplace=True)
            else:
                df['CPI_YoY'] = 0

            # --- NEW: Tier 1 Interaction Features ---
            df['Mom_x_Vol'] = df['M12'] / (df['Vol3'] + 1e-9)
            df['Val_x_VIX'] = df['PE'] * df['VIX']
            df['SecMom_x_VIX'] = df['SectorMom_1M'] * df['VIX']
            
            df['Target'] = ret.shift(-1)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if not df.empty: features_dict[ticker] = df
        except Exception: continue
    return features_dict

# --- Portfolio Optimization ---
def optimize_portfolio_weights(expected_returns, historical_returns, max_stock_weight):
    tickers, num_assets = expected_returns.index, len(expected_returns)
    cov_matrix = historical_returns.cov() * 12
    def neg_sharpe(weights):
        p_return = np.sum(expected_returns * weights) * 12
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -p_return / (p_vol + 1e-9)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, max_stock_weight) for _ in range(num_assets))
    result = minimize(neg_sharpe, [1./num_assets]*num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=expected_returns.index)

# --- Live Prediction Pipeline ---
def run_live_prediction_pipeline(st_status, max_stock_weight):
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    START_DATE = (datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')

    tickers, sector_map = fetch_sp500_constituents()
    prices, highs, lows, macro_data = fetch_market_data(tickers, START_DATE, END_DATE)
    fundamentals_cache = load_fundamentals_cache()
    features = engineer_features(prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    st_status.text("Training final model on all data...")
    X_train_list, y_train_list, X_pred_list, pred_tickers = [], [], [], []
    for ticker, df in features.items():
        X_train_list.append(df.drop(columns='Target').values)
        y_train_list.append(df['Target'].values)
        X_pred_list.append(df.drop(columns='Target').iloc[-1].values)
        pred_tickers.append(ticker)
    
    if not X_pred_list: return pd.DataFrame()

    X_train, y_train, X_pred = np.vstack(X_train_list), np.hstack(y_train_list), np.vstack(X_pred_list)
    scaler = StandardScaler().fit(X_train)
    p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_pred))
    predictions_df = pd.DataFrame({'Ticker': pred_tickers, 'P50': p50}).set_index('Ticker')
    
    st_status.text("Optimizing live portfolio...")
    watchlist = predictions_df[predictions_df['P50'] > 0.005].sort_values('P50', ascending=False).head(15)
    if len(watchlist) > 1:
        hist_ret_for_cov = prices.resample('M').last().pct_change()
        cov_end_date, cov_start_date = hist_ret_for_cov.index[-1], hist_ret_for_cov.index[-1] - pd.DateOffset(months=12)
        historical_cov_data = hist_ret_for_cov.loc[cov_start_date:cov_end_date][watchlist.index]
        if historical_cov_data.shape[0] > 10 and not historical_cov_data.isnull().values.any():
            optimal_weights = optimize_portfolio_weights(watchlist['P50'], historical_cov_data, max_stock_weight)
            return pd.DataFrame(optimal_weights, columns=['Weight'])
    return pd.DataFrame()

# --- Backtest Pipeline (for reference, not used by the live app) ---
# The full backtest logic remains in the Colab notebook for research.
# This backend is now focused solely on fast, live predictions.
