# ==============================================================================
# V10 - MACRO-AWARE ENGINE (BACKEND)
# ==============================================================================
# This version enhances the feature set with key macro-economic indicators
# (CPI for inflation, PMI for economic health) to make the model regime-aware.
# VERSION 10.2: Added error handling for FRED data fetching to prevent crashes.
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
import pandas_datareader.data as web
from scipy.optimize import minimize
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
    prices = yf.download(all_tickers, start=start, end=end, auto_adjust=True, timeout=30)['Close']
    
    # --- MODIFICATION START: Added error handling for FRED data ---
    try:
        macro_data = web.DataReader(['CPIAUCSL', 'ISM'], 'fred', start, end)
        macro_data['CPI_YoY'] = macro_data['CPIAUCSL'].pct_change(12) * 100
    except Exception as e:
        print(f"Could not fetch FRED data. Continuing without macro features. Error: {e}")
        # Create an empty dataframe with expected columns if the fetch fails
        macro_data = pd.DataFrame(columns=['CPI_YoY', 'ISM'])
    # --- MODIFICATION END ---
    
    return prices, macro_data

# --- Feature Engineering ---
def engineer_features(prices, macro_data, sector_map, fundamentals_cache, st_status=None):
    if st_status: st_status.text("Engineering features...")
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    monthly_spy_returns = monthly_prices['SPY'].pct_change()

    if st_status: st_status.text("Calculating sector momentum...")
    ticker_to_sector = pd.Series(sector_map)
    sector_monthly_returns = monthly_returns.groupby(ticker_to_sector, axis=1).mean()
    sector_mom_1m = sector_monthly_returns.rolling(1).mean()
    sector_mom_3m = sector_monthly_returns.rolling(3).mean()
    
    features_dict = {}
    tickers = [t for t in prices.columns if t not in ['SPY', '^VIX']]
    
    for i, ticker in enumerate(tickers):
        if st_status: st_status.text(f"Engineering features for {ticker} ({i+1}/{len(tickers)})...")
        try:
            ret = monthly_returns[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M1'] = ret.shift(1)
            df['M3'] = ret.rolling(3).mean().shift(1)
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['Beta'] = ret.rolling(12).cov(monthly_spy_returns).shift(1) / monthly_spy_returns.rolling(12).var().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            
            ticker_sector = sector_map.get(ticker)
            if ticker_sector in sector_mom_1m.columns:
                df['SectorMom_1M'] = sector_mom_1m[ticker_sector].shift(1)
                df['SectorMom_3M'] = sector_mom_3m[ticker_sector].shift(1)
                df['SectorRel_1M'] = df['M1'] - df['SectorMom_1M']
            else:
                df['SectorMom_1M'], df['SectorMom_3M'], df['SectorRel_1M'] = 0, 0, 0

            fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
            df['PE'], df['PB'], df['ROE'] = fundamentals['PE'], fundamentals['PB'], fundamentals['ROE']
            
            # --- MODIFICATION START: Check if macro data is available before joining ---
            if not macro_data.empty:
                df = df.join(macro_data[['CPI_YoY', 'ISM']].resample('M').last().shift(1))
                df.fillna(method='ffill', inplace=True)
            else:
                df['CPI_YoY'] = 0
                df['ISM'] = 0
            # --- MODIFICATION END ---

            df['Target'] = ret.shift(-1)
            df.dropna(inplace=True)
            if not df.empty: features_dict[ticker] = df
        except Exception: continue
            
    return features_dict

# --- Portfolio Optimization ---
def optimize_portfolio_weights(expected_returns, historical_returns):
    num_assets, cov_matrix = len(expected_returns), historical_returns.cov() * 12
    def neg_sharpe(weights):
        p_return = np.sum(expected_returns * weights) * 12
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -p_return / (p_vol + 1e-9)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.25) for _ in range(num_assets))
    result = minimize(neg_sharpe, [1./num_assets]*num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=expected_returns.index)

# --- Walk-Forward Validation ---
def run_walk_forward_validation(features_dict, validation_start_date, st_status=None):
    all_results = []
    prediction_dates = pd.to_datetime(sorted([d for d in list(features_dict.values())[0].index if d >= validation_start_date]))
    for i, prediction_date in enumerate(prediction_dates):
        if st_status: st_status.text(f"Running backtest for {prediction_date.strftime('%Y-%m')} ({i+1}/{len(prediction_dates)})...")
        train_end_date = prediction_date - relativedelta(months=1)
        X_train, y_train, X_test, tickers, actuals = [], [], [], [], []
        for ticker, df in features_dict.items():
            train_df = df[df.index <= train_end_date]
            if len(train_df) >= 24:
                X_train.append(train_df.drop(columns='Target').values)
                y_train.append(train_df['Target'].values)
            if prediction_date in df.index:
                test_row = df.loc[prediction_date]
                X_test.append(test_row.drop('Target').values)
                actuals.append(test_row['Target'])
                tickers.append(ticker)
        if not X_test or not X_train: continue
        X_train, y_train, X_test = np.vstack(X_train), np.hstack(y_train), np.vstack(X_test)
        scaler = StandardScaler().fit(X_train)
        p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_test))
        month_results = pd.DataFrame({'PredictionDate': prediction_date, 'Ticker': tickers, 'P50': p50, 'ActualReturn': actuals})
        all_results.append(month_results)
    return pd.concat(all_results, ignore_index=True)

# --- Main Orchestration Functions ---
def run_backtest_pipeline(st_status):
    START_DATE, END_DATE = '2013-01-01', datetime.today().strftime('%Y-%m-%d')
    VALIDATION_START_DATE = pd.to_datetime('2016-01-01')
    
    if st_status: st_status.text("Fetching S&P 500 constituents...")
    tickers, sector_map = fetch_sp500_constituents()
    
    if st_status: st_status.text("Downloading market and macro data...")
    prices, macro_data = fetch_market_data(tickers, START_DATE, END_DATE)
    
    fundamentals_cache = load_fundamentals_cache()
    features = engineer_features(prices, macro_data, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    validation_results = run_walk_forward_validation(features, VALIDATION_START_DATE, st_status)
    return validation_results, prices

def run_live_prediction_pipeline(st_status):
    START_DATE, END_DATE = '2013-01-01', datetime.today().strftime('%Y-%m-%d')

    if st_status: st_status.text("Fetching S&P 500 constituents...")
    tickers, sector_map = fetch_sp500_constituents()

    if st_status: st_status.text("Downloading market and macro data...")
    prices, macro_data = fetch_market_data(tickers, START_DATE, END_DATE)

    fundamentals_cache = load_fundamentals_cache()
    features = engineer_features(prices, macro_data, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    st_status.text("Training final model on all data...")
    X_train_list, y_train_list, X_pred_list, pred_tickers = [], [], [], []
    for ticker, df in features.items():
        X_train_list.append(df.drop(columns='Target').values)
        y_train_list.append(df['Target'].values)
        X_pred_list.append(df.drop(columns='Target').iloc[-1].values)
        pred_tickers.append(ticker)
    X_train, y_train, X_pred = np.vstack(X_train_list), np.hstack(y_train_list), np.vstack(X_pred_list)
    scaler = StandardScaler().fit(X_train)
    p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_pred))
    predictions_df = pd.DataFrame({'Ticker': pred_tickers, 'P50': p50}).set_index('Ticker')
    
    st_status.text("Optimizing live portfolio...")
    watchlist = predictions_df[predictions_df['P50'] > 0.005].sort_values('P50', ascending=False).head(15)
    if len(watchlist) > 1:
        hist_ret_for_cov = prices.resample('M').last().pct_change()
        cov_end_date = hist_ret_for_cov.index[-1]
        cov_start_date = cov_end_date - pd.DateOffset(months=12)
        historical_cov_data = hist_ret_for_cov.loc[cov_start_date:cov_end_date][watchlist.index]
        if historical_cov_data.shape[0] > 10 and not historical_cov_data.isnull().values.any():
            optimal_weights = optimize_portfolio_weights(watchlist['P50'], historical_cov_data)
            return pd.DataFrame(optimal_weights, columns=['Weight'])
            
    return pd.DataFrame()
