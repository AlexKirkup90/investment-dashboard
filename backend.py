# ==============================================================================
# V14 - DEFINITIVE ENGINE (BACKEND) - REPAIRED
# ==============================================================================
# This version fixes all critical faults and is re-architected to be more
# memory-efficient to prevent silent crashes on Streamlit Cloud.
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
        rates2 = fred.get_series("DGS2", start, end)
        rates10 = fred.get_series("DGS10", start, end)
        yc_slope = (rates10 - rates2).dropna()
    except Exception as e:
        print(f"Could not fetch FRED data. Error: {e}")
        yc_slope = pd.Series(dtype=float)
    return prices, highs, lows, yc_slope

# --- Feature Engineering ---
def _calculate_adx(high, low, close, n=14):
    plus_dm, minus_dm = high.diff(), low.diff().mul(-1)
    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-9)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

def engineer_features(prices, highs, lows, yc_slope, sector_map, fundamentals_cache, st_status=None):
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
            ret, prc, high, low = monthly_returns[ticker], monthly_prices[ticker], monthly_highs[ticker], monthly_lows[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M1'], df['M12'] = ret.shift(1), ret.rolling(12).mean().shift(1)
            df['Vol3'], df['Beta'] = ret.rolling(3).std().shift(1), ret.rolling(12).cov(monthly_spy_returns).shift(1) / monthly_spy_returns.rolling(12).var().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            df['MACD'] = prc.ewm(span=12, adjust=False).mean() - prc.ewm(span=26, adjust=False).mean()
            df['ADX'] = _calculate_adx(high, low, prc)
            ticker_sector = sector_map.get(ticker)
            if ticker_sector in sector_mom_1m.columns:
                df['SectorMom_1M'] = sector_mom_1m[ticker_sector].shift(1)
                df['SectorRel_1M'] = df['M1'] - df['SectorMom_1M']
            else:
                df['SectorMom_1M'], df['SectorRel_1M'] = 0, 0
            fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
            df['PE'], df['PB'], df['ROE'] = fundamentals['PE'], fundamentals['PB'], fundamentals['ROE']
            if not yc_slope.empty:
                df = df.join(yc_slope.resample('M').last().shift(1).rename('YC_slope'))
                df['YC_slope'].fillna(method='ffill', inplace=True)
            else:
                df['YC_slope'] = 0
            df['Target'] = ret.shift(-1)
            df.dropna(inplace=True)
            if not df.empty: features_dict[ticker] = df
        except Exception: continue
    return features_dict

# --- Portfolio Optimization & Performance Calculation ---
def optimize_portfolio_weights(expected_returns, historical_returns, sector_map, max_stock_weight):
    tickers, num_assets = expected_returns.index, len(expected_returns)
    cov_matrix = historical_returns.cov() * 12
    sectors = list(set(sector_map.values()))
    sector_mappings = {sec: [i for i, t in enumerate(tickers) if sector_map.get(t) == sec] for sec in sectors}
    def neg_sharpe(weights):
        p_return = np.sum(expected_returns * weights) * 12
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -p_return / (p_vol + 1e-9)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    for _, indices in sector_mappings.items():
        if indices: constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: 0.30 - np.sum(w[idx])})
    bounds = tuple((0, max_stock_weight) for _ in range(num_assets))
    result = minimize(neg_sharpe, [1./num_assets]*num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=expected_returns.index)

# --- Walk-Forward Validation (Memory Efficient) ---
def run_walk_forward_validation(features_dict, prices, sector_map, validation_start_date, risk_free_rate, st_status=None):
    # FIXED: This function is now memory-efficient. It calculates returns month-by-month.
    portfolio_monthly_returns = []
    hist_ret_for_cov = prices.resample('M').last().pct_change()
    prediction_dates = pd.to_datetime(sorted([d for d in list(features_dict.values())[0].index if d >= pd.to_datetime(validation_start_date)]))
    
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
        
        if not X_test or not X_train:
            portfolio_monthly_returns.append(0)
            continue
            
        X_train, y_train, X_test = np.vstack(X_train), np.hstack(y_train), np.vstack(X_test)
        scaler = StandardScaler().fit(X_train)
        p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_test))
        
        month_preds = pd.DataFrame({'Ticker': tickers, 'P50': p50, 'ActualReturn': actuals}).set_index('Ticker')
        watchlist = month_preds[month_preds['P50'] > 0.005].sort_values('P50', ascending=False).head(15)
        
        if len(watchlist) > 1:
            cov_end_date, cov_start_date = prediction_date - pd.DateOffset(months=1), prediction_date - pd.DateOffset(months=13)
            historical_cov_data = hist_ret_for_cov.loc[cov_start_date:cov_end_date][watchlist.index]
            if historical_cov_data.shape[0] > 10 and not historical_cov_data.isnull().values.any():
                optimal_weights = optimize_portfolio_weights(watchlist['P50'], historical_cov_data, sector_map, 0.25)
                portfolio_return = np.dot(optimal_weights, watchlist['ActualReturn'])
            else: portfolio_return = 0
        else: portfolio_return = 0
        portfolio_monthly_returns.append(portfolio_return)

    portfolio_df = pd.DataFrame({'Date': prediction_dates, 'Return': portfolio_monthly_returns}).set_index('Date')
    spy_monthly_returns = prices['SPY'].resample('M').last().pct_change()
    portfolio_df['SPY_Return'] = spy_monthly_returns.reindex(portfolio_df.index)
    portfolio_df.dropna(inplace=True)
    
    portfolio_df['Cumulative'] = (1 + portfolio_df['Return']).cumprod()
    portfolio_df['SPY_Cumulative'] = (1 + portfolio_df['SPY_Return']).cumprod()
    portfolio_df['Drawdown'] = 1 - portfolio_df['Cumulative'] / portfolio_df['Cumulative'].cummax()
    
    metrics = {}
    metrics['Sharpe'] = ((portfolio_df['Return'].mean() * 12 - risk_free_rate) / (portfolio_df['Return'].std() * np.sqrt(12)))
    metrics['MaxDrawdown'] = portfolio_df['Drawdown'].max()
    metrics['AnnualReturn'] = portfolio_df['Return'].mean() * 12
    
    return portfolio_df, metrics

# --- Main Orchestration Functions ---
def run_backtest_pipeline(st_status, start_date, risk_free_rate):
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    tickers, sector_map = fetch_sp500_constituents()
    prices, highs, lows, yc_slope = fetch_market_data(tickers, '2013-01-01', END_DATE)
    fundamentals_cache = load_fundamentals_cache()
    features = engineer_features(prices, highs, lows, yc_slope, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    # The main validation function now also calculates performance
    portfolio_df, metrics = run_walk_forward_validation(features, prices, sector_map, start_date, risk_free_rate, st_status)
    
    return portfolio_df, metrics

def run_live_prediction_pipeline(st_status, selected_sectors, max_stock_weight):
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    tickers, sector_map = fetch_sp500_constituents()
    if selected_sectors:
        tickers = [t for t in tickers if sector_map.get(t) in selected_sectors]
    prices, highs, lows, yc_slope = fetch_market_data(tickers, '2013-01-01', END_DATE)
    fundamentals_cache = load_fundamentals_cache()
    features = engineer_features(prices, highs, lows, yc_slope, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    st_status.text("Training final model on all data...")
    X_train_list, y_train_list, X_pred_list, pred_tickers = [], [], [], []
    for ticker, df in features.items():
        if not selected_sectors or sector_map.get(ticker) in selected_sectors:
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
            optimal_weights = optimize_portfolio_weights(watchlist['P50'], historical_cov_data, sector_map, max_stock_weight)
            return pd.DataFrame(optimal_weights, columns=['Weight'])
    return pd.DataFrame()

def get_available_sectors():
    _, sector_map = fetch_sp500_constituents()
    return sorted(list(set(sector_map.values())))
