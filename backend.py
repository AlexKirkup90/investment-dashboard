# ==============================================================================
# V40 - ULTIMATE MODEL LIVE ENGINE
# ==============================================================================
# This is the definitive backend for our Streamlit app. It contains all the
# advanced logic for the final model, including XGBoost, polynomial features,
# dynamic leverage, and a multi-layered risk management system.
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import os
import json
import ta
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# --- Setup Caching ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUNDAMENTALS_CACHE_FILE = os.path.join(CACHE_DIR, 'fundamentals_ultimate.json')

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
        fcf = info.get('freeCashflow', 0)
        mkt_cap = info.get('marketCap', 0)
        fcf_yield = fcf / mkt_cap if mkt_cap and fcf is not None else np.nan
        fundamentals = {
            'PE': info.get('trailingPE'), 'PS': info.get('priceToSalesTrailing12Months'),
            'DividendYield': info.get('dividendYield'), 'DebtToEquity': info.get('debtToEquity'),
            'GrossMargin': info.get('grossMargins'), 'ROE': info.get('returnOnEquity'),
            'EVEBITDA': info.get('enterpriseToEbitda'), 'OperatingMargin': info.get('operatingMargins'),
            'ROA': info.get('returnOnAssets'), 'FCFYield': fcf_yield
        }
        cache[ticker] = fundamentals
        time.sleep(0.2) # Pace API calls
        return fundamentals
    except Exception as e:
        print(f"Could not fetch fundamentals for {ticker}. Error: {e}")
        cache[ticker] = {
            'PE': np.nan, 'PS': np.nan, 'DividendYield': np.nan, 'DebtToEquity': np.nan,
            'GrossMargin': np.nan, 'ROE': np.nan, 'EVEBITDA': np.nan,
            'OperatingMargin': np.nan, 'ROA': np.nan, 'FCFYield': np.nan
        }
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
def fetch_market_data(tickers, start, end, st_status=None):
    all_tickers = tickers + ['SPY', '^VIX']
    all_data = []
    chunk_size = 100
    
    if st_status: st_status.text(f"Downloading market data for {len(all_tickers)} tickers...")
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        if st_status: st_status.text(f"  Downloading chunk {i//chunk_size + 1}/{(len(all_tickers)//chunk_size) + 1}...")
        try:
            data = yf.download(chunk, start=start, end=end, auto_adjust=True, timeout=30, threads=False)
            downloaded_tickers = data.columns.get_level_values(1).unique().tolist()
            failed_tickers = [t for t in chunk if t not in downloaded_tickers]
            if not data.empty: all_data.append(data)
            if failed_tickers:
                if st_status: st_status.text(f"    -> {len(failed_tickers)} tickers failed. Retrying individually...")
                for ticker in failed_tickers:
                    try:
                        single_data = yf.download(ticker, start=start, end=end, auto_adjust=True, timeout=30)
                        if not single_data.empty:
                            single_data.columns = pd.MultiIndex.from_product([single_data.columns, [ticker]])
                            all_data.append(single_data)
                        time.sleep(0.1)
                    except Exception as single_e:
                        print(f"      -> FAILED to download single ticker {ticker}. Error: {single_e}")
        except Exception as e:
            print(f"    -> An error occurred with chunk download: {e}. Skipping.")
        time.sleep(1)

    if not all_data: raise ConnectionError("Failed to download any market data.")
    raw_data = pd.concat(all_data, axis=1)
    raw_data = raw_data.loc[:,~raw_data.columns.duplicated()]
    prices, highs, lows = raw_data['Close'], raw_data['High'], raw_data['Low']
    prices, highs, lows = prices.dropna(axis=1, how='all'), highs.dropna(axis=1, how='all'), lows.dropna(axis=1, how='all')
    return prices, highs, lows

# --- Feature Engineering Functions ---
def engineer_features_high_return(prices, monthly_prices, monthly_returns, monthly_spy_returns, sector_mom_1m, sector_map, fundamentals_cache, st_status=None):
    features_dict = {}
    tickers = [t for t in prices.columns if t not in ['SPY', '^VIX']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    for i, ticker in enumerate(tickers):
        if st_status: st_status.text(f"Engineering High-Return features for {ticker} ({i+1}/{len(tickers)})...")
        try:
            ret = monthly_returns[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M6'] = ret.rolling(6).mean().shift(1)
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['CompositeMom'] = 0.5 * df['M6'] + 0.5 * df['M12']
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            df['Beta'] = ret.rolling(12).cov(monthly_spy_returns).shift(1) / monthly_spy_returns.rolling(12).var().shift(1)
            ticker_sector = sector_map.get(ticker)
            df['SectorMom_1M'] = sector_mom_1m[ticker_sector].shift(1) if ticker_sector in sector_mom_1m.columns else 0
            fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
            for key, value in fundamentals.items(): df[key] = value
            df['Mom_x_Vol'] = df['CompositeMom'] / (df['Vol3'] + 1e-9)
            df['Val_x_VIX'] = df['PE'] * df['VIX']
            df['SecMom_x_VIX'] = df['SectorMom_1M'] * df['VIX']
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            poly_cols = ['CompositeMom', 'Vol3', 'Beta', 'VIX']
            poly_feats = poly.fit_transform(df[poly_cols])
            feature_names = poly.get_feature_names_out(poly_cols)
            poly_feats_new = poly_feats[:, len(poly_cols):]
            feature_names_new = feature_names[len(poly_cols):]
            poly_df = pd.DataFrame(poly_feats_new, index=df.index, columns=feature_names_new)
            df = pd.concat([df, poly_df], axis=1)
            df['Target'] = ret.shift(-1)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if not df.empty: features_dict[ticker] = df
        except Exception: continue
    return features_dict

def engineer_features_low_risk(prices, highs, monthly_prices, monthly_returns, monthly_spy_returns, sector_mom_1m, sector_map, fundamentals_cache, st_status=None):
    rolling_200d_ma = prices.rolling(200).mean()
    stocks_above_200d_ma = (prices > rolling_200d_ma).sum(axis=1)
    features_dict = {}
    tickers = [t for t in prices.columns if t not in ['SPY', '^VIX']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    for i, ticker in enumerate(tickers):
        if st_status: st_status.text(f"Engineering Low-Risk features for {ticker} ({i+1}/{len(tickers)})...")
        try:
            ret = monthly_returns[ticker]
            daily_prices = prices[ticker]
            daily_highs = highs[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M6'] = ret.rolling(6).mean().shift(1)
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['CompositeMom'] = 0.5 * df['M6'] + 0.5 * df['M12']
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            df['Beta'] = ret.rolling(12).cov(monthly_spy_returns).shift(1) / monthly_spy_returns.rolling(12).var().shift(1)
            ticker_sector = sector_map.get(ticker)
            df['SectorMom_1M'] = sector_mom_1m[ticker_sector].shift(1) if ticker_sector in sector_mom_1m.columns else 0
            fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
            for key, value in fundamentals.items(): df[key] = value
            df['Mom_x_Vol'] = df['CompositeMom'] / (df['Vol3'] + 1e-9)
            df['Val_x_VIX'] = df['PE'] * df['VIX']
            df['SecMom_x_VIX'] = df['SectorMom_1M'] * df['VIX']
            df['RSI_14D'] = ta.momentum.rsi(daily_prices, window=14).resample('M').last().shift(1)
            df['RSI_Deviation'] = df['RSI_14D'] - df['RSI_14D'].rolling(12).mean().shift(1)
            rolling_52w_high = daily_highs.rolling(252).max()
            df['52W_High_Pct'] = (monthly_prices[ticker] / rolling_52w_high.resample('M').last()).shift(1)
            df['Return_Spread_6M'] = (ret.rolling(6).sum() - monthly_spy_returns.rolling(6).sum()).shift(1)
            df['Corr_SPY_3M'] = ret.rolling(3).corr(monthly_spy_returns).shift(1)
            df['Market_Breadth_200D'] = stocks_above_200d_ma.resample('M').last().shift(1)
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            poly_cols = ['CompositeMom', 'Vol3', 'Beta', 'VIX', 'RSI_14D', 'RSI_Deviation']
            poly_feats = poly.fit_transform(df[poly_cols])
            feature_names = poly.get_feature_names_out(poly_cols)
            poly_feats_new = poly_feats[:, len(poly_cols):]
            feature_names_new = feature_names[len(poly_cols):]
            poly_df = pd.DataFrame(poly_feats_new, index=df.index, columns=feature_names_new)
            df = pd.concat([df, poly_df], axis=1)
            df['Target'] = ret.shift(-1) / (df['Vol3'] + 1e-9) # Optimize for Sharpe
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if not df.empty: features_dict[ticker] = df
        except Exception: continue
    return features_dict

# --- Live Prediction Pipeline ---
def run_live_prediction_pipeline(st_status):
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    START_DATE = (datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')

    st_status.text("Fetching S&P 500 constituents...")
    tickers, sector_map = fetch_sp500_constituents()
    
    prices, highs, lows = fetch_market_data(tickers, START_DATE, END_DATE, st_status)
    fundamentals_cache = load_fundamentals_cache()
    
    # --- Pre-calculate shared data ---
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    monthly_spy_returns = monthly_prices['SPY'].pct_change()
    ticker_to_sector = pd.Series(sector_map)
    sector_monthly_returns = monthly_returns.groupby(ticker_to_sector, axis=1).mean()
    sector_mom_1m = sector_monthly_returns.rolling(1).mean()

    # --- Generate Features for Both Models ---
    features_hr = engineer_features_high_return(prices, monthly_prices, monthly_returns, monthly_spy_returns, sector_mom_1m, sector_map, fundamentals_cache, st_status)
    features_lr = engineer_features_low_risk(prices, highs, monthly_prices, monthly_returns, monthly_spy_returns, sector_mom_1m, sector_map, fundamentals_cache, st_status)
    
    save_fundamentals_cache(fundamentals_cache)
    
    if not features_hr or not features_lr:
        st_status.warning("Could not generate features for one or both models.")
        return pd.DataFrame()

    # --- DYNAMIC RISK MANAGEMENT CHECK ---
    st_status.text("Checking market regime filter...")
    spy_50ma = prices['SPY'].rolling(50).mean().iloc[-1]
    current_spy = prices['SPY'].iloc[-1]
    current_vix = prices['^VIX'].iloc[-1]
    if current_spy < spy_50ma or current_vix > 35:
        st_status.warning("Market regime is unfavorable (SPY below 50-day MA or VIX > 35). Recommending cash.")
        return pd.DataFrame()

    # --- Train and Predict for High-Return Model ---
    st_status.text("Training High-Return model...")
    X_train_hr, y_train_hr, X_pred_hr, tickers_hr = [], [], [], []
    for ticker, df in features_hr.items():
        X_train_hr.append(df.drop(columns='Target').values)
        y_train_hr.append(df['Target'].values)
        X_pred_hr.append(df.drop(columns='Target').iloc[-1].values)
        tickers_hr.append(ticker)
    
    p50_hr = pd.Series(np.nan, index=tickers_hr)
    if X_train_hr and X_pred_hr:
        X_train_hr, y_train_hr, X_pred_hr = np.vstack(X_train_hr), np.hstack(y_train_hr), np.vstack(X_pred_hr)
        scaler_hr = StandardScaler().fit(X_train_hr)
        model_hr = XGBRegressor(objective='reg:squarederror', n_estimators=120, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, gamma=0.1, min_child_weight=1).fit(scaler_hr.transform(X_train_hr), y_train_hr)
        p50_hr = pd.Series(model_hr.predict(scaler_hr.transform(X_pred_hr)), index=tickers_hr)

    # --- Train and Predict for Low-Risk Model ---
    st_status.text("Training Low-Risk model...")
    X_train_lr, y_train_lr, X_pred_lr, tickers_lr, vols_lr = [], [], [], [], []
    for ticker, df in features_lr.items():
        X_train_lr.append(df.drop(columns='Target').values)
        y_train_lr.append(df['Target'].values)
        X_pred_lr.append(df.drop(columns='Target').iloc[-1].values)
        vols_lr.append(df['Vol3'].iloc[-1])
        tickers_lr.append(ticker)

    p50_lr = pd.Series(np.nan, index=tickers_lr)
    if X_train_lr and X_pred_lr:
        X_train_lr, y_train_lr, X_pred_lr = np.vstack(X_train_lr), np.hstack(y_train_lr), np.vstack(X_pred_lr)
        scaler_lr = StandardScaler().fit(X_train_lr)
        model_lr = XGBRegressor(objective='reg:squarederror', n_estimators=120, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, gamma=0.1, min_child_weight=1).fit(scaler_lr.transform(X_train_lr), y_train_lr)
        p50_lr = pd.Series(model_lr.predict(scaler_lr.transform(X_pred_lr)) * np.array(vols_lr), index=tickers_lr)
        
    # --- ENSEMBLE & PORTFOLIO CONSTRUCTION ---
    st_status.text("Combining predictions and constructing portfolio...")
    month_preds = pd.DataFrame({'P50_HR': p50_hr, 'P50_LR': p50_lr})
    month_preds.dropna(inplace=True)
    month_preds['EnsembleScore'] = 0.65 * month_preds['P50_HR'] + 0.35 * month_preds['P50_LR']
    
    vix_adj_threshold = 0.005 + 0.001 * current_vix
    watchlist = month_preds[month_preds['EnsembleScore'] > vix_adj_threshold].sort_values('EnsembleScore', ascending=False).head(15)
    
    if len(watchlist) > 1:
        # --- VOLATILITY-SCALED SIZING & LEVERAGE ---
        hist_vol = monthly_returns[watchlist.index].tail(24).std()
        inv_vol = 1 / (hist_vol + 1e-9)
        raw_weights = watchlist['EnsembleScore'] * inv_vol
        final_weights = raw_weights / raw_weights.sum()
        
        spy_200ma = prices['SPY'].rolling(200).mean().iloc[-1]
        ma_gap = (current_spy - spy_200ma) / spy_200ma
        leverage = 1.4 if ma_gap > 0.15 else 1.3 if ma_gap > 0.1 else 1.2 if ma_gap > 0.05 else 1.0
        
        final_weights *= leverage
        
        return pd.DataFrame(final_weights, columns=['Weight'])
        
    return pd.DataFrame()
