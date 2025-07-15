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
from xgboost import XGBRegressor
import pandas_datareader.data as web
from scipy.optimize import minimize
from joblib import Parallel, delayed
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# --- Setup Caching ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
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
    if ticker in cache:
        return cache[ticker]
    try:
        if st_status:
            st_status.text(f"Fetching fundamentals for {ticker}...")
        info = yf.Ticker(ticker).info
        fundamentals = {
            'PE': info.get('trailingPE', np.nan),
            'PB': info.get('priceToBook', np.nan),
            'ROE': info.get('returnOnEquity', np.nan)
        }
        cache[ticker] = fundamentals
        return fundamentals
    except Exception as e:
        logging.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        cache[ticker] = {'PE': np.nan, 'PB': np.nan, 'ROE': np.nan}
        return cache[ticker]

def get_fundamentals_batch(tickers, cache, st_status=None):
    for ticker in tickers:
        if ticker not in cache:
            cache[ticker] = get_fundamentals(ticker, cache, st_status)
    return cache

# --- Data Fetching ---
def fetch_sp500_constituents(st_status=None):
    if st_status:
        st_status.text("Fetching S&P 500 constituents...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df_sp500 = pd.read_html(requests.get(url).text)[0]
        tickers = [t.replace('.', '-') for t in df_sp500['Symbol'].tolist()]
        sector_map = dict(zip(tickers, df_sp500['GICS Sector']))
        return tickers, sector_map
    except Exception as e:
        logging.error(f"Failed to fetch S&P 500 constituents: {e}")
        return [], {}

def fetch_market_data(tickers, start, end, st_status=None):
    if st_status:
        st_status.text("Downloading market and macro data...")
    all_tickers = tickers + ['SPY', '^VIX']
    try:
        raw_data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, timeout=30)
        prices = raw_data['Close']
        highs = raw_data['High']
        lows = raw_data['Low']
    except Exception as e:
        logging.error(f"Failed to fetch market data: {e}")
        prices, highs, lows = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        macro_data = web.DataReader(['CPIAUCSL', 'ISM'], 'fred', start, end)
        macro_data['CPI_YoY'] = macro_data['CPIAUCSL'].pct_change(12) * 100
    except Exception as e:
        logging.error(f"Could not fetch FRED data: {e}")
        macro_data = pd.DataFrame(columns=['CPI_YoY', 'ISM'])
    return prices, highs, lows, macro_data

# --- Feature Engineering ---
def engineer_features_single(ticker, prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status=None):
    try:
        monthly_prices = prices[ticker].resample('M').last()
        monthly_highs = highs[ticker].resample('M').max()
        monthly_lows = lows[ticker].resample('M').min()
        monthly_returns = monthly_prices.pct_change()
        monthly_spy_returns = prices['SPY'].resample('M').last().pct_change()
        ticker_sector = sector_map.get(ticker)
        sector_monthly_returns = monthly_returns.groupby(pd.Series(sector_map), axis=1).mean()
        sector_mom_1m = sector_monthly_returns.rolling(1).mean()
        sector_mom_3m = sector_monthly_returns.rolling(3).mean()
        
        df = pd.DataFrame(index=monthly_returns.index)
        df['M1'] = monthly_returns.shift(1)
        df['M3'] = monthly_returns.rolling(3).mean().shift(1)
        df['M12'] = monthly_returns.rolling(12).mean().shift(1)
        df['Vol3'] = monthly_returns.rolling(3).std().shift(1)
        df['Beta'] = monthly_returns.rolling(12).cov(monthly_spy_returns).shift(1) / monthly_spy_returns.rolling(12).var().shift(1)
        df['VIX'] = prices['^VIX'].resample('M').last().shift(1)
        
        # Technical Indicators
        ema_12 = monthly_prices.ewm(span=12, adjust=False).mean()
        ema_26 = monthly_prices.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        plus_dm = monthly_highs.diff()
        minus_dm = monthly_lows.diff().mul(-1)
        plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0
        tr1 = monthly_highs - monthly_lows
        tr2 = abs(monthly_highs - monthly_prices.shift(1))
        tr3 = abs(monthly_lows - monthly_prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-9)) * 100
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        if ticker_sector in sector_mom_1m.columns:
            df['SectorMom_1M'] = sector_mom_1m[ticker_sector].shift(1)
            df['SectorMom_3M'] = sector_mom_3m[ticker_sector].shift(1)
            df['SectorRel_1M'] = df['M1'] - df['SectorMom_1M']
        else:
            df['SectorMom_1M'], df['SectorMom_3M'], df['SectorRel_1M'] = 0, 0, 0
        
        fundamentals = get_fundamentals(ticker, fundamentals_cache, st_status)
        df['PE'], df['PB'], df['ROE'] = fundamentals['PE'], fundamentals['PB'], fundamentals['ROE']
        
        if not macro_data.empty:
            df = df.join(macro_data[['CPI_YoY', 'ISM']].resample('M').last().shift(1))
            df.fillna(method='ffill', inplace=True)
        else:
            df['CPI_YoY'], df['ISM'] = 0, 0
        
        df['Target'] = monthly_returns.shift(-1)
        df.dropna(inplace=True)
        return ticker, df
    except Exception as e:
        logging.error(f"Feature engineering failed for {ticker}: {e}")
        return ticker, pd.DataFrame()

def engineer_features(prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status=None):
    if st_status:
        st_status.text("Engineering features...")
    tickers = [t for t in prices.columns if t not in ['SPY', '^VIX']]
    results = Parallel(n_jobs=-1)(
        delayed(engineer_features_single)(
            ticker, prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status
        ) for ticker in tickers
    )
    return {ticker: df for ticker, df in results if not df.empty}

# --- Portfolio Optimization ---
def optimize_portfolio_weights(expected_returns, historical_returns, sector_map, max_stock_weight=0.25):
    tickers = expected_returns.index
    num_assets = len(tickers)
    cov_matrix = historical_returns.cov() * 12
    sectors = list(set(sector_map.values()))
    sector_mappings = {sec: [i for i, t in enumerate(tickers) if sector_map.get(t) == sec] for sec in sectors}
    
    def neg_sharpe(weights):
        p_return = np.sum(expected_returns * weights) * 12
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -p_return / (p_vol + 1e-9)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    for sector, indices in sector_mappings.items():
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: 0.30 - np.sum(w[idx])})
    
    bounds = tuple((0, max_stock_weight) for _ in range(num_assets))
    result = minimize(neg_sharpe, [1./num_assets]*num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=expected_returns.index)

# --- Walk-Forward Validation ---
def run_walk_forward_validation(features_dict, validation_start_date, st_status=None):
    all_results = []
    prediction_dates = pd.to_datetime(sorted([d for d in list(features_dict.values())[0].index if d >= validation_start_date]))
    for i, prediction_date in enumerate(prediction_dates):
        if st_status:
            st_status.text(f"Running backtest for {prediction_date.strftime('%Y-%m')} ({i+1}/{len(prediction_dates)})...")
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
            continue
        X_train, y_train, X_test = np.vstack(X_train), np.hstack(y_train), np.vstack(X_test)
        scaler = StandardScaler().fit(X_train)
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        p50 = model.fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_test))
        month_results = pd.DataFrame({'PredictionDate': prediction_date, 'Ticker': tickers, 'P50': p50, 'ActualReturn': actuals})
        all_results.append(month_results)
    return pd.concat(all_results, ignore_index=True)

# --- Performance Metrics ---
def calculate_performance_metrics(results_df, prices, sector_map, max_stock_weight=0.25):
    portfolio_monthly_returns = []
    hist_ret_for_cov = prices.resample('M').last().pct_change()
    previous_weights = pd.Series(0, index=prices.columns)
    for date in results_df['PredictionDate'].unique():
        month_preds = results_df[results_df['PredictionDate'] == date].set_index('Ticker')
        watchlist = month_preds[month_preds['P50'] > np.percentile(month_preds['P50'], 90)].sort_values('P50', ascending=False).head(15)
        if len(watchlist) > 1:
            expected_returns = watchlist['P50']
            cov_end_date = date - pd.DateOffset(months=1)
            cov_start_date = cov_end_date - pd.DateOffset(months=12)
            historical_cov_data = hist_ret_for_cov.loc[cov_start_date:cov_end_date][watchlist.index]
            if historical_cov_data.shape[0] > 10 and not historical_cov_data.isnull().values.any():
                optimal_weights = optimize_portfolio_weights(expected_returns, historical_cov_data, sector_map, max_stock_weight)
                portfolio_return = np.dot(optimal_weights, watchlist['ActualReturn'])
                turnover = np.sum(np.abs(optimal_weights - previous_weights.reindex(optimal_weights.index, fill_value=0))) * 0.001
                portfolio_return -= turnover
                previous_weights = optimal_weights
            else:
                portfolio_return = 0
        else:
            portfolio_return = 0
        portfolio_monthly_returns.append(portfolio_return)
    
    portfolio_df = pd.DataFrame({
        'Date': pd.to_datetime(results_df['PredictionDate'].unique()),
        'Model Portfolio': portfolio_monthly_returns
    }).set_index('Date')
    spy_monthly_returns = prices['SPY'].resample('M').last().pct_change()
    portfolio_df['SPY Benchmark'] = spy_monthly_returns.reindex(portfolio_df.index)
    portfolio_df['Cumulative'] = (1 + portfolio_df['Model Portfolio']).cumprod()
    portfolio_df.dropna(inplace=True)
    
    risk_free_rate = 0.04
    model_sharpe = ((portfolio_df['Model Portfolio'].mean() * 12 - risk_free_rate) /
                    (portfolio_df['Model Portfolio'].std() * np.sqrt(12)))
    spy_sharpe = ((portfolio_df['SPY Benchmark'].mean() * 12 - risk_free_rate) /
                  (portfolio_df['SPY Benchmark'].std() * np.sqrt(12)))
    max_drawdown = (1 - portfolio_df['Cumulative'] / portfolio_df['Cumulative'].cummax()).max()
    annual_return = portfolio_df['Model Portfolio'].mean() * 12
    
    return portfolio_df, {
        "Sharpe": model_sharpe,
        "MaxDrawdown": max_drawdown,
        "AnnualReturn": annual_return,
        "SPY_Sharpe": spy_sharpe
    }

# --- Main Orchestration Functions ---
def get_available_sectors():
    _, sector_map = fetch_sp500_constituents()
    return list(set(sector_map.values()))

def run_backtest_pipeline(st_status=None, start_date='2016-01-01'):
    START_DATE, END_DATE = '2013-01-01', datetime.today().strftime('%Y-%m-%d')
    VALIDATION_START_DATE = pd.to_datetime(start_date)
    
    logging.info(f"Starting backtest for {VALIDATION_START_DATE} to {END_DATE}")
    tickers, sector_map = fetch_sp500_constituents(st_status)
    prices, highs, lows, macro_data = fetch_market_data(tickers, START_DATE, END_DATE, st_status)
    if prices.empty:
        raise ValueError("Failed to fetch market data")
    
    fundamentals_cache = load_fundamentals_cache()
    fundamentals_cache = get_fundamentals_batch(tickers, fundamentals_cache, st_status)
    features = engineer_features(prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    validation_results = run_walk_forward_validation(features, VALIDATION_START_DATE, st_status)
    portfolio_df, metrics = calculate_performance_metrics(validation_results, prices, sector_map)
    
    return portfolio_df, metrics

def run_live_prediction_pipeline(st_status=None, selected_sectors=None, max_stock_weight=0.25):
    START_DATE, END_DATE = '2013-01-01', datetime.today().strftime('%Y-%m-%d')
    
    logging.info(f"Starting live prediction at {END_DATE}")
    tickers, sector_map = fetch_sp500_constituents(st_status)
    if selected_sectors:
        tickers = [t for t in tickers if sector_map.get(t) in selected_sectors]
        sector_map = {t: sector_map[t] for t in tickers}
    
    prices, highs, lows, macro_data = fetch_market_data(tickers, START_DATE, END_DATE, st_status)
    if prices.empty:
        raise ValueError("Failed to fetch market data")
    
    fundamentals_cache = load_fundamentals_cache()
    fundamentals_cache = get_fundamentals_batch(tickers, fundamentals_cache, st_status)
    features = engineer_features(prices, highs, lows, macro_data, sector_map, fundamentals_cache, st_status)
    save_fundamentals_cache(fundamentals_cache)
    
    st_status.text("Training final model on all data...")
    X_train_list, y_train_list, X_pred_list, pred_tickers = [], [], [], []
    for ticker, df in features.items():
        X_train_list.append(df.drop(columns='Target').values)
        y_train_list.append(df['Target'].values)
        X_pred_list.append(df.drop(columns='Target').iloc[-1].values)
        pred_tickers.append(ticker)
    
    if not X_train_list:
        return pd.DataFrame()
    
    X_train, y_train, X_pred = np.vstack(X_train_list), np.hstack(y_train_list), np.vstack(X_pred_list)
    scaler = StandardScaler().fit(X_train)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    p50 = model.fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_pred))
    
    predictions_df = pd.DataFrame({'Ticker': pred_tickers, 'P50': p50}).set_index('Ticker')
    
    st_status.text("Optimizing live portfolio...")
    watchlist = predictions_df[predictions_df['P50'] > np.percentile(predictions_df['P50'], 90)].sort_values('P50', ascending=False).head(15)
    
    if len(watchlist) > 1:
        hist_ret_for_cov = prices.resample('M').last().pct_change()
        cov_end_date = hist_ret_for_cov.index[-1]
        cov_start_date = cov_end_date - pd.DateOffset(months=12)
        historical_cov_data = hist_ret_for_cov.loc[cov_start_date:cov_end_date][watchlist.index]
        
        if historical_cov_data.shape[0] > 10 and not historical_cov_data.isnull().values.any():
            optimal_weights = optimize_portfolio_weights(watchlist['P50'], historical_cov_data, sector_map, max_stock_weight)
            final_portfolio = pd.DataFrame(optimal_weights, columns=['Weight'])
            return final_portfolio
    
    return pd.DataFrame()
