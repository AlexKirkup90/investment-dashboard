# backend.py
# ==============================================================================
# V40.2 - ULTIMATE MODEL LIVE ENGINE (BULLETPROOF DATA FETCHING)
# ==============================================================================
import os
import json
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import ta
import joblib
from requests import Session
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
import warnings
import streamlit as st
from twelvedata import TDClient

warnings.filterwarnings('ignore')

# ——— Setup ———
session = Session()
session.headers['User-agent'] = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/58.0.3029.110 Safari/537.3'
)
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUND_CACHE_FILE = os.path.join(CACHE_DIR, 'fundamentals_ultimate.json')

# ——— Load Twelve Data client ———
TD = TDClient(apikey=st.secrets["TWELVE_DATA_API_KEY"])

# ——— Fundamentals caching ———
def load_fundamentals_cache():
    if os.path.exists(FUND_CACHE_FILE):
        return json.load(open(FUND_CACHE_FILE))
    return {}

def save_fundamentals_cache(cache):
    json.dump(cache, open(FUND_CACHE_FILE, 'w'), indent=4)

def get_fundamentals(ticker, cache, st_status=None):
    if ticker in cache:
        return cache[ticker]
    try:
        if st_status:
            st_status.text(f"Fetching fundamentals for {ticker}...")
        info = yf.Ticker(ticker, session=session).info
        fcf = info.get('freeCashflow', 0)
        mkt = info.get('marketCap', 0)
        fy = fcf/mkt if mkt else np.nan
        fund = {
            'PE': info.get('trailingPE'),
            'PS': info.get('priceToSalesTrailing12Months'),
            'DividendYield': info.get('dividendYield'),
            'DebtToEquity': info.get('debtToEquity'),
            'GrossMargin': info.get('grossMargins'),
            'ROE': info.get('returnOnEquity'),
            'EVEBITDA': info.get('enterpriseToEbitda'),
            'OperatingMargin': info.get('operatingMargins'),
            'ROA': info.get('returnOnAssets'),
            'FCFYield': fy
        }
        cache[ticker] = fund
        time.sleep(0.2)
        return fund
    except:
        nan_f = dict.fromkeys([
            'PE','PS','DividendYield','DebtToEquity',
            'GrossMargin','ROE','EVEBITDA','OperatingMargin',
            'ROA','FCFYield'
        ], np.nan)
        cache[ticker] = nan_f
        return nan_f

# ——— Constituents ———
@memory.cache
def fetch_sp500_constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(requests.get(url).text)[0]
    tickers = [t.replace('.', '-') for t in df['Symbol']]
    sector_map = dict(zip(tickers, df['GICS Sector']))
    return tickers, sector_map

# ——— Market Data with Twelve Data fallback ———
@memory.cache
def fetch_market_data(tickers, start, end, st_status=None):
    all_tickers = tickers + ['SPY', '^VIX']
    chunk, all_data, successful = 20, [], set()

    # 1) Chunked Yahoo Finance
    if st_status:
        st_status.text(f"Downloading {len(all_tickers)} tickers via YFinance…")
    for i in range(0, len(all_tickers), chunk):
        block = all_tickers[i:i+chunk]
        try:
            df = yf.download(
                block, start=start, end=end,
                auto_adjust=True, timeout=30,
                threads=False, session=session
            )
            if not df.empty:
                all_data.append(df)
                succ = df.columns.get_level_values(1).unique().tolist()
                successful.update(succ)
        except Exception:
            pass
        time.sleep(5)

    # 2) Identify missing tickers
    missing = [t for t in all_tickers if t not in successful]

    # 3) Twelve Data fallback for any missing equities (batch ≤120)  [oai_citation:2‡PythonFix.com](https://pythonfix.com/pkg/t/twelvedata/?utm_source=chatgpt.com)
    if missing:
        batches = [missing[i:i+120] for i in range(0, len(missing), 120)]
        for idx, batch in enumerate(batches, 1):
            if st_status:
                st_status.text(f"Twelve Data batch {idx}/{len(batches)}…")
            sym_list = ",".join(batch)
            resp = TD.time_series(
                symbol=sym_list,
                interval="1day",
                start_date=start,
                end_date=end,
                outputsize=5000
            )
            df_td = resp.as_pandas()           # index=(symbol,datetime)
            df_mc = df_td.unstack(level=0)     # columns=(field,symbol)
            # unify column names to Title case
            df_mc.columns = pd.MultiIndex.from_tuples([
                (lvl0.title(), lvl1) for lvl0,lvl1 in df_mc.columns
            ])
            all_data.append(df_mc)
            time.sleep(60/8)  # stay under 8 calls/min  [oai_citation:3‡support.twelvedata.com](https://support.twelvedata.com/en/articles/5194820-api-credits-limits?utm_source=chatgpt.com)

    # 4) Combine & clean
    if not all_data:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = pd.concat(all_data, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    prices = raw['Close'].dropna(how='all', axis=1)
    highs  = raw['High'].dropna(how='all', axis=1)
    lows   = raw['Low'].dropna(how='all', axis=1)
    return prices, highs, lows

# --- Feature Engineering Functions ---
def engineer_features_high_return(prices, monthly_prices, monthly_returns,
                                  monthly_spy_returns, sector_mom_1m,
                                  sector_map, fundamentals_cache, st_status=None):
    features = {}
    tickers = [t for t in prices.columns if t not in ['SPY','^VIX']]
    poly = PolynomialFeatures(degree=2, include_bias=False)

    for i, ticker in enumerate(tickers):
        if st_status:
            st_status.text(f"High-Return features for {ticker} ({i+1}/{len(tickers)})…")
        try:
            ret = monthly_returns[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M6'] = ret.rolling(6).mean().shift(1)
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['CompositeMom'] = 0.5*df['M6'] + 0.5*df['M12']
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            df['Beta'] = (ret.rolling(12).cov(monthly_spy_returns).shift(1) /
                          monthly_spy_returns.rolling(12).var().shift(1))
            sector = sector_map.get(ticker)
            df['SectorMom_1M'] = sector_mom_1m.get(sector, 0).shift(1) if sector else 0
            fund = get_fundamentals(ticker, fundamentals_cache, st_status)
            for k, v in fund.items():
                df[k] = v
            df['Mom_x_Vol']    = df['CompositeMom'] / (df['Vol3'] + 1e-9)
            df['Val_x_VIX']    = df['PE'] * df['VIX']
            df['SecMom_x_VIX'] = df['SectorMom_1M'] * df['VIX']
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)

            base_cols = ['CompositeMom','Vol3','Beta','VIX']
            X_poly = poly.fit_transform(df[base_cols])
            names = poly.get_feature_names_out(base_cols)
            df_poly = pd.DataFrame(
                X_poly[:,len(base_cols):],
                index=df.index, columns=names[len(base_cols):]
            )
            df = pd.concat([df, df_poly], axis=1)
            df['Target'] = ret.shift(-1)
            df.replace([np.inf,-np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if not df.empty:
                features[ticker] = df
        except Exception:
            continue
    return features

def engineer_features_low_risk(prices, highs, monthly_prices,
                               monthly_returns, monthly_spy_returns,
                               sector_mom_1m, sector_map,
                               fundamentals_cache, st_status=None):
    rolling_200d = prices.rolling(200).mean()
    breadth = (prices > rolling_200d).sum(axis=1)
    features = {}
    tickers = [t for t in prices.columns if t not in ['SPY','^VIX']]
    poly = PolynomialFeatures(degree=2, include_bias=False)

    for i, ticker in enumerate(tickers):
        if st_status:
            st_status.text(f"Low-Risk features for {ticker} ({i+1}/{len(tickers)})…")
        try:
            ret = monthly_returns[ticker]
            df = pd.DataFrame(index=monthly_returns.index)
            df['M6'] = ret.rolling(6).mean().shift(1)
            df['M12'] = ret.rolling(12).mean().shift(1)
            df['CompositeMom'] = 0.5*df['M6'] + 0.5*df['M12']
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['VIX'] = monthly_prices['^VIX'].shift(1)
            df['Beta'] = (ret.rolling(12).cov(monthly_spy_returns).shift(1) /
                          monthly_spy_returns.rolling(12).var().shift(1))
            sector = sector_map.get(ticker)
            df['SectorMom_1M'] = sector_mom_1m.get(sector, 0).shift(1) if sector else 0
            fund = get_fundamentals(ticker, fundamentals_cache, st_status)
            for k, v in fund.items():
                df[k] = v
            df['Mom_x_Vol']    = df['CompositeMom'] / (df['Vol3'] + 1e-9)
            df['Val_x_VIX']    = df['PE'] * df['VIX']
            df['SecMom_x_VIX'] = df['SectorMom_1M'] * df['VIX']
            df['RSI_14D']      = ta.momentum.rsi(prices[ticker], window=14)\
                                     .resample('M').last().shift(1)
            df['RSI_Deviation'] = df['RSI_14D'] - df['RSI_14D'].rolling(12).mean().shift(1)
            high_52w = highs[ticker].rolling(252).max()
            df['52W_High_Pct'] = (monthly_prices[ticker] /
                                  high_52w.resample('M').last()).shift(1)
            df['Return_Spread_6M'] = (
                ret.rolling(6).sum() - monthly_spy_returns.rolling(6).sum()
            ).shift(1)
            df['Corr_SPY_3M'] = ret.rolling(3).corr(monthly_spy_returns).shift(1)
            df['Market_Breadth_200D'] = breadth.resample('M').last().shift(1)

            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)

            cols = ['CompositeMom','Vol3','Beta','VIX','RSI_14D','RSI_Deviation']
            X_poly = poly.fit_transform(df[cols])
            names = poly.get_feature_names_out(cols)
            df_poly = pd.DataFrame(
                X_poly[:,len(cols):],
                index=df.index, columns=names[len(cols):]
            )
            df = pd.concat([df, df_poly], axis=1)
            df['Target'] = ret.shift(-1) / (df['Vol3'] + 1e-9)
            df.replace([np.inf,-np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if not df.empty:
                features[ticker] = df
        except Exception:
            continue
    return features

# --- Live Prediction Pipeline ---
def run_live_prediction_pipeline(st_status):
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    START_DATE = (datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')

    st_status.text("Fetching S&P 500 constituents…")
    tickers, sector_map = fetch_sp500_constituents()

    prices, highs, lows = fetch_market_data(tickers, START_DATE, END_DATE, st_status)
    fundamentals_cache = load_fundamentals_cache()

    monthly_prices      = prices.resample('M').last()
    monthly_returns     = monthly_prices.pct_change()
    monthly_spy_returns = monthly_prices['SPY'].pct_change()
    sector_monthly      = monthly_returns.groupby(pd.Series(sector_map), axis=1).mean()
    sector_mom_1m       = sector_monthly.rolling(1).mean()

    features_hr = engineer_features_high_return(
        prices, monthly_prices, monthly_returns,
        monthly_spy_returns, sector_mom_1m,
        sector_map, fundamentals_cache, st_status
    )
    features_lr = engineer_features_low_risk(
        prices, highs, monthly_prices,
        monthly_returns, monthly_spy_returns,
        sector_mom_1m, sector_map, fundamentals_cache, st_status
    )

    save_fundamentals_cache(fundamentals_cache)

    if not features_hr or not features_lr:
        st_status.warning("Could not generate features for one or both models.")
        return pd.DataFrame()

    # Dynamic regime check
    spy_50ma   = prices['SPY'].rolling(50).mean().iloc[-1]
    current_spy = prices['SPY'].iloc[-1]
    current_vix = prices['^VIX'].iloc[-1]
    if current_spy < spy_50ma or current_vix > 35:
        st_status.warning(
            "Market regime is unfavorable "
            "(SPY below 50-day MA or VIX > 35). Recommending cash."
        )
        return pd.DataFrame()

    # Train & Predict High-Return
    st_status.text("Training High-Return model…")
    X_hr, y_hr, Xp_hr, tkr_hr = [], [], [], []
    for tk, df in features_hr.items():
        X_hr.append(df.drop(columns='Target').values)
        y_hr.append(df['Target'].values)
        Xp_hr.append(df.drop(columns='Target').iloc[-1].values)
        tkr_hr.append(tk)
    if X_hr:
        X_hr  = np.vstack(X_hr)
        y_hr  = np.hstack(y_hr)
        Xp_hr = np.vstack(Xp_hr)
        scaler_hr = StandardScaler().fit(X_hr)
        model_hr  = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=120, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, gamma=0.1,
            min_child_weight=1
        ).fit(scaler_hr.transform(X_hr), y_hr)
        p50_hr = pd.Series(
            model_hr.predict(scaler_hr.transform(Xp_hr)),
            index=tkr_hr
        )
    else:
        p50_hr = pd.Series(dtype=float)

    # Train & Predict Low-Risk
    st_status.text("Training Low-Risk model…")
    X_lr, y_lr, Xp_lr, tkr_lr, vols = [], [], [], [], []
    for tk, df in features_lr.items():
        X_lr.append(df.drop(columns='Target').values)
        y_lr.append(df['Target'].values)
        Xp_lr.append(df.drop(columns='Target').iloc[-1].values)
        vols.append(df['Vol3'].iloc[-1])
        tkr_lr.append(tk)
    if X_lr:
        X_lr  = np.vstack(X_lr)
        y_lr  = np.hstack(y_lr)
        Xp_lr = np.vstack(Xp_lr)
        scaler_lr = StandardScaler().fit(X_lr)
        model_lr  = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=120, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, gamma=0.1,
            min_child_weight=1
        ).fit(scaler_lr.transform(X_lr), y_lr)
        p50_lr = pd.Series(
            model_lr.predict(scaler_lr.transform(Xp_lr)) * np.array(vols),
            index=tkr_lr
        )
    else:
        p50_lr = pd.Series(dtype=float)

    # Ensemble & Portfolio
    st_status.text("Combining predictions and constructing portfolio…")
    preds = pd.DataFrame({'P50_HR': p50_hr, 'P50_LR': p50_lr}).dropna()
    preds['EnsembleScore'] = 0.65*preds['P50_HR'] + 0.35*preds['P50_LR']
    threshold = 0.005 + 0.001 * current_vix
    watch = preds[preds['EnsembleScore'] > threshold].sort_values(
        'EnsembleScore', ascending=False
    ).head(15)

    if len(watch) > 1:
        hist_vol = monthly_returns[watch.index].tail(24).std()
        inv_vol  = 1/(hist_vol + 1e-9)
        raw_w    = watch['EnsembleScore'] * inv_vol
        weights  = raw_w / raw_w.sum()
        spy_200ma= prices['SPY'].rolling(200).mean().iloc[-1]
        ma_gap   = (current_spy - spy_200ma)/spy_200ma
        leverage = 1.4 if ma_gap>0.15 else 1.3 if ma_gap>0.1 else 1.2 if ma_gap>0.05 else 1.0
        weights *= leverage
        return pd.DataFrame(weights, columns=['Weight'])

    return pd.DataFrame()
