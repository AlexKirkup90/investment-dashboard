# ==============================================================================
# backend.py — V12 “Ultimate Engine” Complete Backend
# ==============================================================================
# Single-pass OHLCV, vectorized MACD/ADX, AutoML-lite, confidence-weighted & turnover-aware
# optimizer with sector caps, regime features, and backwards-compatible alias.
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# LightGBM fallback
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
import pandas_datareader.data as web
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# --- Setup Caching ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUND_CACHE = os.path.join(CACHE_DIR, 'fundamentals.json')

# --- Fundamentals Cache Utilities ---
def load_fundamentals_cache():
    return json.load(open(FUND_CACHE)) if os.path.exists(FUND_CACHE) else {}

def save_fundamentals_cache(cache):
    json.dump(cache, open(FUND_CACHE, 'w'), indent=2)

def get_fundamentals(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        f = {
            'PE': info.get('trailingPE', np.nan),
            'PB': info.get('priceToBook', np.nan),
            'ROE': info.get('returnOnEquity', np.nan)
        }
    except:
        f = {'PE': np.nan, 'PB': np.nan, 'ROE': np.nan}
    cache[ticker] = f
    return f

# --- AutoML-lite Model Chooser ---
def choose_best_model(X, y):
    candidates = {
        'GBM': GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100, random_state=42)
    }
    if _HAS_LGBM:
        candidates['LGBM'] = LGBMRegressor(n_estimators=100, random_state=42)

    tscv = TimeSeriesSplit(n_splits=3)
    best_score, best_model = np.inf, None
    for name, mdl in candidates.items():
        scores = -cross_val_score(mdl, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        mean_score = scores.mean()
        if mean_score < best_score:
            best_score, best_model = mean_score, mdl
    return best_model

# --- Data Fetching (cached) ---
@memory.cache
def fetch_sp500_constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(requests.get(url).text)[0]
    tickers = [t.replace('.', '-') for t in df['Symbol']]
    sector_map = dict(zip(tickers, df['GICS Sector']))
    return tickers, sector_map

@memory.cache
def fetch_data(tickers, start, end):
    all_t = tickers + ['SPY', '^VIX']
    raw = yf.download(all_t, start=start, end=end, auto_adjust=True, timeout=30, group_by='ticker')
    closes = raw.xs('Close', axis=1, level=1)
    highs  = raw.xs('High',   axis=1, level=1)
    lows   = raw.xs('Low',    axis=1, level=1)
    try:
        macro = web.DataReader(['CPIAUCSL','ISM'], 'fred', start, end)
        macro['CPI_YoY'] = macro['CPIAUCSL'].pct_change(12)*100
    except:
        macro = pd.DataFrame()
    return closes, highs, lows, macro

# --- Feature Engineering ---
def engineer_features(closes, highs, lows, macro, sector_map, fund_cache):
    mp = closes.resample('M').last()
    mh = highs.resample('M').max()
    ml = lows.resample('M').min()
    mr = mp.pct_change().dropna()
    ms = mp['SPY'].pct_change().dropna()

    # Vectorized MACD
    ema12 = mp.ewm(span=12, adjust=False).mean()
    ema26 = mp.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()

    # Vectorized ADX
    pdm = mh.diff().clip(lower=0)
    mdm = -ml.diff().clip(upper=0)
    tr1 = mh - ml
    tr2 = (mh - mp.shift(1)).abs()
    tr3 = (ml - mp.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di  = 100 * pdm.ewm(alpha=1/14).mean().div(atr)
    minus_di = 100 * mdm.ewm(alpha=1/14).mean().div(atr)
    dx = ((plus_di-minus_di).abs().div(plus_di+minus_di+1e-9))*100
    adx = dx.ewm(alpha=1/14).mean()

    # Sector momentum
    t2s = pd.Series(sector_map)
    sec_ret = mr.groupby(t2s, axis=1).mean()
    sec_mom1 = sec_ret.rolling(1).mean()

    features = {}
    for t in [c for c in mp.columns if c not in ['SPY','^VIX']]:
        df = pd.DataFrame(index=mr.index)
        ret = mr[t]

        # Base
        df['M1']   = ret.shift(1)
        df['Vol3'] = ret.rolling(3).std().shift(1)
        df['Beta'] = ret.rolling(12).cov(ms).shift(1).div(ms.rolling(12).var())
        df['VIX']  = mp['^VIX'].shift(1)

        # Tech
        df['MACD']        = macd[t]
        df['MACD_Signal'] = macd_sig[t]
        df['ADX']         = adx[t]

        # Sector
        sec = sector_map.get(t)
        df['SecMom1'] = sec_mom1[sec].shift(1) if sec in sec_mom1 else 0
        df['SecRel1'] = df['M1'] - df['SecMom1']

        # Fundamentals
        f = get_fundamentals(t, fund_cache)
        df['PE'], df['PB'], df['ROE'] = f['PE'], f['PB'], f['ROE']

        # Macro
        if not macro.empty:
            md = macro[['CPI_YoY','ISM']].resample('M').last().shift(1)
            df = df.join(md).ffill()
        else:
            df['CPI_YoY'], df['ISM'] = 0, 0

        # Target
        df['Target'] = ret.shift(-1)
        df.dropna(inplace=True)
        if not df.empty:
            features[t] = df

    return features

# --- Optimizer with Confidence & Turnover & Sector Caps ---
def optimize_weights(exp_df, hist_ret, prev_w=None, trade_cost=0.0015, sector_map=None):
    tickers = exp_df.index.tolist()
    p50 = exp_df['P50']
    # confidence
    unc  = exp_df['P90'] - exp_df['P10']
    conf = p50.div(unc+1e-9)
    adj  = p50.mul(conf.div(conf.sum()))

    cov = hist_ret.cov()*12

    def neg_sharpe(w):
        ret = w.dot(adj)*12
        vol = np.sqrt(w.dot(cov).dot(w))
        return -ret/(vol+1e-9)

    cons = [{'type':'eq','fun':lambda w: w.sum()-1}]
    if sector_map:
        # sector indices
        sectors = {}
        for i,t in enumerate(tickers):
            s = sector_map.get(t)
            sectors.setdefault(s, []).append(i)
        for inds in sectors.values():
            cons.append({'type':'ineq','fun':lambda w,inds=inds: 0.30 - w[inds].sum()})

    bnds = [(0,0.25)]*len(tickers)
    init = np.ones(len(tickers))/len(tickers)
    res = minimize(neg_sharpe, init, method='SLSQP', bounds=bnds, constraints=cons)
    w   = pd.Series(res.x, index=tickers)

    # turnover cost
    if prev_w is not None and not prev_w.empty:
        tr   = np.abs(w - prev_w).sum()
        w   *= (1 - tr*trade_cost)

    return w

# --- Walk-Forward Backtester ---
def run_backtest(start='2013-01-01', end=None, val_start='2016-01-01'):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    tickers, sector_map = fetch_sp500_constituents()
    closes, highs, lows, macro = fetch_data(tickers, start, end)

    fund_cache = load_fundamentals_cache()
    feats      = engineer_features(closes, highs, lows, macro, sector_map, fund_cache)
    save_fundamentals_cache(fund_cache)

    results, prev_w = [], pd.Series()
    dates = sorted([d for d in next(iter(feats.values())).index if d >= pd.to_datetime(val_start)])

    for dt in dates:
        train_x, train_y, test_x, actuals, ts = [], [], [], [], []
        for t,df in feats.items():
            tr = df[df.index < dt]
            if len(tr) >= 24:
                train_x.append(tr.drop(columns='Target').values)
                train_y.append(tr['Target'].values)
            if dt in df.index:
                row = df.loc[dt]
                test_x.append(row.drop('Target').values)
                actuals.append(row['Target'])
                ts.append(t)
        if not test_x:
            continue

        Xtr = np.vstack(train_x); Ytr = np.hstack(train_y)
        Xte = np.vstack(test_x)

        mdl    = choose_best_model(Xtr, Ytr)
        scaler = StandardScaler().fit(Xtr)
        P50    = mdl.fit(scaler.transform(Xtr), Ytr).predict(scaler.transform(Xte))

        dfm = pd.DataFrame({'Ticker':ts, 'P50':P50, 'Actual':actuals}).set_index('Ticker')
        # P10/P90 from quantile GBM for bounds
        gb   = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100, random_state=42)
        p10  = gb.fit(scaler.transform(Xtr), Ytr).predict(scaler.transform(Xte))
        p90  = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100, random_state=42)\
                  .fit(scaler.transform(Xtr), Ytr).predict(scaler.transform(Xte))
        dfm['P10'], dfm['P90'] = p10, p90

        # optimize
        hist = closes.resample('M').last().pct_change().loc[:dt].tail(12)[ts]
        w    = optimize_weights(dfm, hist, prev_w, sector_map=sector_map)
        prev_w = w

        ret = w.dot(dfm['Actual'])
        results.append({'Date':dt, 'Return':ret, 'Sharpe':None, 'Weights':w})

    out = pd.DataFrame(results).set_index('Date')
    out['Cumulative'] = (1 + out['Return']).cumprod()
    return out

# alias for backwards compatibility
run_backtest_pipeline = run_backtest
