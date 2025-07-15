# backend.py
"""
Backend logic for the Quantitative Model Dashboard.

Provides:
- Data fetching (S&P 500, market OHLC, VIX, yield curve via FRED)
- Feature engineering (technicals, sector momentum, fundamentals)
- Walk‐forward backtest engine
- Live portfolio construction engine
- Portfolio optimization with sector and weight constraints
"""

import os
import json
import logging
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import streamlit as st
from fredapi import Fred
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from joblib import Memory, Parallel, delayed

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(filename='backend.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Setup disk caching for yfinance/FRED calls
CACHE_DIR = "backend_cache"
memory = Memory(CACHE_DIR, verbose=0)

# Fundamentals cache file
FUND_CACHE = os.path.join(CACHE_DIR, "fundamentals.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize FRED client using Streamlit secrets
fred = Fred(api_key=st.secrets["FRED"]["API_KEY"])


# ─── Utility / Caching for Fundamentals ────────────────────────────────────────
def load_fund_cache():
    if os.path.exists(FUND_CACHE):
        with open(FUND_CACHE, "r") as f:
            return json.load(f)
    return {}

def save_fund_cache(cache):
    with open(FUND_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

def get_fundamentals(ticker, cache, st_status=None):
    """Fetch trailing PE, PB, ROE for a ticker, with local JSON caching."""
    if ticker in cache:
        return cache[ticker]
    try:
        if st_status: st_status.text(f"Fetching fundamentals for {ticker}…")
        info = yf.Ticker(ticker).info
        data = {
            "PE": info.get("trailingPE", np.nan),
            "PB": info.get("priceToBook", np.nan),
            "ROE": info.get("returnOnEquity", np.nan)
        }
    except Exception as e:
        logging.warning(f"Fundamentals fetch failed for {ticker}: {e}")
        data = {"PE": np.nan, "PB": np.nan, "ROE": np.nan}
    cache[ticker] = data
    return data

def batch_fundamentals(tickers, cache, st_status=None):
    """Ensure fundamentals are cached for all tickers."""
    for t in tickers:
        if t not in cache:
            get_fundamentals(t, cache, st_status)
    return cache


# ─── 1. Fetch S&P 500 Constituents ─────────────────────────────────────────────
@memory.cache
def fetch_sp500_constituents():
    """Returns list of tickers and sector mapping from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(requests.get(url).text)[0]
    tickers = [s.replace(".", "-") for s in df.Symbol]
    sector_map = dict(zip(tickers, df["GICS Sector"]))
    return tickers, sector_map


# ─── 2. Fetch Market & Regime Data ──────────────────────────────────────────────
@memory.cache
def fetch_market_data(start: str, end: str):
    """
    Downloads OHLC for S&P 500 + SPY + VIX and computes monthly VIX & yield‐curve slope.
    """
    tickers, _ = fetch_sp500_constituents()
    all_syms = tickers + ["SPY", "^VIX"]
    df = yf.download(all_syms, start=start, end=end, auto_adjust=True, timeout=30)

    prices = df["Close"]
    highs  = df["High"]
    lows   = df["Low"]

    # monthly VIX
    monthly_vix = prices["^VIX"].resample("M").last()

    # yield curve slope via FRED
    rates2  = fred.get_series("DGS2", start, end)
    rates10 = fred.get_series("DGS10", start, end)
    ycs    = (rates10 - rates2).dropna()
    monthly_yc = ycs.resample("M").last()

    return prices, highs, lows, monthly_vix, monthly_yc


# ─── 3. Feature Engineering ─────────────────────────────────────────────────────
def engineer_features_single(ticker, prices, highs, lows, monthly_vix, monthly_yc, sector_map, fund_cache, st_status=None):
    """Compute features & target for one ticker; return (ticker, df)."""
    try:
        # monthly series
        prc_m = prices[ticker].resample("M").last()
        ret_m = prc_m.pct_change()
        vix_m = monthly_vix.shift(1)
        yc_s   = monthly_yc.shift(1)

        df = pd.DataFrame(index=ret_m.index)
        # momentum
        df["M1"]  = ret_m.shift(1)
        df["M3"]  = ret_m.rolling(3).mean().shift(1)
        df["M6"]  = ret_m.rolling(6).mean().shift(1)
        df["M12"] = ret_m.rolling(12).mean().shift(1)
        # volatility
        df["Vol3"] = ret_m.rolling(3).std().shift(1)
        # beta
        spy_m = prices["SPY"].resample("M").last().pct_change()
        df["Beta"] = ret_m.rolling(12).cov(spy_m).shift(1) / spy_m.rolling(12).var().shift(1)
        # regime
        df["VIX"]     = vix_m
        df["YC_slope"]= yc_s

        # MACD & signal
        ema12 = prc_m.ewm(span=12).mean()
        ema26 = prc_m.ewm(span=26).mean()
        macd  = ema12 - ema26
        df["MACD"]        = macd
        df["MACD_Signal"] = macd.ewm(span=9).mean()

        # ADX
        up = highs[ticker].diff()
        dn = lows[ticker].diff() * -1
        plus = up.where((up>0)&(up>dn), 0.0)
        minus= dn.where((dn>0)&(dn>up),0.0)
        tr1 = highs[ticker] - lows[ticker]
        tr2 = (highs[ticker] - prc_m.shift(1)).abs()
        tr3 = (lows[ticker]  - prc_m.shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14).mean()
        di_plus  = 100*(plus.ewm(alpha=1/14).mean()/atr)
        di_minus = 100*(minus.ewm(alpha=1/14).mean()/atr)
        df["ADX"] = ( (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9) *100 ).ewm(alpha=1/14).mean()

        # sector momentum
        sector = sector_map.get(ticker)
        sec_rets = ret_m.groupby(pd.Series(sector_map)).mean()
        sec1 = sec_rets[sector].rolling(1).mean().shift(1) if sector in sec_rets.columns else 0
        sec3 = sec_rets[sector].rolling(3).mean().shift(1) if sector in sec_rets.columns else 0
        df["SecMom1"] = sec1
        df["SecMom3"] = sec3
        df["SecRel"]  = df["M1"] - df["SecMom1"]

        # fundamentals
        f = get_fundamentals(ticker, fund_cache, st_status)
        df["PE"], df["PB"], df["ROE"] = f["PE"], f["PB"], f["ROE"]

        # target
        df["Target"] = ret_m.shift(-1)

        df.dropna(inplace=True)
        return ticker, df

    except Exception as e:
        logging.warning(f"Feat eng failed for {ticker}: {e}")
        return ticker, pd.DataFrame()


def engineer_features(prices, highs, lows, monthly_vix, monthly_yc, sector_map, fund_cache, st_status=None):
    """Parallel feature engineering across all tickers."""
    tickers = [t for t in prices.columns if t not in ["SPY","^VIX"]]
    results = Parallel(n_jobs=-1)(
        delayed(engineer_features_single)(
            t, prices, highs, lows, monthly_vix, monthly_yc,
            sector_map, fund_cache, st_status
        ) for t in tickers
    )
    feats = {t:df for t,df in results if not df.empty}
    return feats


# ─── 4. Portfolio Optimization ─────────────────────────────────────────────────
def optimize_portfolio(expected, cov, sector_map, max_w=0.25):
    """Maximize Sharpe with sector cap 30% and individual cap max_w."""
    tickers = expected.index
    S = cov * 12
    def neg_sharpe(w):
        r = np.dot(w, expected) * 12
        vol = np.sqrt(w @ S @ w)
        return -r/ (vol + 1e-9)

    # constraints
    cons = [{"type":"eq","fun":lambda w: w.sum()-1}]
    # sector caps
    sectors = {}
    for i,t in enumerate(tickers):
        sec = sector_map.get(t)
        sectors.setdefault(sec,[]).append(i)
    for idx in sectors.values():
        cons.append({"type":"ineq","fun":lambda w,idx=idx:0.30 - w[idx].sum()})

    bounds = [(0,max_w)]*len(tickers)
    w0 = np.ones(len(tickers))/len(tickers)
    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=tickers)


# ─── 5. Walk‐Forward Validation ────────────────────────────────────────────────
def run_walk_forward(features, start_date, st_status=None):
    """Backtest by expanding window, returns DataFrame of monthly results."""
    dates = sorted([d for d in next(iter(features.values())).index if d>=start_date])
    allres=[]
    for i,d in enumerate(dates):
        if st_status: st_status.text(f"Backtest {d.strftime('%Y-%m')} ({i+1}/{len(dates)})")
        train_end = d - relativedelta(months=1)
        Xtr, ytr = [],[]
        Xt, tickers, acts = [],[],[]
        for t,df in features.items():
            tr = df[df.index<=train_end]
            if len(tr)>=24:
                Xtr.append(tr.drop("Target",axis=1).values)
                ytr.append(tr["Target"].values)
            if d in df.index:
                row = df.loc[d]
                Xt.append(row.drop("Target").values)
                acts.append(row["Target"])
                tickers.append(t)
        if not Xtr or not Xt: continue
        Xtr = np.vstack(Xtr); ytr = np.hstack(ytr); Xt = np.vstack(Xt)
        scaler = StandardScaler().fit(Xtr)
        model  = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        p50    = model.fit(scaler.transform(Xtr), ytr).predict(scaler.transform(Xt))
        month_df = pd.DataFrame({
            "Date":d, "Ticker":tickers,
            "P50":p50, "Actual":acts
        })
        allres.append(month_df)
    return pd.concat(allres,ignore_index=True)


# ─── 6. Performance Metrics ───────────────────────────────────────────────────
def analyze_results(res_df, prices, sector_map):
    """Turn monthly predictions into portfolio returns & metrics."""
    port_rets=[]
    hist = prices["SPY"].resample("M").last().pct_change()  # we'll compare to SPY
    prev_w = None

    for dt in res_df.Date.unique():
        sub = res_df[res_df.Date==dt].set_index("Ticker")
        subs = sub[sub.P50>0].nlargest(10,"P50")
        if len(subs)>1:
            er = subs.P50
            cov = prices[tickers].resample("M").last().pct_change().loc[:dt - relativedelta(months=1)][subs.index].cov()
            w = optimize_portfolio(er, cov, sector_map)
            port_rets.append((w*subs.Actual).sum())
            prev_w = w
        else:
            port_rets.append(0.0)

    portfolio = pd.Series(port_rets,index=sorted(res_df.Date.unique()))
    df = pd.DataFrame({
        "Model": portfolio,
        "SPY": hist.reindex(portfolio.index)
    })
    df["Cumulative"] = (1+df.Model).cumprod()
    rf = 0.0

    sharpe = (df.Model.mean()*12 - rf)/(df.Model.std()*np.sqrt(12))
    maxdd  = (1 - df.Cumulative/df.Cumulative.cummax()).max()
    annret = df.Model.mean()*12

    metrics = {"Sharpe":sharpe, "MaxDrawdown":maxdd, "AnnualReturn":annret}
    return df, metrics


# ─── 7. Public API ─────────────────────────────────────────────────────────────
def get_available_sectors():
    _, sm = fetch_sp500_constituents()
    return sorted(set(sm.values()))

def run_backtest_pipeline(st_status=None, start_date="2016-01-01"):
    """Entry point for Strategy Backtest tab."""
    sd = pd.to_datetime(start_date)
    tickers, sector_map = fetch_sp500_constituents()
    prices, highs, lows, mvix, myc = fetch_market_data("2013-01-01", datetime.today().strftime("%Y-%m-%d"))
    fund_cache = load_fund_cache()
    batch_fundamentals(tickers, fund_cache, st_status)
    save_fund_cache(fund_cache)

    feats = engineer_features(prices, highs, lows, mvix, myc, sector_map, fund_cache, st_status)
    if not feats:
        raise RuntimeError("Feature engineering produced NO series—check backend.log")

    res = run_walk_forward(feats, sd, st_status)
    df, mets = analyze_results(res, prices, sector_map)
    return df, mets

def run_live_prediction_pipeline(st_status=None, selected_sectors=None, max_stock_weight=0.25):
    """Entry point for Live Portfolio tab."""
    tickers, sector_map = fetch_sp500_constituents()
    if selected_sectors:
        tickers = [t for t in tickers if sector_map.get(t) in selected_sectors]
        sector_map = {t: sector_map[t] for t in tickers}
    prices, highs, lows, mvix, myc = fetch_market_data("2013-01-01", datetime.today().strftime("%Y-%m-%d"))
    fund_cache = load_fund_cache()
    batch_fundamentals(tickers, fund_cache, st_status)
    save_fund_cache(fund_cache)

    feats = engineer_features(prices, highs, lows, mvix, myc, sector_map, fund_cache, st_status)
    # build final prediction matrix
    Xp, tickers_p = [], []
    for t,df in feats.items():
        Xp.append(df.drop("Target",axis=1).iloc[-1].values)
        tickers_p.append(t)
    if not Xp:
        return pd.DataFrame()
    Xp = np.vstack(Xp)
    # train on all history
    full = pd.concat(feats.values())
    scaler = StandardScaler().fit(full.drop("Target",axis=1).values)
    model  = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(scaler.transform(full.drop("Target",axis=1).values), full.Target.values)
    p50 = model.predict(scaler.transform(Xp))

    preds = pd.Series(p50, index=tickers_p, name="P50").nlargest(15)
    cov   = prices[tickers_p].resample("M").last().pct_change().iloc[-12:].cov()
    w     = optimize_portfolio(preds, cov, sector_map, max_stock_weight)
    return pd.DataFrame({"Weight":w})
