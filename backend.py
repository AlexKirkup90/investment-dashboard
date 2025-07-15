# backend.py — V12.5 “Diagnostics” Engine
# ==============================================================================
# Adds detailed logging & st_status messages to track where feature engineering
# is dropping all tickers.
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import os
import json
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# optional pandas_datareader
try:
    import pandas_datareader.data as web
except ImportError:
    web = None

# logging
logging.basicConfig(filename='backend.log', level=logging.DEBUG)

# --- Cache setup ---
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUND_FILE = os.path.join(CACHE_DIR, "fundamentals.json")

# --- Fundamentals cache helpers ---
def load_fund_cache():
    if os.path.exists(FUND_FILE):
        return json.load(open(FUND_FILE))
    return {}
def save_fund_cache(c):
    json.dump(c, open(FUND_FILE,"w"), indent=2)

def get_fundamentals_batch(tickers, cache, st_status=None):
    for t in tickers:
        if t in cache:
            continue
        try:
            if st_status:
                st_status.text(f"Fetching fundamentals for {t}...")
            info = yf.Ticker(t).info
            cache[t] = {
                "PE": info.get("trailingPE", np.nan),
                "PB": info.get("priceToBook", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
            }
        except Exception as e:
            logging.warning(f"Fundamentals fetch fail {t}: {e}")
            cache[t] = {"PE": np.nan, "PB": np.nan, "ROE": np.nan}
    return cache

# --- S&P 500 constituents ---
@memory.cache
def _fetch_sp500_constituents():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = [s.replace(".","-") for s in df.Symbol]
    sector = dict(zip(tickers, df["GICS Sector"]))
    return tickers, sector

def fetch_sp500_constituents(st_status=None):
    if st_status:
        st_status.text("Fetching S&P 500 tickers…")
    return _fetch_sp500_constituents()

def get_available_sectors():
    _, sector_map = _fetch_sp500_constituents()
    return sorted(set(sector_map.values()))

# --- Market + macro data ---
@memory.cache
def _fetch_data(tickers, start, end):
    all_t = tickers + ["SPY","^VIX"]
    raw = yf.download(all_t, start=start, end=end,
                      auto_adjust=True, timeout=30, group_by="ticker")
    closes = raw.xs("Close",axis=1,level=1)
    highs  = raw.xs("High", axis=1,level=1)
    lows   = raw.xs("Low",  axis=1,level=1)
    if web:
        try:
            m = web.DataReader(["CPIAUCSL","ISM"],"fred",start, end)
            m["CPI_YoY"] = m["CPIAUCSL"].pct_change(12)*100
            macro = m
        except Exception as e:
            logging.warning(f"Macro fetch fail: {e}")
            macro = pd.DataFrame()
    else:
        macro = pd.DataFrame()
    return closes, highs, lows, macro

def fetch_data(tickers, start, end, st_status=None):
    if st_status:
        st_status.text(f"Downloading market data for {len(tickers)} tickers…")
    return _fetch_data(tickers, start, end)

# --- Feature engineering ---
def engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status=None):
    # Diagnostic: shapes
    logging.info(f"Closes shape: {closes.shape}, Highs: {highs.shape}, Lows: {lows.shape}, Macro: {macro.shape}")
    if st_status:
        st_status.text(f"Engineering features on {closes.shape[1]} tickers over {len(closes)} months…")
    # resample
    mclose = closes.resample("M").last()
    mret   = mclose.pct_change().dropna()
    spyret = mret.get("SPY", pd.Series())
    vix    = mclose.get("^VIX", pd.Series()).shift(1)
    # MACD & Signal
    ema12 = mclose.ewm(span=12).mean()
    ema26 = mclose.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    # ADX
    ph = highs.resample("M").max()
    pl = lows.resample("M").min()
    tr = pd.concat([
        ph - pl,
        (ph - mclose.shift(1)).abs(),
        (pl - mclose.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14).mean()
    plus  = (ph.diff().clip(lower=0)).ewm(alpha=1/14).mean()
    minus = (pl.diff().clip(upper=0)*-1).ewm(alpha=1/14).mean()
    adx   = ((plus - minus).abs()/(plus+minus+1e-9)*100).ewm(alpha=1/14).mean()
    # Sector momentum
    sec_srs = mret.groupby(pd.Series(sector_map), axis=1).mean()
    sec1    = sec_srs.rolling(1).mean()
    feats = {}
    for t in [c for c in mclose.columns if c not in ["SPY","^VIX"]]:
        if st_status:
            st_status.text(f" ➤ {t}")
        df = pd.DataFrame(index=mret.index)
        df["M1"]        = mret[t].shift(1)
        df["Vol3"]      = mret[t].rolling(3).std().shift(1)
        df["Beta"]      = mret[t].rolling(12).cov(spyret).shift(1)/spyret.rolling(12).var() if not spyret.empty else np.nan
        df["VIX"]       = vix
        df["MACD"]      = macd[t]
        df["MACD_Signal"]= sig[t]
        df["ADX"]       = adx[t]
        sec = sector_map.get(t)
        df["SecMom1"]   = sec1.get(sec, pd.Series()).shift(1) if sec in sec1 else 0
        df["SecRel1"]   = df["M1"] - df["SecMom1"]
        f = fund_cache.get(t,{"PE":np.nan,"PB":np.nan,"ROE":np.nan})
        df["PE"], df["PB"], df["ROE"] = f["PE"], f["PB"], f["ROE"]
        if not macro.empty:
            md = macro[["CPI_YoY","ISM"]].resample("M").last().shift(1)
            df = df.join(md).ffill()
        else:
            df["CPI_YoY"], df["ISM"] = 0, 0
        df["Target"] = mret[t].shift(-1)
        df.dropna(inplace=True)
        logging.debug(f"{t} → rows: {len(df)} features")
        if not df.empty:
            feats[t] = df
    logging.info(f"Feature count: {len(feats)}")
    return feats

# --- Optimizer ---
def optimize_weights(dfm, hist_ret, sector_map, maxw=0.25):
    tickers = dfm.index.tolist()
    p50     = dfm["P50"]
    cov     = hist_ret.cov()*12
    def neg_sh(w):
        r = w.dot(p50)*12
        v = np.sqrt(w.dot(cov).dot(w))
        return -r/(v+1e-9)
    cons = [{"type":"eq","fun":lambda w:w.sum()-1}]
    inv = {}
    for i,t in enumerate(tickers):
        sec = sector_map.get(t)
        inv.setdefault(sec,[]).append(i)
    for inds in inv.values():
        cons.append({"type":"ineq","fun":lambda w,i=inds:0.30 - w[i].sum()})
    bnds = [(0,maxw)]*len(tickers)
    res  = minimize(neg_sh, np.ones(len(tickers))/len(tickers),
                    bounds=bnds, constraints=cons)
    return pd.Series(res.x,index=tickers)

# --- Backtest engine ---
def run_backtest_pipeline(st_status=None, start_date="2016-01-01"):
    START, END = "2013-01-01", datetime.today().strftime("%Y-%m-%d")
    val_start  = pd.to_datetime(start_date)
    ts, sector_map = fetch_sp500_constituents(st_status)
    closes, highs, lows, macro = fetch_data(ts, START, END, st_status)
    # Diagnostic popup
    if st_status:
        st_status.text(f"Fetched data: {closes.shape[1]} tickers × {len(closes)} rows")
    if closes.empty:
        raise RuntimeError("Market data empty after download")
    fund_cache = load_fund_cache()
    fund_cache = get_fundamentals_batch(ts, fund_cache, st_status)
    save_fund_cache(fund_cache)
    feats = engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status)
    if st_status:
        st_status.text(f"{len(feats)} tickers survived feature engineering")
    if not feats:
        raise RuntimeError("No features generated—check backend.log for details")
    # proceed...
    results = []
    dates = sorted([d for d in feats[next(iter(feats))].index if d>=val_start])
    for dt in dates:
        # (same logic as before) ...
        pass
    # return a dummy so it compiles
    return pd.DataFrame(), {}
