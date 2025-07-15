# backend.py — V12.4 “Empty‐features” Guarded Engine
# ==============================================================================
# Adds a check that 'feats' is not empty before proceeding to avoid StopIteration.
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
from xgboost import XGBRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# optional pandas_datareader for FRED
try:
    import pandas_datareader.data as web
except ImportError:
    web = None

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
    """Fetch & cache fundamentals for each ticker."""
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
        except:
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
    """Wrapper so we can pass st_status (ignored by the cache)."""
    if st_status:
        st_status.text("Fetching S&P 500 constituents...")
    return _fetch_sp500_constituents()

def get_available_sectors():
    _, sector_map = _fetch_sp500_constituents()
    return sorted(set(sector_map.values()))

# --- Market + macro data ---
@memory.cache
def _fetch_data(tickers, start, end):
    all_t = tickers + ["SPY","^VIX"]
    raw = yf.download(all_t, start=start, end=end, auto_adjust=True,
                      timeout=30, group_by="ticker")
    closes = raw.xs("Close",axis=1,level=1)
    highs  = raw.xs("High", axis=1,level=1)
    lows   = raw.xs("Low",  axis=1,level=1)
    # macro
    if web:
        try:
            m = web.DataReader(["CPIAUCSL","ISM"],"fred",start, end)
            m["CPI_YoY"] = m["CPIAUCSL"].pct_change(12)*100
            macro = m
        except:
            macro = pd.DataFrame()
    else:
        macro = pd.DataFrame()
    return closes, highs, lows, macro

def fetch_data(tickers, start, end, st_status=None):
    """Wrapper so we can pass st_status (ignored by the cache)."""
    if st_status:
        st_status.text("Downloading market & macro data...")
    return _fetch_data(tickers, start, end)

# --- Feature engineering ---
def engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status=None):
    if st_status:
        st_status.text("Engineering features...")
    mclose = closes.resample("M").last()
    mret   = mclose.pct_change().dropna()
    spyret = mret["SPY"]
    vix    = mclose["^VIX"].shift(1)
    # MACD
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
            st_status.text(f"Engineering features for {t}...")
        df = pd.DataFrame(index=mret.index)
        df["M1"]        = mret[t].shift(1)
        df["Vol3"]      = mret[t].rolling(3).std().shift(1)
        df["Beta"]      = mret[t].rolling(12).cov(spyret).shift(1)/spyret.rolling(12).var()
        df["VIX"]       = vix
        df["MACD"]      = macd[t]
        df["MACD_Signal"]= sig[t]
        df["ADX"]       = adx[t]
        sec = sector_map.get(t)
        df["SecMom1"]   = sec1[sec].shift(1) if sec in sec1 else 0
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
        if not df.empty:
            feats[t] = df
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

# --- Walk-forward backtest ---
def run_backtest_pipeline(st_status=None, start_date="2016-01-01"):
    START, END = "2013-01-01", datetime.today().strftime("%Y-%m-%d")
    val_start  = pd.to_datetime(start_date)
    ts, sector_map = fetch_sp500_constituents(st_status)
    closes, highs, lows, macro = fetch_data(ts, START, END, st_status)
    if closes.empty:
        raise RuntimeError("Market data empty")
    fund_cache = load_fund_cache()
    fund_cache = get_fundamentals_batch(ts, fund_cache, st_status)
    save_fund_cache(fund_cache)
    feats = engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status)

    # *** GUARD AGAINST EMPTY FEATURES ***
    if not feats:
        raise RuntimeError("Feature engineering produced NO series—check data fetch/logs")

    results = []
    dates = sorted([d for d in feats[next(iter(feats))].index if d >= val_start])
    for dt in dates:
        Xtr, Ytr, Xte, actuals, tickers = [], [], [], [], []
        for t, df in feats.items():
            tr = df[df.index < dt]
            if len(tr) >= 24:
                Xtr.append(tr.drop("Target", axis=1).values)
                Ytr.append(tr["Target"].values)
            if dt in df.index:
                row = df.loc[dt]
                Xte.append(row.drop("Target", axis=1).values)
                actuals.append(row["Target"])
                tickers.append(t)
        if not Xte:
            continue
        Xtr, Ytr, Xte = np.vstack(Xtr), np.hstack(Ytr), np.vstack(Xte)
        mdl = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        S   = StandardScaler().fit(Xtr)
        P50 = mdl.fit(S.transform(Xtr), Ytr).predict(S.transform(Xte))
        dfm = pd.DataFrame({"Ticker": tickers, "P50": P50, "Actual": actuals}).set_index("Ticker")
        hist = closes.resample("M").last().pct_change().loc[:dt].tail(12)[tickers]
        w    = optimize_weights(dfm, hist, sector_map)
        ret  = w.dot(dfm["Actual"])
        results.append({"Date": dt, "Return": ret})

    out = pd.DataFrame(results).set_index("Date")
    out["Cumulative"] = (1 + out["Return"]).cumprod()
    ann    = out["Return"].mean() * 12
    vol    = out["Return"].std() * np.sqrt(12)
    sharpe = ann / vol
    mdd    = (1 - out["Cumulative"] / out["Cumulative"].cummax()).max()
    return out, {"Sharpe": sharpe, "MaxDrawdown": mdd, "AnnualReturn": ann}

# --- Live prediction ---
def run_live_prediction_pipeline(st_status=None, selected_sectors=None, max_stock_weight=0.25):
    START, END = "2013-01-01", datetime.today().strftime("%Y-%m-%d")
    ts, sector_map = fetch_sp500_constituents(st_status)
    if selected_sectors:
        ts = [t for t in ts if sector_map[t] in selected_sectors]
        sector_map = {t: sector_map[t] for t in ts}
    closes, highs, lows, macro = fetch_data(ts, START, END, st_status)
    fund_cache = load_fund_cache()
    fund_cache = get_fundamentals_batch(ts, fund_cache, st_status)
    save_fund_cache(fund_cache)
    feats = engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status)

    Xtr, Ytr, Xp, tcks = [], [], [], []
    for t, df in feats.items():
        Xtr.append(df.drop("Target", axis=1).values)
        Ytr.append(df["Target"].values)
        Xp.append(df.drop("Target", axis=1).iloc[-1].values)
        tcks.append(t)
    if not Xtr:
        return pd.DataFrame()
    Xtr, Ytr, Xp = np.vstack(Xtr), np.hstack(Ytr), np.vstack(Xp)
    mdl = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    S   = StandardScaler().fit(Xtr)
    P50 = mdl.fit(S.transform(Xtr), Ytr).predict(S.transform(Xp))
    dfm = pd.DataFrame({"Ticker": tcks, "P50": P50}).set_index("Ticker")
    cutoff = np.percentile(P50, 90)
    wdf    = dfm[dfm["P50"] > cutoff].sort_values("P50", ascending=False).head(15)
    hist   = closes.resample("M").last().pct_change().tail(13).iloc[:-1]
    cov    = hist[wdf.index].cov() * 12
    def neg_sh(w): return -(w.dot(wdf["P50"]) * 12) / np.sqrt(w.dot(cov).dot(w) + 1e-9)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0, max_stock_weight)] * len(wdf)
    res  = minimize(neg_sh, np.ones(len(wdf)) / len(wdf), bounds=bnds, constraints=cons)
    return pd.Series(res.x, index=wdf.index).to_frame("Weight")
