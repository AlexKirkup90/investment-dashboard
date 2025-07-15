# backend.py — V12.7 “fredapi & Single-Download” Engine
import os, json, logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from xgboost import XGBRegressor
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename="backend.log", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")

# ─── CACHE SETUP ──────────────────────────────────────────────────────────────
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)
FUND_FILE = os.path.join(CACHE_DIR, "fundamentals.json")

def load_fund_cache():
    if os.path.exists(FUND_FILE):
        return json.load(open(FUND_FILE))
    return {}

def save_fund_cache(c):
    json.dump(c, open(FUND_FILE, "w"), indent=2)

# ─── S&P 500 CONSTITUENTS ────────────────────────────────────────────────────
@memory.cache
def fetch_sp500_constituents():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = [s.replace(".", "-") for s in df.Symbol]
    sector_map = dict(zip(tickers, df["GICS Sector"]))
    return tickers, sector_map

def get_available_sectors():
    _, sm = fetch_sp500_constituents()
    return sorted(set(sm.values()))

# ─── MARKET + MACRO DATA ──────────────────────────────────────────────────────
@memory.cache
def fetch_market_data(tickers, start, end):
    # one call for OHLC
    raw = yf.download(tickers + ["SPY", "^VIX"], start=start, end=end,
                      auto_adjust=True, timeout=30)
    closes = raw["Close"]
    highs  = raw["High"]
    lows   = raw["Low"]
    # FRED via fredapi
    macro = pd.DataFrame()
    try:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        cpi = fred.get_series("CPIAUCSL", start, end)
        ism = fred.get_series("ISM", start, end)
        yc2 = fred.get_series("DGS2", start, end)
        yc10= fred.get_series("DGS10", start, end)
        macro = pd.DataFrame({
            "CPI_YoY": cpi.pct_change(12) * 100,
            "ISM": ism,
            "YC_slope": (yc10 - yc2)
        })
    except Exception as e:
        logging.warning(f"fredapi error, skipping macro: {e}")
    return closes, highs, lows, macro

# ─── FUNDAMENTALS CACHING ─────────────────────────────────────────────────────
def fetch_fundamentals_batch(tickers, cache, st_status=None):
    for t in tickers:
        if t in cache: continue
        try:
            if st_status: st_status.text(f"Loading fundamentals for {t}…")
            info = yf.Ticker(t).info
            cache[t] = {
                "PE": info.get("trailingPE", np.nan),
                "PB": info.get("priceToBook", np.nan),
                "ROE": info.get("returnOnEquity", np.nan)
            }
        except Exception as e:
            logging.warning(f"fund fetch failed {t}: {e}")
            cache[t] = {"PE": np.nan, "PB": np.nan, "ROE": np.nan}
    return cache

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
def engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status=None):
    # monthly closes & returns
    mclose = closes.resample("M").last()
    mret   = mclose.pct_change().dropna(how="all")
    spyret = mret["SPY"]
    vix    = mclose["^VIX"].shift(1)

    # MACD & signal
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
    plus  = ph.diff().clip(lower=0).ewm(alpha=1/14).mean()
    minus = (-pl.diff().clip(upper=0)).ewm(alpha=1/14).mean()
    adx   = ((plus - minus).abs()/(plus + minus + 1e-9)*100).ewm(alpha=1/14).mean()

    # Sector momentum
    sec_srs = mret.groupby(pd.Series(sector_map), axis=1).mean()
    sec1    = sec_srs.rolling(1).mean()

    feats = {}
    for t in [c for c in mclose.columns if c not in ["SPY","^VIX"]]:
        if st_status: st_status.text(f"Feature-engineering {t}")
        df = pd.DataFrame(index=mret.index)
        df["M1"]    = mret[t].shift(1)
        df["Vol3"]  = mret[t].rolling(3).std().shift(1)
        df["Beta"]  = mret[t].rolling(12).cov(spyret).shift(1) / spyret.rolling(12).var()
        df["VIX"]   = vix
        df["MACD"]  = macd[t]
        df["Signal"]= sig[t]
        df["ADX"]   = adx[t]
        df["SecMom"]= sec1.get(t, 0.0)
        df["SecRel"]= df["M1"] - df["SecMom"]

        # fundamentals
        f = fund_cache.get(t, {})
        df["PE"], df["PB"], df["ROE"] = f.get("PE", np.nan), f.get("PB", np.nan), f.get("ROE", np.nan)

        # macro & yield-curve slope
        if not macro.empty:
            md = macro.resample("M").last().shift(1)
            df = df.join(md, how="left").ffill().fillna(0)
        else:
            df["CPI_YoY"], df["ISM"], df["YC_slope"] = 0,0,0

        df["Target"] = mret[t].shift(-1)
        df.dropna(inplace=True)
        if not df.empty:
            feats[t] = df
        logging.debug(f"{t}: {df.shape[0]} rows")
    logging.info(f"Built features for {len(feats)} tickers")
    return feats

# ─── PORTFOLIO OPTIMIZER ──────────────────────────────────────────────────────
def optimize_weights(dfm, hist_ret, sector_map, maxw=0.25):
    tickers = dfm.index.tolist()
    p50     = dfm["P50"]
    cov     = hist_ret.cov()*12

    def negsh(w):
        r = w.dot(p50)*12
        v = np.sqrt(w.dot(cov).dot(w))
        return -r/(v+1e-9)

    # constraints: sum=1 + each sector ≤30%
    cons = [{"type":"eq","fun":lambda w:w.sum()-1}]
    bysec = {}
    for i,t in enumerate(tickers):
        sec = sector_map.get(t)
        bysec.setdefault(sec, []).append(i)
    for idx in bysec.values():
        cons.append({"type":"ineq","fun":lambda w,i=idx:0.30 - w[i].sum()})
    bounds = [(0,maxw)]*len(tickers)
    res = minimize(negsh, np.ones(len(tickers))/len(tickers),
                   bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=tickers)

# ─── WALK-FORWARD BACKTEST ────────────────────────────────────────────────────
def run_backtest_pipeline(st_status=None, start_date="2016-01-01"):
    START, END = "2013-01-01", datetime.today().strftime("%Y-%m-%d")
    val_start  = pd.to_datetime(start_date)

    tickers, sector_map = fetch_sp500_constituents()
    closes, highs, lows, macro = fetch_market_data(tickers, START, END)
    fund_cache = load_fund_cache()
    fund_cache = fetch_fundamentals_batch(tickers, fund_cache, st_status)
    save_fund_cache(fund_cache)

    feats = engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status)
    if st_status:
        st_status.text(f"{len(feats)} tickers after feature-engineering")
    if not feats:
        raise RuntimeError("No features generated—check backend.log")

    results = []
    dates = sorted([d for d in feats[next(iter(feats))].index if d >= val_start])
    for dt in dates:
        Xtr, Ytr, Xte, actuals, names = [], [], [], [], []
        for t, df in feats.items():
            train = df[df.index < dt]
            if len(train) >= 24:
                Xtr.append(train.drop("Target",axis=1).values)
                Ytr.append(train["Target"].values)
            if dt in df.index:
                row = df.loc[dt]
                Xte.append(row.drop("Target").values)
                actuals.append(row["Target"])
                names.append(t)
        if not Xte: continue

        Xtr, Ytr, Xte = np.vstack(Xtr), np.hstack(Ytr), np.vstack(Xte)
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        scaler = StandardScaler().fit(Xtr)
        P50    = model.fit(scaler.transform(Xtr), Ytr).predict(scaler.transform(Xte))

        dfm = pd.DataFrame({"Ticker":names,"P50":P50,"Actual":actuals}).set_index("Ticker")
        hist = closes.resample("M").last().pct_change().loc[:dt].tail(12)[names]
        w    = optimize_weights(dfm, hist, sector_map)
        ret  = w.dot(dfm["Actual"])
        results.append({"Date":dt,"Return":ret})

    out = pd.DataFrame(results).set_index("Date")
    out["Cumulative"] = (1+out["Return"]).cumprod()
    ann  = out["Return"].mean()*12
    vol  = out["Return"].std()*np.sqrt(12)
    shar = ann/vol
    mdd  = (1 - out["Cumulative"]/out["Cumulative"].cummax()).max()
    return out, {"Sharpe":shar,"MaxDrawdown":mdd,"AnnualReturn":ann}

# ─── LIVE PREDICTION ───────────────────────────────────────────────────────────
def run_live_prediction_pipeline(st_status=None, selected_sectors=None, max_stock_weight=0.25):
    START, END = "2013-01-01", datetime.today().strftime("%Y-%m-%d")
    tickers, sector_map = fetch_sp500_constituents()
    if selected_sectors:
        tickers = [t for t in tickers if sector_map.get(t) in selected_sectors]
        sector_map = {t:sector_map[t] for t in tickers}

    closes, highs, lows, macro = fetch_market_data(tickers, START, END)
    fund_cache = load_fund_cache()
    fund_cache = fetch_fundamentals_batch(tickers, fund_cache, st_status)
    save_fund_cache(fund_cache)

    feats = engineer_features(closes, highs, lows, macro, sector_map, fund_cache, st_status)
    if not feats:
        return pd.DataFrame()

    Xtr, Ytr, Xp, names = [], [], [], []
    for t, df in feats.items():
        Xtr.append(df.drop("Target",axis=1).values)
        Ytr.append(df["Target"].values)
        Xp.append(df.drop("Target",axis=1).iloc[-1].values)
        names.append(t)
    Xtr, Ytr, Xp = np.vstack(Xtr), np.hstack(Ytr), np.vstack(Xp)

    model  = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    P50    = model.fit(scaler.transform(Xtr), Ytr).predict(scaler.transform(Xp))

    dfm = pd.DataFrame({"Ticker":names,"P50":P50}).set_index("Ticker")
    cutoff = np.percentile(P50, 90)
    wdf    = dfm[dfm["P50"]>cutoff].nlargest(15,"P50")
    hist   = closes.resample("M").last().pct_change().tail(13).iloc[:-1]
    cov    = hist[wdf.index].cov()*12

    def negsh2(w): return -(w.dot(wdf["P50"])*12)/np.sqrt(w.dot(cov).dot(w)+1e-9)
    cons = [{"type":"eq","fun":lambda w:w.sum()-1}]
    bnds = [(0,max_stock_weight)]*len(wdf)
    res  = minimize(negsh2, np.ones(len(wdf))/len(wdf), bounds=bnds, constraints=cons)
    return pd.Series(res.x,index=wdf.index).to_frame("Weight")
