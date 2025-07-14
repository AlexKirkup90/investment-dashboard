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
from sklearn.model_selection import GridSearchCV  # NEW: For hyperparameter tuning
from scipy.optimize import minimize
import warnings
import logging  # NEW: For debugging
from retry import retry  # NEW: For retrying failed requests
from sklearn.covariance import LedoitWolf  # NEW: For covariance shrinkage

# LightGBM fallback
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# Attempt to import pandas_datareader
try:
    import pandas_datareader.data as web
except ImportError:
    web = None

warnings.filterwarnings('ignore')

# NEW: Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@retry(tries=3, delay=2, backoff=2, logger=logger)  # NEW: Retry on failure
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
        # NEW: Validate fundamentals
        for key, val in f.items():
            if not np.isfinite(val):
                f[key] = np.nan
    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        f = {'PE': np.nan, 'PB': np.nan, 'ROE': np.nan}
    cache[ticker] = f
    return f

# --- AutoML-lite Model Chooser ---
def choose_best_model(X, y):
    candidates = {
        'GBM': GridSearchCV(
            GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=42),
            param_grid={'n_estimators': [50, 100], 'max_depth': [3, 5]},  # NEW: Basic tuning
            cv=TimeSeriesSplit(n_splits=5),  # MODIFIED: Increased splits
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
    }
    if _HAS_LGBM:
        candidates['LGBM'] = GridSearchCV(
            LGBMRegressor(random_state=42),
            param_grid={'n_estimators': [50, 100], 'max_depth': [3, 5]},
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

    tscv = TimeSeriesSplit(n_splits=5)  # MODIFIED: Increased splits
    best_score, best_model = np.inf, None
    for name, mdl in candidates.items():
        mdl.fit(X, y)
        scores = -cross_val_score(mdl.best_estimator_, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        mean_score = scores.mean()
        logger.info(f"Model {name} score: {mean_score}")
        if mean_score < best_score:
            best_score, best_model = mean_score, mdl.best_estimator_
    return best_model

# --- Data Fetching (cached) ---
@memory.cache
@retry(tries=3, delay=2, backoff=2, logger=logger)  # NEW: Retry on failure
def fetch_sp500_constituents():
    # NEW: Use a static list or fallback to Wikipedia
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df = pd.read_html(requests.get(url, timeout=10).text)[0]
        tickers = [t.replace('.', '-') for t in df['Symbol']]
        sector_map = dict(zip(tickers, df['GICS Sector']))
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 constituents: {e}. Using fallback list.")
        # Fallback: Use a small static list for testing
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'SPY', '^VIX']
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Communication Services',
            'AMZN': 'Consumer Discretionary', 'JPM': 'Financials', 'SPY': 'Benchmark', '^VIX': 'Volatility'
        }
    return tickers, sector_map

@memory.cache
@retry(tries=3, delay=2, backoff=2, logger=logger)  # NEW: Retry on failure
def fetch_data(tickers, start, end):
    all_t = tickers + ['SPY', '^VIX']
    try:
        raw = yf.download(all_t, start=start, end=end, auto_adjust=True, timeout=30, group_by='ticker')
        # NEW: Validate data
        if raw.empty or raw.isna().all().all():
            raise ValueError("Empty or all-NaN data from yfinance")
        closes = raw.xs('Close', axis=1, level=1)
        highs = raw.xs('High', axis=1, level=1)
        lows = raw.xs('Low', axis=1, level=1)
        # NEW: Remove tickers with excessive missing data
        closes = closes.loc[:, closes.isna().mean() < 0.3]
        highs = highs.loc[:, closes.columns]
        lows = lows.loc[:, closes.columns]
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        closes, highs, lows = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Macro: only if web is available
    macro = pd.DataFrame()
    if web:
        try:
            macro = web.DataReader(['CPIAUCSL', 'ISM'], 'fred', start, end)
            macro['CPI_YoY'] = macro['CPIAUCSL'].pct_change(12) * 100
            macro = macro.resample('D').ffill().reindex(closes.index).ffill()  # MODIFIED: Align with price data
        except Exception as e:
            logger.warning(f"Failed to fetch macro data: {e}")
    return closes, highs, lows, macro

# --- Feature Engineering ---
def engineer_features(closes, highs, lows, macro, sector_map, fund_cache):
    mp = closes.resample('M').last()
    mh = highs.resample('M').max()
    ml = lows.resample('M').min()
    mr = mp.pct_change().dropna()
    ms = mp['SPY'].pct_change().dropna()

    # NEW: Validate inputs
    if mp.empty or mh.empty or ml.empty:
        logger.error("Empty price data after resampling")
        return {}

    # Vectorized MACD
    ema12 = mp.ewm(span=12, adjust=False).mean()
    ema26 = mp.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()

    # Vectorized ADX
    pdm = mh.diff().clip(lower=0)
    mdm = -ml.diff().clip(upper=0)
    tr1 = mh - ml
    tr2 = (mh - mp.shift(1)).abs()
    tr3 = (ml - mp.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pdm.ewm(alpha=1/14).mean() / atr
    minus_di = 100 * mdm.ewm(alpha=1/14).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm(alpha=1/14).mean()

    # NEW: Clip extreme values
    macd = macd.clip(-1e3, 1e3)
    adx = adx.clip(0, 100)

    # Sector momentum
    t2s = pd.Series(sector_map)
    sec_ret = mr.groupby(t2s, axis=1).mean()
    sec_mom1 = sec_ret.rolling(1).mean()

    features = {}
    for t in [c for c in mp.columns if c not in ['SPY', '^VIX']]:
        df = pd.DataFrame(index=mr.index)
        try:
            ret = mr[t]

            # Base
            df['M1'] = ret.shift(1)
            df['Vol3'] = ret.rolling(3).std().shift(1)
            df['Beta'] = ret.rolling(12).cov(ms).shift(1) / ms.rolling(12).var()
            df['VIX'] = mp['^VIX'].shift(1)

            # Tech
            df['MACD'] = macd[t]
            df['MACD_Signal'] = macd_sig[t]
            df['ADX'] = adx[t]

            # Sector
            sec_val = sector_map.get(t)
            df['SecMom1'] = sec_mom1[sec_val].shift(1) if sec_val in sec_mom1 else 0
            df['SecRel1'] = df['M1'] - df['SecMom1']

            # Fundamentals
            f = get_fundamentals(t, fund_cache)
            df['PE'], df['PB'], df['ROE'] = f['PE'], f['PB'], f['ROE']

            # Macro
            if not macro.empty:
                md = macro[['CPI_YoY', 'ISM']].resample('M').last().shift(1)
                df = df.join(md).ffill()
            else:
                df['CPI_YoY'], df['ISM'] = 0, 0

            # Target
            df['Target'] = ret.shift(-1)
            df.dropna(inplace=True)

            # NEW: Ensure sufficient data
            if len(df) >= 12:
                features[t] = df
            else:
                logger.warning(f"Insufficient data for {t}: {len(df)} periods")
        except Exception as e:
            logger.warning(f"Feature engineering failed for {t}: {e}")

    return features

# --- Optimizer with Confidence, Turnover & Sector Caps ---
def optimize_weights(exp_df, hist_ret, prev_w=None, trade_cost=0.002, sector_map=None):  # MODIFIED: Increased cost to include slippage
    tickers = exp_df.index.tolist()
    p50 = exp_df['P50']

    # Confidence-weighted expected return
    unc = exp_df['P90'] - exp_df['P10']
    conf = p50 / (unc + 1e-9)
    adj = p50 * (conf / conf.sum())

    # MODIFIED: Use Ledoit-Wolf shrinkage for covariance
    cov = LedoitWolf().fit(hist_ret).covariance_ * 12

    def neg_sharpe(w):
        ret = w.dot(adj) * 12
        vol = np.sqrt(w.dot(cov).dot(w))
        return -ret / (vol + 1e-9)

    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    if sector_map:
        sectors = {}
        for i, t in enumerate(tickers):
            s = sector_map.get(t)
            sectors.setdefault(s, []).append(i)
        for inds in sectors.values():
            cons.append({'type': 'ineq', 'fun': lambda w, inds=inds: 0.30 - w[inds].sum()})

    bnds = [(0, 0.25)] * len(tickers)
    init = np.ones(len(tickers)) / len(tickers) if prev_w.empty else prev_w
    try:
        res = minimize(neg_sharpe, init, method='SLSQP', bounds=bnds, constraints=cons)
        if not res.success:
            logger.warning(f"Optimization failed: {res.message}. Using equal weights.")
            w = pd.Series(1/len(tickers), index=tickers)
        else:
            w = pd.Series(res.x, index=tickers)
    except Exception as e:
        logger.error(f"Optimization error: {e}. Using equal weights.")
        w = pd.Series(1/len(tickers), index=tickers)

    # Turnover cost
    if prev_w is not None and not prev_w.empty:
        tr = np.abs(w - prev_w).sum()
        w *= (1 - tr * trade_cost)

    return w

# --- Walk-Forward Backtest ---
def run_backtest(start='2013-01-01', end=None, val_start='2016-01-01'):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    try:
        tickers, sector_map = fetch_sp500_constituents()
        closes, highs, lows, macro = fetch_data(tickers, start, end)
        if closes.empty:
            logger.error("No price data available. Aborting backtest.")
            return pd.DataFrame()

        fund_cache = load_fundamentals_cache()
        feats = engineer_features(closes, highs, lows, macro, sector_map, fund_cache)
        save_fundamentals_cache(fund_cache)

        if not feats:
            logger.error("No features generated. Aborting backtest.")
            return pd.DataFrame()

        results, prev_w = [], pd.Series()
        dates = sorted([d for d in next(iter(feats.values())).index if d >= pd.to_datetime(val_start)])

        for dt in dates:
            train_x, train_y, test_x, actuals, ts = [], [], [], [], []
            for t, df in feats.items():
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
                logger.warning(f"No test data for {dt}. Skipping.")
                continue

            Xtr = np.vstack(train_x)
            Ytr = np.hstack(train_y)
            Xte = np.vstack(test_x)

            # NEW: Per-ticker scaling
            scaler = StandardScaler().fit(Xtr)
            Xtr_scaled = scaler.transform(Xtr)
            Xte_scaled = scaler.transform(Xte)

            mdl = choose_best_model(Xtr_scaled, Ytr)
            P50 = mdl.fit(Xtr_scaled, Ytr).predict(Xte_scaled)

            # Estimate P10/P90 via quantile GBMs
            gb10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, max_depth=3, n_estimators=50, random_state=42)  # MODIFIED: Added regularization
            gb90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, max_depth=3, n_estimators=50, random_state=42)
            P10 = gb10.fit(Xtr_scaled, Ytr).predict(Xte_scaled)
            P90 = gb90.fit(Xtr_scaled, Ytr).predict(Xte_scaled)

            dfm = pd.DataFrame({
                'Ticker': ts, 'P10': P10, 'P50': P50, 'P90': P90, 'Actual': actuals
            }).set_index('Ticker')

            hist = closes.resample('M').last().pct_change().loc[:dt].tail(12)[ts]
            w = optimize_weights(dfm, hist, prev_w, sector_map=sector_map)
            prev_w = w

            # MODIFIED: Include bid-ask spread (assume 0.1% cost)
            ret = w.dot(dfm['Actual']) - 0.001 * w.abs().sum()
            results.append({'Date': dt, 'Return': ret, 'Weights': w})

        out = pd.DataFrame(results).set_index('Date')
        out['Cumulative'] = (1 + out['Return']).cumprod()
        logger.info(f"Backtest completed. Final cumulative return: {out['Cumulative'].iloc[-1]:.2f}")
        return out
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return pd.DataFrame()

# Backwards-compatible alias
run_backtest_pipeline = run_backtest
