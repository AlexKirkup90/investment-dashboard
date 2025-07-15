import streamlit as st
import pandas as pd
import numpy as np
import backend  # Import backend module
import sys
import os
import logging
import traceback

# Setup logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)
logging.debug(f"App started at {pd.Timestamp.now()}")
logging.debug(f"Current working directory: {os.getcwd()}")
logging.debug(f"Python path: {sys.path}")

# Debug import of backend
try:
    if not hasattr(backend, 'run_backtest_pipeline'):
        logging.error("Debug: 'run_backtest_pipeline' not found in backend module.")
        st.error("Debug: 'run_backtest_pipeline' not found in backend module. Check file location and name.")
        st.write(f"Files in current directory: {os.listdir('.')}")
    else:
        logging.debug("Debug: 'run_backtest_pipeline' found in backend module.")
        st.write("Debug: 'run_backtest_pipeline' found in backend module.")
except ImportError as e:
    logging.error(f"Import error for backend module: {e}")
    st.error(f"Failed to import backend module: {e}")
except Exception as e:
    logging.error(f"Unexpected error during import check: {e}")
    st.error(f"Unexpected error: {e}")

st.set_page_config(layout="wide", page_title="Quantitative Model Dashboard V12.4")
st.title("Quantitative Model Dashboard V12.4")

tabs = st.tabs(["Live Portfolio", "Strategy Backtest"])

# --- Live Portfolio Tab ---
with tabs[0]:
    st.header("Live Portfolio Allocation")
    available_sectors = backend.get_available_sectors() if hasattr(backend, 'get_available_sectors') else []
    selected_sectors = st.multiselect("Select Sectors (leave empty for all)", available_sectors, default=[])
    max_stock_weight = st.slider("Max Stock Weight", 0.1, 0.5, 0.25, 0.05)
    if st.button("Generate Live Portfolio"):
        with st.spinner("Computing live portfolio..."):
            try:
                portfolio = backend.run_live_prediction_pipeline(
                    st_status=st,
                    selected_sectors=selected_sectors,
                    max_stock_weight=max_stock_weight
                )
                if portfolio is not None and not portfolio.empty:
                    st.subheader("Recommended Weights")
                    st.dataframe(portfolio.style.format("{:.2%}"))
                else:
                    st.error("No valid portfolio generated.")
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"⚠️ Live portfolio generation failed:\n```\n{e}\n{tb}\n```")
                logging.error("Live portfolio error", exc_info=True)

# --- Strategy Backtest Tab ---
with tabs[1]:
    st.header("Full Historical Strategy Backtest")
    st.info("Note: Backtest may take a few minutes due to comprehensive feature engineering.")
    start_date = st.date_input("Backtest Start Date", value=pd.to_datetime("2016-01-01"))
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.1) / 100
    if st.button("Run Full Backtest"):
        with st.spinner("Running walk-forward backtest..."):
            try:
                if hasattr(backend, 'run_backtest_pipeline'):
                    df, metrics = backend.run_backtest_pipeline(
                        st_status=st,
                        start_date=start_date.strftime("%Y-%m-%d")
                    )
                    # Compute additional cumulative series and drawdown
                    df["Cumulative"] = (1 + df["Return"]).cumprod()
                    df["Drawdown"] = 1 - df["Cumulative"] / df["Cumulative"].cummax()
                    metrics["Sharpe"] = ((df["Return"].mean() * 12 - risk_free_rate) /
                                         (df["Return"].std() * np.sqrt(12)))
                    st.success("Backtest complete!")
                    st.subheader("Cumulative Returns")
                    st.line_chart(df["Cumulative"])
                    st.subheader("Drawdown")
                    st.line_chart(df["Drawdown"])
                    st.subheader("Performance Metrics")
                    perf_table = pd.DataFrame({
                        "Metric": ["Annualized Sharpe", "Max Drawdown", "Annual Return"],
                        "Value": [
                            metrics.get("Sharpe", np.nan),
                            metrics.get("MaxDrawdown", np.nan),
                            metrics.get("AnnualReturn", np.nan)
                        ]
                    })
                    st.table(perf_table)
                    logging.info("Backtest completed successfully")
                else:
                    st.error("Debug: 'run_backtest_pipeline' not available in backend module.")
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"⚠️ Backtest failed:\n```\n{e}\n{tb}\n```")
                logging.error("Backtest error", exc_info=True)
