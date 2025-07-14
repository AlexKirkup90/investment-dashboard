# ==============================================================================
# app.py — Streamlit Dashboard
# ==============================================================================

import streamlit as st
import pandas as pd
import backend  # our complete backend module

st.set_page_config(layout="wide", page_title="Quantitative Model Dashboard")
st.title("Quantitative Model Dashboard")

tabs = st.tabs(["Live Portfolio", "Strategy Backtest"])

# --- Live Portfolio Tab ---
with tabs[0]:
    st.header("Live Portfolio Allocation")
    if st.button("Generate Live Portfolio"):
        with st.spinner("Computing live portfolio..."):
            portfolio = backend.run_live_prediction_pipeline()
        if portfolio is not None and not portfolio.empty:
            st.subheader("Recommended Weights")
            st.dataframe(portfolio)
        else:
            st.error("No valid portfolio generated.")

# --- Strategy Backtest Tab ---
with tabs[1]:
    st.header("Full Historical Strategy Backtest")
    st.info("Note: This is slow — it walks through every month since 2016.")
    if st.button("Run Full Backtest"):
        with st.spinner("Running walk-forward backtest..."):
            try:
                df = backend.run_backtest_pipeline()
                st.success("Backtest complete!")
                st.line_chart(df["Cumulative"])
                metrics = {
                    "Annualized Sharpe": df["Return"].mean()*12/df["Return"].std()* (12**0.5)
                }
                st.json(metrics)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
