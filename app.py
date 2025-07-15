# ==============================================================================
# V14 - DEFINITIVE APP (REPAIRED)
# ==============================================================================
# This version is simplified to correctly call the memory-efficient backend
# and display the results without performing complex calculations itself.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import backend  # Import our repaired backend module
import traceback

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantitative Model Dashboard V14")
st.title("Quantitative Model Dashboard V14")

# --- Initialize Session State ---
if 'backtest_df' not in st.session_state:
    st.session_state.backtest_df = None
if 'backtest_metrics' not in st.session_state:
    st.session_state.backtest_metrics = None

# --- Main App ---
tabs = st.tabs(["Live Portfolio", "Strategy Backtest"])

# --- Live Portfolio Tab ---
with tabs[0]:
    st.header("Live Portfolio Allocation")
    try:
        available_sectors = backend.get_available_sectors()
        selected_sectors = st.multiselect("Select Sectors (leave empty for all)", available_sectors, default=[])
        max_stock_weight = st.slider("Max Stock Weight", 0.10, 0.50, 0.25, 0.05)
        
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
                        st.dataframe(portfolio.style.format("{:.2%}"), use_container_width=True)
                    else:
                        st.warning("No valid portfolio generated. This might suggest a cautious market outlook.")
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"⚠️ Live portfolio generation failed:\n```\n{e}\n{tb}\n```")
    except Exception as e:
        st.error(f"Could not load sectors. Please ensure your FRED API key is set in secrets.toml. Error: {e}")

# --- Strategy Backtest Tab ---
with tabs[1]:
    st.header("Full Historical Strategy Backtest")
    st.info("Note: Backtest may take a few minutes due to comprehensive feature engineering.")
    
    start_date = st.date_input("Backtest Start Date", value=pd.to_datetime("2016-01-01"))
    risk_free_rate = st.slider("Annual Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.1) / 100
    
    if st.button("Run Full Backtest"):
        with st.spinner("Running walk-forward backtest..."):
            try:
                df, metrics = backend.run_backtest_pipeline(
                    st_status=st,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    risk_free_rate=risk_free_rate
                )
                st.session_state.backtest_df = df
                st.session_state.backtest_metrics = metrics
                st.success("Backtest complete!")
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"⚠️ Backtest failed:\n```\n{e}\n{tb}\n```")

    # Display results if they exist in session state
    if st.session_state.backtest_df is not None and st.session_state.backtest_metrics is not None:
        df = st.session_state.backtest_df
        metrics = st.session_state.backtest_metrics
        
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Annualized Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
        col2.metric("Max Drawdown", f"{metrics.get('MaxDrawdown', 0):.2%}")
        col3.metric("Annual Return", f"{metrics.get('AnnualReturn', 0):.2%}")

        st.subheader("Cumulative Returns")
        st.line_chart(df[['Cumulative', 'SPY_Cumulative']])

        st.subheader("Drawdown")
        st.line_chart(df['Drawdown'])
