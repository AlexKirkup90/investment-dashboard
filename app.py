# ==============================================================================
# V40 - ULTIMATE MODEL LIVE PREDICTION APP
# ==============================================================================
# This app generates a live portfolio based on the final, high-performing
# model (Sharpe 1.33) developed through our rigorous research process.
# ==============================================================================

import streamlit as st
import pandas as pd
import backend  # Import our final backend
import traceback

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Ultimate Model Portfolio Generator")
st.title("🚀 Ultimate Model Portfolio Generator (V40)")

# --- Main App ---
st.header("Actionable Portfolio for the Upcoming Month")
st.markdown("""
Click the button below to generate a portfolio with exact weights based on our definitive **Ultimate Model**.

This is a highly advanced quantitative system that incorporates:
- **An Upgraded XGBoost Engine:** For superior predictive power.
- **Advanced Feature Engineering:** Including composite momentum and non-linear interactions.
- **Dynamic Risk Management:** An adaptive system that adjusts leverage, portfolio size, and hedging based on real-time market conditions (VIX, SPY trend).

The final portfolio represents the model's highest-conviction, risk-managed allocation for the month ahead.
""")

if st.button("Generate Live Portfolio", type="primary", use_container_width=True):
    # Create a status container for real-time updates
    status = st.empty()
    with st.spinner(""): # Use an empty spinner and rely on the status text
        try:
            portfolio = backend.run_live_prediction_pipeline(status)
            if portfolio is not None and not portfolio.empty:
                status.empty() # Clear the status text
                st.subheader("Recommended Portfolio Weights")
                st.dataframe(portfolio.style.format("{:.2%}"), use_container_width=True)
                st.success("✅ Your portfolio is ready. Rebalance your Pie to match these target weights.")
            else:
                status.empty() # Clear the status text
                st.warning("⚠️ The model's risk management system recommends holding cash this month (no stocks met the strict selection criteria).")
        except Exception as e:
            status.empty() # Clear the status text
            tb = traceback.format_exc()
            st.error(f"⚠️ Live portfolio generation failed:\n```\n{e}\n{tb}\n```")
