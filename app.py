# ==============================================================================
# V37 - ENSEMBLE MODEL LIVE PREDICTION APP
# ==============================================================================
# This app generates a live portfolio based on the final, high-performing
# Ensemble Model developed through our rigorous research process.
# ==============================================================================

import streamlit as st
import pandas as pd
import backend  # Import our final backend
import traceback

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Ensemble Model Portfolio Generator")
st.title("📈 Ensemble Model Portfolio Generator (V37)")

# --- Main App ---
st.header("Actionable Portfolio for the Upcoming Month")
st.markdown("""
Click the button below to generate a portfolio with exact weights based on our definitive **Ensemble Model**.

This model combines the predictions of two specialist strategies:
1.  **The High-Return Champion (Sharpe 1.17):** Focuses on maximizing risk-adjusted returns.
2.  **The Low-Risk Champion (Max Drawdown 15.79%):** Focuses on stability and capital preservation.

The final portfolio represents the consensus view of these two models. Use these weights to rebalance your investment pie for the month ahead.
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
                st.warning("⚠️ The model recommends holding cash this month (no stocks met the selection criteria).")
        except Exception as e:
            status.empty() # Clear the status text
            tb = traceback.format_exc()
            st.error(f"⚠️ Live portfolio generation failed:\n```\n{e}\n{tb}\n```")
