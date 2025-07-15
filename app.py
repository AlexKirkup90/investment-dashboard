# ==============================================================================
# V15 - LIVE PREDICTION APP
# ==============================================================================
# This is a lightweight app designed ONLY to generate a live portfolio.
# The heavy backtesting logic has been moved to the Colab Research Notebook.
# ==============================================================================

import streamlit as st
import pandas as pd
import backend  # Import our lightweight backend
import traceback

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Live Portfolio Generator V15")
st.title("Live Portfolio Generator V15")

# --- Main App ---
st.header("Actionable Portfolio for Next Month")
st.markdown("Click the button to generate a portfolio with exact weights. Use this to update your Trading 212 Pie for the upcoming month.")

if st.button("Generate Live Portfolio", type="primary", use_container_width=True):
    with st.spinner("Computing live portfolio..."):
        try:
            portfolio = backend.run_live_prediction_pipeline(st)
            if portfolio is not None and not portfolio.empty:
                st.subheader("Recommended Weights")
                st.dataframe(portfolio.style.format("{:.2%}"), use_container_width=True)
                st.success("✅ Your portfolio is ready. Rebalance your Pie to match these target weights.")
            else:
                st.warning("⚠️ The model recommends holding cash this month (no stocks met the criteria).")
        except Exception as e:
            tb = traceback.format_exc()
            st.error(f"⚠️ Live portfolio generation failed:\n```\n{e}\n{tb}\n```")
