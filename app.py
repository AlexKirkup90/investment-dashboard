import streamlit as st
import pandas as pd
import backend  # V12 backend module
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.info(f"App started at {pd.Timestamp.now()}")

st.set_page_config(layout="wide", page_title="Quantitative Model Dashboard V12")
st.title("Quantitative Model Dashboard V12")

tabs = st.tabs(["Live Portfolio", "Strategy Backtest"])

# --- Live Portfolio Tab ---
with tabs[0]:
    st.header("Live Portfolio Allocation")
    selected_sectors = st.multiselect(
        "Select Sectors (leave empty for all)",
        backend.get_available_sectors(),
        default=[]
    )
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
                    # Bar chart for weights
                    st.subheader("Portfolio Allocation")
                    chart_data = {
                        "type": "bar",
                        "data": {
                            "labels": portfolio.index.tolist(),
                            "datasets": [{
                                "label": "Portfolio Weights",
                                "data": portfolio["Weight"].tolist(),
                                "backgroundColor": ["#1f77b4"] * len(portfolio),
                                "borderColor": ["#1f77b4"] * len(portfolio),
                                "borderWidth": 1
                            }]
                        },
                        "options": {
                            "scales": {
                                "y": {"beginAtZero": True, "title": {"display": True, "text": "Weight"}},
                                "x": {"title": {"display": True, "text": "Stocks"}}
                            }
                        }
                    }
                    st.json(chart_data)  # Render as Chart.js bar chart
                else:
                    st.error("No valid portfolio generated.")
            except Exception as e:
                st.error(f"Portfolio generation failed: {e}")
                logging.error(f"Live portfolio error: {e}")

# --- Strategy Backtest Tab ---
with tabs[1]:
    st.header("Full Historical Strategy Backtest")
    st.info("Note: Backtest may take a few minutes due to comprehensive feature engineering.")
    start_date = st.date_input("Backtest Start Date", value=pd.to_datetime("2016-01-01"))
    if st.button("Run Full Backtest"):
        with st.spinner("Running walk-forward backtest..."):
            try:
                df, metrics = backend.run_backtest_pipeline(st_status=st, start_date=start_date)
                st.success("Backtest complete!")
                st.subheader("Cumulative Returns")
                st.line_chart(df["Cumulative"])
                st.subheader("Performance Metrics")
                st.table({
                    "Metric": ["Annualized Sharpe (4% Risk-Free)", "Max Drawdown", "Annual Return"],
                    "Value": [
                        metrics["Sharpe"],
                        metrics["MaxDrawdown"],
                        metrics["AnnualReturn"]
                    ]
                })
                logging.info("Backtest completed successfully")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                logging.error(f"Backtest error: {e}")
