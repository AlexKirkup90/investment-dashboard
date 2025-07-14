# ==============================================================================
# V6 - STREAMLIT USER INTERFACE
# ==============================================================================
# This script creates the web application dashboard for our model.
# To run from your terminal: `streamlit run app.py`
# VERSION 6.1: Added matplotlib import to fix background_gradient error.
# ==============================================================================

import streamlit as st
import pandas as pd
import backend  # Import our backend logic
import matplotlib.pyplot as plt # Explicitly import matplotlib

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantitative Investment Assistant",
    page_icon="📈",
    layout="wide"
)

# --- App State Management ---
# Use session state to store results and avoid re-running on every interaction
if 'latest_predictions' not in st.session_state:
    st.session_state['latest_predictions'] = None
if 'model_run_completed' not in st.session_state:
    st.session_state['model_run_completed'] = False

# --- Sidebar ---
with st.sidebar:
    st.title("📈 Investment Assistant")
    st.markdown("---")
    
    st.header("Actions")
    # The main button to run the model pipeline
    if st.button("Generate Live Watchlist", type="primary", use_container_width=True):
        # Use a status box to show progress
        with st.status("Running model pipeline...", expanded=True) as status:
            try:
                # Call the main function from our backend, passing the status object
                predictions_df = backend.run_prediction_pipeline(status)
                st.session_state['latest_predictions'] = predictions_df
                st.session_state['model_run_completed'] = True
                status.update(label="Pipeline executed successfully!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error", expanded=True)

    st.markdown("---")
    st.info("This dashboard provides AI-driven predictions for S&P 500 stocks. "
            "Click the button above to generate the latest monthly watchlist.")
    st.markdown("Built with the power of Gemini.")

# --- Main Page Content ---
st.title("Quantitative Model Dashboard")
st.markdown("Welcome to your personal investment analysis platform. This tool trains a model on over a decade of market data to generate a forward-looking watchlist.")

if not st.session_state['model_run_completed']:
    st.info("👈 Please click 'Generate Live Watchlist' in the sidebar to begin. The first run may take several minutes to download and cache data.")
else:
    # Display the results once the model has been run
    predictions = st.session_state['latest_predictions']
    
    if predictions is not None and not predictions.empty:
        st.header("Live Prediction Watchlist")
        st.markdown("Top 15 stocks ranked by predicted median return (P50) for the upcoming month.")

        # Create the watchlist from stocks with positive predicted returns
        watchlist = predictions[predictions['Predicted_Return_P50'] > 0].head(15)

        # Display the watchlist table
        st.dataframe(
            watchlist.style.format({
                'Predicted_Return_P50': '{:.2%}'
            }).background_gradient(
                cmap='RdYlGn',
                subset=['Predicted_Return_P50']
            ),
            use_container_width=True
        )

        # Display a chart of the predictions
        st.header("Prediction Visualization")
        
        chart_data = watchlist.set_index('Ticker')
        
        st.bar_chart(
            chart_data,
            y='Predicted_Return_P50'
        )
        st.caption("Chart shows the P50 (median) predicted return for the top watchlist stocks.")

    else:
        st.warning("The model did not generate any positive predictions. This may suggest a cautious market outlook.")

