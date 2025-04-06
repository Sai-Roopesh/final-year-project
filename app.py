# app.py (Main Entry Point)

import streamlit as st
from datetime import datetime, timedelta

# Import necessary components from the src package
from src import config
from src.utils import logger, initialize_api_clients, prepare_llm_context
from src.ui import configure_streamlit_page, display_sidebar
from src.main_analyser import MainAnalyzer  # Import the main coordinator class
from src.plotting import (  # Import all display functions
    plot_stock_data, display_company_info, display_sentiment_analysis,
    display_prophet_forecast, display_portfolio_analysis, display_correlation_analysis,
    display_ml_prediction, display_esg_scores, display_earnings_calendar,
    display_dividend_history, display_sector_performance
)

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------


def main():
    # --- Initial Setup ---
    configure_streamlit_page()
    logger.info("--- Streamlit App Session Started ---")
    newsapi, gemini_client = initialize_api_clients()  # Cached resource

    # Instantiate the main analyzer (which initializes sub-components)
    analyzer = MainAnalyzer(newsapi, gemini_client)

    # --- Initialize Session State ---
    # Ensures keys exist on first run or after a full refresh
    default_keys = {
        'last_inputs': {}, 'analysis_results': None, 'analysis_done': False,
        'current_symbol': None, 'current_company_input': None, 'messages': []
    }
    for key, default_value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Sidebar ---
    input_params = display_sidebar(st.session_state.get('last_inputs', {}))

    # --- Main Area Header ---
    st.markdown(
        f"<div class='gradient-header'><h1>{config.APP_TITLE}</h1></div>", unsafe_allow_html=True)

    # --- Sector Performance (Optional Display) ---
    if input_params['show_sector']:
        with st.expander("Sector & S&P 500 Performance Snapshot", expanded=False):
            display_sector_performance()

    # --- Process Inputs & Run Analysis on Button Click ---
    if input_params['submitted']:
        logger.info("Analyze button clicked.")
        # Clear previous results and chat, keep inputs
        st.session_state['analysis_results'] = None
        st.session_state['analysis_done'] = False
        st.session_state['messages'] = []
        st.session_state['current_symbol'] = None
        st.session_state['current_company_input'] = None

        if not input_params['company_input']:
            st.warning("Please enter a company name or stock symbol.")
            st.stop()

        # Store current inputs for next run's defaults
        st.session_state['last_inputs'] = input_params.copy()

        # Resolve symbol first
        with st.spinner("Resolving symbol..."):
            symbol = analyzer.data_fetcher.get_stock_symbol(
                input_params['company_input'])

        if not symbol:
            st.error(
                f"Could not find or validate symbol for '{input_params['company_input']}'.")
            logger.error(
                f"Symbol resolution failed for '{input_params['company_input']}'.")
            st.stop()

        st.session_state['current_symbol'] = symbol
        st.session_state['current_company_input'] = input_params['company_input']
        logger.info(f"Resolved symbol: {symbol}")

        # Prepare parameters for the main analysis run
        run_flags = {k: v for k, v in input_params.items(
        ) if k.startswith('run_') or k.startswith('show_')}
        advanced_params = {
            'prediction_days': input_params['prediction_days'],
            'portfolio_input': input_params['portfolio_input'],
            'initial_investment': input_params['initial_investment'],
            'strategy': input_params['strategy'],
            'correlation_input': input_params['correlation_input'],
        }

        # --- Execute Full Analysis Pipeline ---
        with st.spinner(f"Fetching data and analyzing {symbol}..."):
            analysis_results = analyzer.run_full_analysis(
                symbol, input_params['start_date'], input_params['end_date'],
                run_flags, advanced_params
            )

        # Check if core data fetching was successful within the results
        if analysis_results.get('stock_data') is None:
            # Error message was already shown by data_fetcher or run_full_analysis
            logger.error(
                f"Analysis aborted for {symbol} due to data loading failure.")
            st.session_state['analysis_done'] = False  # Ensure flag is false
        else:
            st.session_state['analysis_results'] = analysis_results
            st.session_state['analysis_done'] = True
            logger.info(
                f"Analysis complete for {symbol}. Triggering display rerun.")
            st.rerun()  # Rerun to display results cleanly

    # --- Display Results Area ---
    if st.session_state.get('analysis_done', False) and st.session_state.get('analysis_results'):
        results = st.session_state['analysis_results']
        symbol = st.session_state.get('current_symbol', 'N/A')
        company_input_display = st.session_state.get(
            'current_company_input', symbol)
        st.header(f"Analysis Results for: {company_input_display} ({symbol})")
        logger.info(f"Displaying analysis results for {symbol}")

        # Retrieve needed params from last inputs for display functions
        last_inputs = st.session_state.get('last_inputs', {})
        indicator_toggles = {k: last_inputs.get(k, False) for k in [
            'show_sma', 'show_rsi', 'show_macd', 'show_bollinger']}
        show_tooltips = last_inputs.get('show_tooltips', True)
        days_predicted = last_inputs.get('prediction_days', 30)
        initial_investment_disp = last_inputs.get(
            'initial_investment', config.DEFAULT_INVESTMENT)

        # --- Tabs ---
        tab_titles = ["📊 Overview & Chart", "📰 News & Sentiment"]
        enabled_tabs = []
        if results.get('prophet_forecast_data') is not None:
            enabled_tabs.append("📈 Forecast")
        if results.get('portfolio_result') is not None:
            enabled_tabs.append("💼 Portfolio Sim")
        if results.get('correlation_matrix') is not None:
            enabled_tabs.append("🔗 Correlation")
        if results.get('ml_results') is not None:
            enabled_tabs.append("🤖 ML Prediction")
        if results.get('esg_scores') is not None:
            # Show tab even if scores are empty dict
            enabled_tabs.append("🌱 ESG")
        if results.get('dividend_history') is not None or results.get('earnings_calendar') is not None:
            enabled_tabs.append("💰 Financials")

        tabs = st.tabs(tab_titles + enabled_tabs)
        tab_map = {title: tab for title, tab in zip(
            tab_titles + enabled_tabs, tabs)}

        # Populate Tabs
        with tab_map["📊 Overview & Chart"]:
            display_company_info(results.get('stock_info'), show_tooltips)
            st.markdown("---")
            plot_stock_data(results.get('tech_data'), symbol,
                            indicator_toggles, results.get('patterns', []))

        with tab_map["📰 News & Sentiment"]:
            display_sentiment_analysis(results.get('analyzed_articles', []), results.get(
                'avg_sentiment', 0.0), results.get('sentiment_df'))

        if "📈 Forecast" in tab_map:
            with tab_map["📈 Forecast"]:
                display_prophet_forecast(results.get('prophet_forecast_data'), results.get(
                    'stock_data'), symbol, days_predicted)
        if "💼 Portfolio Sim" in tab_map:
            with tab_map["💼 Portfolio Sim"]:
                display_portfolio_analysis(results.get(
                    'portfolio_result'), initial_investment_disp)
        if "🔗 Correlation" in tab_map:
            with tab_map["🔗 Correlation"]:
                display_correlation_analysis(results.get('correlation_matrix'))
        if "🤖 ML Prediction" in tab_map:
            with tab_map["🤖 ML Prediction"]:
                display_ml_prediction(results.get('ml_results'), symbol)
        if "🌱 ESG" in tab_map:
            with tab_map["🌱 ESG"]:
                display_esg_scores(results.get('esg_scores'), symbol)
        if "💰 Financials" in tab_map:
            with tab_map["💰 Financials"]:
                if results.get('dividend_history') is not None and last_inputs.get('show_dividends'):
                    display_dividend_history(
                        results.get('dividend_history'), symbol)
                    st.markdown("---")
                if results.get('earnings_calendar') is not None:
                    display_earnings_calendar(
                        results.get('earnings_calendar'), symbol)
                # Add message if both are None/not shown
                if not (results.get('dividend_history') is not None and last_inputs.get('show_dividends')) and \
                   not results.get('earnings_calendar'):
                    st.info("No Dividend or Earnings data available/selected.")

        # --- AI Chat Assistant ---
        st.markdown("---")
        st.subheader("💬 AI Financial Assistant")
        if not analyzer.gemini_client:  # Check if client is available
            st.warning(
                "AI Assistant is unavailable (Gemini client not configured).")
        else:
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input(f"Ask about {symbol} based on analyzed data...", key="chat_input"):
                st.session_state.messages.append(
                    {"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("🧠 Thinking..."):
                        # Prepare context using only the analysis results we have
                        context_for_llm = prepare_llm_context(
                            st.session_state)  # Use results from session_state
                        response = analyzer.generate_chat_response(
                            prompt, context_for_llm)
                        st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})

    # --- Initial Welcome Message ---
    # Show only if no analysis is done/displayed
    elif not st.session_state.get('analysis_done'):
        st.info(
            "👋 Welcome! Configure parameters in the sidebar and click 'Analyze Stock'.")


# --- Run Main ---
if __name__ == "__main__":
    try:
        main()
        logger.info("--- Streamlit App Session Finished Normally ---")
    except Exception as main_err:
        # Log critical errors that weren't caught elsewhere
        logger.critical(
            f"Unhandled critical exception in main: {main_err}", exc_info=True)
        try:  # Try to display error in Streamlit if possible
            st.error(
                f"A critical application error occurred. Please check logs or restart. Error: {main_err}")
            # st.exception(main_err) # Use this for debugging - shows full traceback
        except Exception as display_err:
            logger.error(
                f"Failed to display critical error in Streamlit UI: {display_err}")
