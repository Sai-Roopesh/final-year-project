# app.py (Main Entry Point)

# Assuming correct plotting imports from src.plotting
from src.plotting import (
    plot_stock_data, plot_technical_indicators, display_company_info,
    display_sentiment_analysis, display_prophet_forecast, display_portfolio_analysis,
    display_correlation_analysis, display_ml_prediction, display_esg_scores,
    display_earnings_calendar, display_dividend_history, display_sector_performance,
    display_fundamental_data, display_analyst_recommendations, display_economic_indicators
)
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd  # Ensure pandas is imported

# Import necessary components from the src package
from src import config
# Import prepare_llm_context
from src.utils import logger, initialize_api_clients, prepare_llm_context
from src.ui import configure_streamlit_page, display_sidebar
from src.main_analyser import MainAnalyzer

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------


def main():
    # --- Initial Setup ---
    configure_streamlit_page()
    logger.info("--- Streamlit App Session Started ---")
    # Initialize clients once per session using cache
    # newsapi, gemini_client = initialize_api_clients() # This might be better if called once
    # analyzer = MainAnalyzer(newsapi, gemini_client) # Initialize analyzer once
    # Use st.session_state to manage analyzer instance if needed across reruns
    if 'analyzer' not in st.session_state:
        newsapi, gemini_client = initialize_api_clients()
        st.session_state['analyzer'] = MainAnalyzer(newsapi, gemini_client)
    analyzer = st.session_state['analyzer']

    # --- Initialize Session State ---
    default_keys = {
        'last_inputs': {}, 'analysis_results': None, 'analysis_done': False,
        'current_symbol': None, 'current_company_input': None, 'messages': [],
    }
    for key, default_value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Sidebar ---
    input_params = display_sidebar(st.session_state.get('last_inputs', {}))

    # --- Main Area Header ---
    st.markdown(
        f"<div class='gradient-header'><h1>{config.APP_TITLE}</h1></div>", unsafe_allow_html=True)

    # --- Sector Performance ---
    if input_params.get('show_sector', False):
        with st.expander("Sector & S&P 500 Performance Snapshot", expanded=False):
            display_sector_performance()  # This is a static method in plotting

    # --- Process Inputs & Run Analysis ---
    if input_params['submitted']:
        logger.info("Analyze button clicked.")
        # Reset relevant state keys
        st.session_state['analysis_results'] = None
        st.session_state['analysis_done'] = False
        st.session_state['messages'] = []
        st.session_state['current_symbol'] = None
        st.session_state['current_company_input'] = None

        if not input_params['company_input']:
            st.warning("Please enter a company name or stock symbol.")
            st.stop()

        # Store the latest inputs
        st.session_state['last_inputs'] = input_params.copy()

        with st.spinner("Resolving symbol..."):
            # Use the analyzer instance from session state
            symbol = analyzer.data_fetcher.get_stock_symbol(
                input_params['company_input'])

        if not symbol:
            st.error(
                f"Could not find or validate symbol for '{input_params['company_input']}'.")
            logger.error(
                f"Symbol resolution failed for '{input_params['company_input']}'.")
            st.stop()  # Stop execution if symbol not found

        st.session_state['current_symbol'] = symbol
        st.session_state['current_company_input'] = input_params['company_input']
        logger.info(f"Resolved symbol: {symbol}")

        # Prepare arguments for run_full_analysis
        run_flags = {k: v for k, v in input_params.items(
        ) if k.startswith('run_') or k.startswith('show_')}
        # Pass only relevant parameters to advanced_params
        advanced_params_keys = ['prediction_days', 'portfolio_input', 'initial_investment', 'strategy',
                                'correlation_input', 'selected_models', 'economic_series_input', 'sentiment_method']
        advanced_params = {
            k: v for k, v in input_params.items() if k in advanced_params_keys}

        with st.spinner(f"Fetching data and analyzing {symbol}..."):
            # Use the analyzer instance from session state
            analysis_results = analyzer.run_full_analysis(
                symbol,
                input_params['start_date'],
                input_params['end_date'],
                run_flags,
                advanced_params  # Pass the filtered advanced params
            )

        # Check if core data fetching failed within the results
        if analysis_results.get('stock_data') is None:
            # Error message already shown in main_analyzer
            logger.error(
                f"Analysis aborted for {symbol} due to data loading failure.")
            st.session_state['analysis_done'] = False
        else:
            st.session_state['analysis_results'] = analysis_results
            st.session_state['analysis_done'] = True
            logger.info(
                f"Analysis complete for {symbol}. Triggering display rerun.")
            st.rerun()  # Rerun to update the display area

    # --- Display Results Area ---
    if st.session_state.get('analysis_done', False):
        results = st.session_state.get('analysis_results', {})
        symbol = st.session_state.get('current_symbol', 'N/A')
        company_input_display = st.session_state.get(
            'current_company_input', symbol)

        # Verify results seem valid for the current symbol
        if not results or results.get('symbol') != symbol:
            st.warning(
                "Analysis results missing or mismatched. Please analyze again.")
            st.stop()

        # --- ADDED: Get currency from results ---
        # Default to USD if missing
        currency_code = results.get('currency', 'USD')
        # --- END ADDED ---

        st.header(f"Analysis Results for: {company_input_display} ({symbol})")
        logger.info(f"Displaying analysis results for {symbol}")

        # Retrieve necessary parameters from last_inputs or defaults
        last_inputs = st.session_state.get('last_inputs', {})
        indicator_toggles = {k: last_inputs.get(k, False) for k in [
            'show_sma', 'show_rsi', 'show_macd', 'show_bollinger']}
        show_tooltips = last_inputs.get('show_tooltips', True)
        days_predicted = last_inputs.get('prediction_days', 30)
        initial_investment_disp = last_inputs.get(
            'initial_investment', config.DEFAULT_INVESTMENT)

        # --- Define Tabs ---
        tab_titles = ["📊 Overview & Chart",
                      "🪙 Fundamentals", "📰 News & Sentiment"]
        # Check conditions for optional tabs based on data existence in results
        economic_data_for_tab = results.get('economic_data')
        show_economic_tab = economic_data_for_tab is not None and not economic_data_for_tab.empty
        prophet_data_for_tab = results.get('prophet_forecast_data')
        show_forecast_tab = prophet_data_for_tab is not None and not prophet_data_for_tab.empty
        ml_data_for_tab = results.get('ml_results')
        show_ml_tab = ml_data_for_tab is not None  # Show tab if results dict exists
        dividend_data_for_tab = results.get('dividend_history')
        earnings_data_for_tab = results.get('earnings_calendar')
        show_financials_tab = (dividend_data_for_tab is not None and not dividend_data_for_tab.empty and last_inputs.get('show_dividends', True)) or \
                              (earnings_data_for_tab is not None and not earnings_data_for_tab.empty)

        optional_tabs_config = {
            "🗣️ Analyst Ratings": results.get('analyst_info') is not None,
            "📈 Forecast": show_forecast_tab,
            "💼 Portfolio Sim": results.get('portfolio_result') is not None,
            "🔗 Correlation": results.get('correlation_matrix') is not None and not results.get('correlation_matrix').empty,
            "🤖 ML Prediction": show_ml_tab,
            "🌱 ESG": results.get('esg_scores') is not None,
            "🏛️ Economic Data": show_economic_tab,
            "💰 Financials (Div/Earn)": show_financials_tab
        }
        enabled_tabs = [title for title,
                        enabled in optional_tabs_config.items() if enabled]
        all_tab_titles = tab_titles + enabled_tabs
        tabs = st.tabs(all_tab_titles)
        tab_map = {title: tab for title, tab in zip(all_tab_titles, tabs)}

        # --- Populate Tabs ---
        with tab_map["📊 Overview & Chart"]:
            # --- MODIFIED: Pass currency_code ---
            display_company_info(results.get('stock_info'),
                                 show_tooltips, currency_code=currency_code)
            st.markdown("---")
            plot_stock_data(results.get('tech_data'), symbol, indicator_toggles, results.get(
                'patterns', []), currency_code=currency_code)
            st.markdown("---")
            plot_technical_indicators(
                results.get('tech_data'), indicator_toggles)
            # --- END MODIFIED ---
        with tab_map["🪙 Fundamentals"]:
            # display_fundamental_data doesn't need currency code currently
            display_fundamental_data(results.get('financials'), results.get(
                'balance_sheet'), results.get('cash_flow'))
        with tab_map["📰 News & Sentiment"]:
            # Pass both sets of news results
            display_sentiment_analysis(
                results.get('newsapi_analyzed_articles', []),
                results.get('newsapi_avg_sentiment', 0.0),
                results.get('newsapi_sentiment_df'),
                results.get('yfinance_analyzed_articles', []),
                results.get('yfinance_avg_sentiment', 0.0),
                results.get('yfinance_sentiment_df'),
                results.get('sentiment_method_used', 'vader')
            )

        # Populate Optional Tabs
        if "🗣️ Analyst Ratings" in tab_map:
            with tab_map["🗣️ Analyst Ratings"]:
                # --- MODIFIED: Pass currency_code ---
                display_analyst_recommendations(results.get(
                    'analyst_info'), currency_code=currency_code)
                # --- END MODIFIED ---
        if "📈 Forecast" in tab_map:
            with tab_map["📈 Forecast"]:
                # --- MODIFIED: Pass currency_code ---
                display_prophet_forecast(results.get('prophet_forecast_data'), results.get(
                    'stock_data'), symbol, days_predicted, currency_code=currency_code)
                # --- END MODIFIED ---
        if "💼 Portfolio Sim" in tab_map:
            with tab_map["💼 Portfolio Sim"]:
                # Portfolio assumes USD for now
                display_portfolio_analysis(results.get(
                    'portfolio_result'), initial_investment_disp)
        if "🔗 Correlation" in tab_map:
            with tab_map["🔗 Correlation"]:
                display_correlation_analysis(results.get('correlation_matrix'))
        if "🤖 ML Prediction" in tab_map:
            with tab_map["🤖 ML Prediction"]:
                # --- MODIFIED: Pass currency_code ---
                display_ml_prediction(results.get(
                    'ml_results'), symbol, currency_code=currency_code)
                # --- END MODIFIED ---
        if "🌱 ESG" in tab_map:
            with tab_map["🌱 ESG"]:
                display_esg_scores(results.get('esg_scores'), symbol)
        if "🏛️ Economic Data" in tab_map:
            with tab_map["🏛️ Economic Data"]:
                logger.info("Populating Economic Data tab.")
                display_economic_indicators(results.get(
                    'economic_data'), results.get('stock_data'))
        if "💰 Financials (Div/Earn)" in tab_map:
            with tab_map["💰 Financials (Div/Earn)"]:
                show_div_flag = last_inputs.get('show_dividends', True)
                dividend_data = results.get('dividend_history')
                earnings_data = results.get('earnings_calendar')
                dividend_shown = False
                if dividend_data is not None and not dividend_data.empty and show_div_flag:
                    # --- MODIFIED: Pass currency_code ---
                    display_dividend_history(
                        dividend_data, symbol, currency_code=currency_code)
                    # --- END MODIFIED ---
                    dividend_shown = True
                if earnings_data is not None and not earnings_data.empty:
                    if dividend_shown:
                        st.markdown("---")
                     # --- MODIFIED: Pass currency_code ---
                    display_earnings_calendar(
                        earnings_data, symbol, currency_code=currency_code)
                    # --- END MODIFIED ---
                # Add case where neither is available/shown
                if not dividend_shown and (earnings_data is None or earnings_data.empty):
                    st.info("No Dividend or Earnings data available/selected.")

        # --- AI Chat Assistant ---
        st.markdown("---")
        st.subheader("💬 AI Financial Assistant")
        # Use analyzer instance from session state
        if not analyzer.gemini_client:
            st.warning(
                "AI Assistant is unavailable (Gemini client not configured).")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input(f"Ask about {symbol} based on analyzed data...", key="chat_input"):
                st.session_state.messages.append(
                    {"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Thinking..."):
                        # Pass the full results dictionary to the analyzer's method
                        response = analyzer.generate_chat_response(
                            prompt, results)
                        st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})

    # Initial state or after reset
    elif not st.session_state.get('analysis_done'):
        st.info(
            "👋 Welcome! Configure parameters in the sidebar and click 'Analyze Stock'.")


# --- Run Main ---
if __name__ == "__main__":
    try:
        main()
    except Exception as main_err:
        logger.critical(
            f"Unhandled critical exception in main: {main_err}", exc_info=True)
        try:
            st.error(f"A critical application error occurred: {main_err}")
        except Exception as display_err:
            logger.error(
                f"Failed to display critical error in Streamlit UI: {display_err}")
