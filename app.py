# app.py (Main Entry Point)

import streamlit as st
from datetime import datetime, timedelta

# Import necessary components from the src package
from src import config
from src.utils import logger, initialize_api_clients, prepare_llm_context
from src.ui import configure_streamlit_page, display_sidebar
from src.main_analyser import MainAnalyzer  # Import the main coordinator class
# Import all display functions from plotting, including the new ones
from src.plotting import (
    plot_stock_data,  # Handles Price/Volume/SMA/BB
    plot_technical_indicators,  # NEW: Handles RSI/MACD
    display_company_info, display_sentiment_analysis,
    display_prophet_forecast, display_portfolio_analysis, display_correlation_analysis,
    display_ml_prediction, display_esg_scores, display_earnings_calendar,
    display_dividend_history, display_sector_performance,
    display_fundamental_data, display_analyst_recommendations, display_economic_indicators
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
    default_keys = {
        'last_inputs': {}, 'analysis_results': None, 'analysis_done': False,
        'current_symbol': None, 'current_company_input': None, 'messages': [],
        # Keep specific data keys if needed for context prep, otherwise rely on analysis_results
        # 'financials': None, 'balance_sheet': None, 'cash_flow': None,
        # 'analyst_info': None, 'economic_data': None
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
    if input_params.get('show_sector', False):
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
        # Clear specific data keys if they exist
        # for key in ['financials', 'balance_sheet', 'cash_flow', 'analyst_info', 'economic_data']:
        #     if key in st.session_state: st.session_state[key] = None

        if not input_params['company_input']:
            st.warning("Please enter a company name or stock symbol.")
            st.stop()

        st.session_state['last_inputs'] = input_params.copy()

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

        run_flags = {k: v for k, v in input_params.items(
        ) if k.startswith('run_') or k.startswith('show_')}
        advanced_params = {
            'prediction_days': input_params['prediction_days'],
            'portfolio_input': input_params['portfolio_input'],
            'initial_investment': input_params['initial_investment'],
            'strategy': input_params['strategy'],
            'correlation_input': input_params['correlation_input'],
            'selected_models': input_params['selected_models'],
            'economic_series_input': input_params['economic_series_input']
        }

        with st.spinner(f"Fetching data and analyzing {symbol}..."):
            analysis_results = analyzer.run_full_analysis(
                symbol, input_params['start_date'], input_params['end_date'],
                run_flags, advanced_params
            )

        if analysis_results.get('stock_data') is None:
            logger.error(
                f"Analysis aborted for {symbol} due to data loading failure.")
            st.session_state['analysis_done'] = False
        else:
            st.session_state['analysis_results'] = analysis_results
            # No need to store individual keys if analysis_results holds everything
            st.session_state['analysis_done'] = True
            logger.info(
                f"Analysis complete for {symbol}. Triggering display rerun.")
            st.rerun()

    # --- Display Results Area ---
    if st.session_state.get('analysis_done', False):
        results = st.session_state.get('analysis_results', {})
        symbol = st.session_state.get('current_symbol', 'N/A')
        company_input_display = st.session_state.get(
            'current_company_input', symbol)

        if not results or results.get('symbol') != symbol:
            st.warning(
                "Analysis results seem outdated or missing. Please click 'Analyze Stock' again.")
            st.stop()

        st.header(f"Analysis Results for: {company_input_display} ({symbol})")
        logger.info(f"Displaying analysis results for {symbol}")

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
        economic_data_for_tab = results.get('economic_data')
        show_economic_tab = economic_data_for_tab is not None and not economic_data_for_tab.empty
        logger.info(
            f"Checking tab condition for Economic Data: Data is None? {economic_data_for_tab is None}, Is Empty? {economic_data_for_tab.empty if economic_data_for_tab is not None else 'N/A'}, Show Tab? {show_economic_tab}")

        optional_tabs_config = {
            "🗣️ Analyst Ratings": results.get('analyst_info') is not None,
            "📈 Forecast": results.get('prophet_forecast_data') is not None and not results.get('prophet_forecast_data').empty,
            "💼 Portfolio Sim": results.get('portfolio_result') is not None,
            "🔗 Correlation": results.get('correlation_matrix') is not None and not results.get('correlation_matrix').empty,
            "🤖 ML Prediction": results.get('ml_results') is not None,
            "🌱 ESG": results.get('esg_scores') is not None,
            "🏛️ Economic Data": show_economic_tab,
            "💰 Financials (Div/Earn)": (
                (results.get('dividend_history') is not None and not results.get('dividend_history').empty and last_inputs.get('show_dividends', True)) or
                (results.get('earnings_calendar')
                 is not None and not results.get('earnings_calendar').empty)
            )
        }
        enabled_tabs = [title for title,
                        enabled in optional_tabs_config.items() if enabled]
        all_tab_titles = tab_titles + enabled_tabs
        tabs = st.tabs(all_tab_titles)
        tab_map = {title: tab for title, tab in zip(all_tab_titles, tabs)}

        # --- Populate Tabs ---

        # Tab 1: Overview & Chart
        with tab_map["📊 Overview & Chart"]:
            display_company_info(results.get('stock_info'), show_tooltips)
            st.markdown("---")
            # Call main price chart function
            plot_stock_data(results.get('tech_data'), symbol,
                            indicator_toggles, results.get('patterns', []))
            st.markdown("---")  # Add separator
            # Call the NEW technical indicators function
            plot_technical_indicators(
                results.get('tech_data'), indicator_toggles)

        # Tab 2: Fundamentals
        with tab_map["🪙 Fundamentals"]:
            display_fundamental_data(
                results.get('financials'),
                results.get('balance_sheet'),
                results.get('cash_flow')
            )

        # Tab 3: News & Sentiment
        with tab_map["📰 News & Sentiment"]:
            display_sentiment_analysis(results.get('analyzed_articles', []), results.get(
                'avg_sentiment', 0.0), results.get('sentiment_df'))

        # --- Populate Optional Tabs ---
        # (Analyst, Forecast, Portfolio, Correlation, ML, ESG - no changes needed here)
        if "🗣️ Analyst Ratings" in tab_map:
            with tab_map["🗣️ Analyst Ratings"]:
                display_analyst_recommendations(results.get('analyst_info'))
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

        # Optional Tab: Economic Data
        if "🏛️ Economic Data" in tab_map:
            with tab_map["🏛️ Economic Data"]:
                logger.info("Populating Economic Data tab.")
                display_economic_indicators(
                    results.get('economic_data'),
                    results.get('stock_data')
                )

        # Optional Tab: Financials (Dividends/Earnings)
        if "💰 Financials (Div/Earn)" in tab_map:
            with tab_map["💰 Financials (Div/Earn)"]:
                show_div_flag = last_inputs.get('show_dividends', True)
                dividend_data = results.get('dividend_history')
                earnings_data = results.get('earnings_calendar')
                dividend_shown = False
                if dividend_data is not None and not dividend_data.empty and show_div_flag:
                    display_dividend_history(dividend_data, symbol)
                    dividend_shown = True
                if earnings_data is not None and not earnings_data.empty:
                    if dividend_shown:
                        st.markdown("---")
                    display_earnings_calendar(earnings_data, symbol)
                if not (dividend_data is not None and not dividend_data.empty and show_div_flag) and \
                   not (earnings_data is not None and not earnings_data.empty):
                    st.info(
                        "No Dividend or Earnings data available/selected to display.")

        # --- AI Chat Assistant ---
        st.markdown("---")
        st.subheader("💬 AI Financial Assistant")
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
                        # Prepare context using the utility function - pass results dict
                        context_for_llm = prepare_llm_context(results)
                        response = analyzer.generate_chat_response(
                            prompt, context_for_llm)
                        st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})

    elif not st.session_state.get('analysis_done'):
        st.info(
            "👋 Welcome! Configure parameters in the sidebar and click 'Analyze Stock'.")


# --- Run Main ---
if __name__ == "__main__":
    try:
        main()
        logger.info("--- Streamlit App Session Finished Normally ---")
    except Exception as main_err:
        logger.critical(
            f"Unhandled critical exception in main: {main_err}", exc_info=True)
        try:
            st.error(
                f"A critical application error occurred. Please check logs or restart. Error: {main_err}")
        except Exception as display_err:
            logger.error(
                f"Failed to display critical error in Streamlit UI: {display_err}")
