# src/ui.py

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any

# Import config values
from src import config


def configure_streamlit_page():
    """Configures the Streamlit page settings and applies CSS."""
    st.set_page_config(
        layout="wide",
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON
    )
    # Apply custom CSS for Dark Theme
    st.markdown(config.DARK_THEME_CSS, unsafe_allow_html=True)


def display_sidebar(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the sidebar for user inputs, returning selected parameters."""
    st.sidebar.header("Analysis Parameters")

    # --- Stock Input ---
    company_input = st.sidebar.text_input(
        "Company Name or Stock Symbol",
        defaults.get('company_input', config.DEFAULT_COMPANY_NAME),
        key="sidebar_company_input",  # Add unique keys
        help="Enter the company name (e.g., Apple Inc.) or its stock symbol (e.g., AAPL)."
    )

    st.sidebar.markdown("---")

    # --- Time Range ---
    st.sidebar.subheader("Time Range")
    date_options = {
        '1 Month': 30, '3 Months': 90, '6 Months': 180,
        '1 Year': 365, '2 Years': 730, '5 Years': 1825,
        'Max': 99999, 'Custom': -1
    }
    default_range_key = defaults.get('selected_range_key', '1 Year')
    selected_range_key = st.sidebar.selectbox(
        "Select Period",
        list(date_options.keys()),
        index=list(date_options.keys()).index(default_range_key),
        key="sidebar_period_select",
        help="Select the historical data period for analysis."
    )

    # Calculate default start/end dates based on potentially loaded defaults
    end_date_default = defaults.get(
        'end_date', datetime.today().date())  # Use .date()
    default_days = date_options.get(default_range_key, 365)
    start_date_default = defaults.get('start_date', (datetime.today(
    ) - timedelta(days=default_days if default_days > 0 else 365)).date())

    end_date = datetime.today().date()  # Always end today for presets/max
    start_date = None

    if selected_range_key == 'Custom':
        # Ensure max_value is date, not datetime
        max_start = (datetime.today() - timedelta(days=1)).date()
        max_end = datetime.today().date()

        start_date_input = st.sidebar.date_input(
            "Start Date", start_date_default,
            max_value=max_start,
            key="sidebar_start_date"
        )
        end_date_input = st.sidebar.date_input(
            "End Date", end_date_default,
            max_value=max_end,
            key="sidebar_end_date"
        )

        if start_date_input >= end_date_input:
            st.sidebar.error("Start date must be before end date.")
            start_date = max_start  # Fallback
            end_date = max_end
        else:
            start_date = start_date_input
            end_date = end_date_input
    elif selected_range_key == 'Max':
        # Note: yfinance handles the actual max range, this is just a very early date
        start_date = datetime(1970, 1, 1).date()
    else:
        days = date_options[selected_range_key]
        start_date = (datetime.today() - timedelta(days=days)).date()

    st.sidebar.markdown("---")

    # --- Technical Indicators ---
    st.sidebar.subheader("Technical Indicators")
    tech_indicators = {
        'show_sma': st.sidebar.checkbox("Show SMA", defaults.get('show_sma', True), key="cb_sma"),
        'show_rsi': st.sidebar.checkbox("Show RSI", defaults.get('show_rsi', True), key="cb_rsi"),
        'show_macd': st.sidebar.checkbox("Show MACD", defaults.get('show_macd', True), key="cb_macd"),
        'show_bollinger': st.sidebar.checkbox("Show Bollinger Bands", defaults.get('show_bollinger', True), key="cb_bb")
    }

    st.sidebar.markdown("---")

    st.sidebar.subheader("Sentiment Analysis")
    sentiment_method = st.sidebar.radio(
        "Select Method",
        ('VADER', 'FinBERT'),
        index=0 if defaults.get(
            'sentiment_method', 'VADER') == 'VADER' else 1,  # Default to VADER
        key="radio_sentiment",
        horizontal=True,
        help="Choose the sentiment analysis engine. FinBERT is slower but potentially more accurate for financial news."
    )
    st.sidebar.markdown("---")

    # --- Forecast ---
    st.sidebar.subheader("Forecast")
    prediction_days = st.sidebar.slider(
        "Days to Forecast (Prophet)",
        min_value=7, max_value=365,  # Allow longer forecast
        value=defaults.get('prediction_days', 30),  # Default to 30 days
        step=1, key="slider_forecast_days",
        help="Number of future days to forecast using Prophet."
    )

    st.sidebar.markdown("---")

    # --- Advanced Analyses (using expander) ---
    with st.sidebar.expander("Advanced Analyses", expanded=False):
        run_portfolio = st.checkbox("Portfolio Simulation", value=defaults.get(
            'run_portfolio', False), key="cb_portfolio")
        portfolio_input = defaults.get(
            'portfolio_input', config.DEFAULT_PORTFOLIO_STOCKS)
        initial_investment = defaults.get(
            'initial_investment', config.DEFAULT_INVESTMENT)
        strategy_display = defaults.get(
            'strategy_display', "Equal Weight")  # For display
        strategy_internal = "equal_weight"

        if run_portfolio:
            portfolio_input = st.text_input("Portfolio Symbols", portfolio_input,
                                            key="txt_portfolio_sym", help="Comma-separated (e.g., AAPL,GOOGL,MSFT)")
            initial_investment = st.number_input("Initial Investment ($)", min_value=float(
                config.MIN_INVESTMENT), value=float(initial_investment), step=1000.0, format="%.2f", key="num_investment")
            strategy_options = ["Equal Weight", "Market Cap Weighted"]
            strategy_display = st.selectbox("Strategy", strategy_options, index=strategy_options.index(
                strategy_display), key="sel_strategy")
            strategy_internal = strategy_display.lower().replace(" ", "_")

        run_correlation = st.checkbox("Correlation Analysis", value=defaults.get(
            'run_correlation', False), key="cb_corr")
        correlation_input = defaults.get(
            'correlation_input', config.DEFAULT_CORRELATION_STOCKS)
        if run_correlation:
            correlation_input = st.text_input(
                "Correlation Symbols", correlation_input, key="txt_corr_sym", help="Comma-separated (at least 2)")

        run_ml_section = st.checkbox("Enable ML Predictions", value=defaults.get(
            'run_ml_section', False), key="cb_ml_enable")
        selected_models = []
        if run_ml_section:
            model_options = ["Random Forest", "Linear Regression",
                             "Gradient Boosting"]  # Add more models here
            selected_models = st.multiselect(
                "Select ML Models for Prediction",
                model_options,
                # Default to RF if enabled
                default=defaults.get('selected_models', ["Random Forest"]),
                key="sel_ml_models"
            )

        run_esg = st.checkbox("ESG Performance", value=defaults.get(
            'run_esg', False), key="cb_esg")

        run_economic = st.checkbox(
            "Economic Indicators (FRED)",
            value=defaults.get('run_economic', False),  # Default to False
            key="cb_economic",
            help="Fetch and display relevant macroeconomic indicators from FRED."
        )
        economic_series_input_str = ",".join(
            config.DEFAULT_ECONOMIC_SERIES)  # Get default from config
        if run_economic:
            economic_series_input_str = st.text_input(
                "FRED Series IDs",
                defaults.get('economic_series_input',
                             economic_series_input_str),
                key="txt_economic_series",
                help="Comma-separated FRED series IDs (e.g., GDP,CPIAUCNS,FEDFUNDS)."
            )

    # --- Additional Data Options ---
    with st.sidebar.expander("Additional Data", expanded=True):
        show_dividends = st.checkbox("Dividend History", value=defaults.get(
            'show_dividends', True), key="cb_div")
        # show_earnings = st.checkbox("Earnings Calendar", value=defaults.get('show_earnings', True), key="cb_earn") # Added earnings flag
        show_sector = st.checkbox("Sector Performance", value=defaults.get(
            'show_sector', False), key="cb_sector")  # Default off?

    # --- UI Options ---
    with st.sidebar.expander("UI Options", expanded=False):
        show_tooltips = st.checkbox("Enable Metric Tooltips", defaults.get(
            'show_tooltips', True), key="cb_tooltips")

    st.sidebar.markdown("---")

    # --- Submit Button ---
    submitted = st.sidebar.button(
        "Analyze Stock", key="btn_analyze", use_container_width=True)

    # Return dictionary of parameters
    # Convert dates back to datetime for internal use if needed, although yfinance often handles date objects
    start_dt = datetime.combine(
        start_date, datetime.min.time()) if start_date else None
    end_dt = datetime.combine(
        end_date, datetime.max.time()) if end_date else None  # Use end of day

    return {
        "company_input": company_input,
        "start_date": start_dt,  # Use datetime
        "end_date": end_dt,     # Use datetime
        "selected_range_key": selected_range_key,
        **tech_indicators,
        "prediction_days": prediction_days,
        "run_portfolio": run_portfolio,
        "portfolio_input": portfolio_input,
        "initial_investment": initial_investment,
        "strategy": strategy_internal,
        "strategy_display": strategy_display,
        "run_correlation": run_correlation,
        "correlation_input": correlation_input,
        "run_ml_section": run_ml_section,
        "run_esg": run_esg,
        "show_dividends": show_dividends,
        "selected_models": selected_models,
        # "show_earnings": show_earnings,
        "show_sector": show_sector,
        "show_tooltips": show_tooltips,
        "submitted": submitted,
        "run_economic": run_economic,
        "economic_series_input": economic_series_input_str,
        "submitted": submitted,
        "sentiment_method": sentiment_method,
    }
