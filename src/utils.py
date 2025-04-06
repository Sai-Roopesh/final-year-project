# src/utils.py

from typing import Any  # Ensure this is imported at the top if not already
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import streamlit as st
from newsapi import NewsApiClient
from openai import OpenAI
import pandas as pd  # Added for LLM context prep improvements

# Import config values AFTER defining them in config.py
from src import config

# --- Logging Setup ---


def setup_logging() -> logging.Logger:
    """Sets up logging with rotating file and console handlers."""
    log_dir = config.LOG_DIRECTORY
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Warning: Could not create log directory '{log_dir}': {e}")
            # Fallback or handle error appropriately if logging is critical

    log_filename = datetime.now().strftime(
        f"{log_dir}/app_log_{config.DATE_FORMAT}.log")
    logger = logging.getLogger("StockAppLogger")
    if logger.hasHandlers():
        logger.handlers.clear()
    # Consider changing level (e.g., DEBUG) via config or env var
    logger.setLevel(logging.INFO)

    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to attach file logger: {e}")

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)  # Always add stream handler

    return logger


# --- Initialize Logger ---
logger = setup_logging()

# --- API Client Initialization ---


# Cache resources
@st.cache_resource(show_spinner="Initializing API connections...")
def initialize_api_clients() -> Tuple[Optional[NewsApiClient], Optional[OpenAI]]:
    """Initializes and validates API clients based on config."""
    newsapi_client = None
    gemini_client = None

    # NewsAPI
    if not config.NEWSAPI_KEY:
        logger.error("NewsAPI Key not found.")
        # Don't use st.error here directly, let main app handle UI feedback
    else:
        try:
            newsapi_client = NewsApiClient(api_key=config.NEWSAPI_KEY)
            logger.info("NewsAPI client initialized.")
        except Exception as e:
            logger.exception("Error initializing NewsAPI client.")
            newsapi_client = None  # Ensure it's None on failure

    # Gemini (via OpenAI library)
    if not config.GEMINI_API_KEY:
        logger.error("Gemini API Key not found.")
    else:
        try:
            gemini_client = OpenAI(
                api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)
            # Optional: Test connection (can be slow, might remove in production)
            # try:
            #     gemini_client.models.list()
            #     logger.info("Gemini client initialized and connection tested.")
            # except Exception as test_e:
            #     logger.warning(f"Gemini connection test failed: {test_e}. Client might still work partially.")
            logger.info(
                "Gemini client potentially initialized (connection test skipped).")
        except Exception as e:
            logger.exception("Error initializing Gemini client.")
            gemini_client = None  # Ensure it's None on failure

    # Log final status
    if not newsapi_client:
        logger.warning("NewsAPI client UNAVAILABLE.")
    if not gemini_client:
        logger.warning("Gemini client UNAVAILABLE.")

    return newsapi_client, gemini_client


# --- UI Helpers ---
def create_tooltip_metric(label: str, value: Any, tooltip: str) -> str:
    """ Creates an HTML span with a tooltip for Streamlit metrics. """
    # Basic sanitation
    tooltip_safe = str(tooltip).replace('"', '&quot;').replace(
        '<', '&lt;').replace('>', '&gt;')
    value_str = str(value).replace('<', '&lt;').replace('>', '&gt;')
    label_safe = str(label).replace('<', '&lt;').replace('>', '&gt;')
    # Use markdown span with title attribute
    return f'<span title="{tooltip_safe}"><b>{label_safe}:</b> {value_str}</span>'


# --- LLM Context Preparation ---


def prepare_llm_context(session_state: Any) -> Dict[str, Any]:
    """ Prepares a structured dictionary of context data for the LLM assistant. """
    context = {'symbol': session_state.get('current_symbol', 'N/A')}
    logger.debug("Preparing LLM context...")

    # --- Company Info ---
    stock_info = session_state.get('stock_info')
    if stock_info:
        context['company_summary'] = {
            'longName': stock_info.get('longName'), 'symbol': stock_info.get('symbol'),
            'sector': stock_info.get('sector'), 'industry': stock_info.get('industry'),
            'summary': (stock_info.get('longBusinessSummary', '')[:300] + "...") if stock_info.get('longBusinessSummary') else "N/A"
        }
        ratios = {
            'trailingPE': stock_info.get('trailingPE'), 'forwardPE': stock_info.get('forwardPE'),
            'priceToBook': stock_info.get('priceToBook'), 'dividendYield': stock_info.get('dividendYield'),
            'beta': stock_info.get('beta'), 'profitMargins': stock_info.get('profitMargins')
        }
        context['key_ratios'] = {f"{k}": f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"
                                 for k, v in ratios.items() if v is not None}

    # --- Latest Technical Data & Patterns ---
    tech_data = session_state.get('tech_data')
    if tech_data is not None and not tech_data.empty:
        latest_tech = tech_data.iloc[-1]
        latest_date_str = latest_tech['Date'].strftime(
            config.DATE_FORMAT) if pd.notna(latest_tech.get('Date')) else "Latest"
        tech_summary = {
            'Date': latest_date_str,
            'Close': f"${latest_tech.get('Close'):,.2f}" if pd.notna(latest_tech.get('Close')) else "N/A",
            'Volume': f"{latest_tech.get('Volume'):,.0f}" if pd.notna(latest_tech.get('Volume')) else "N/A",
            'SMA_20': f"${latest_tech.get('SMA_20'):,.2f}" if pd.notna(latest_tech.get('SMA_20')) else "N/A",
            'SMA_50': f"${latest_tech.get('SMA_50'):,.2f}" if pd.notna(latest_tech.get('SMA_50')) else "N/A",
            'RSI': f"{latest_tech.get('RSI'):.2f}" if pd.notna(latest_tech.get('RSI')) else "N/A",
            'MACD': f"{latest_tech.get('MACD'):.3f}" if pd.notna(latest_tech.get('MACD')) else "N/A",
            'Signal': f"{latest_tech.get('Signal_Line'):.3f}" if pd.notna(latest_tech.get('Signal_Line')) else "N/A",
        }
        context['latest_technicals'] = tech_summary
        context['technical_patterns'] = session_state.get('patterns', [])

    # --- Sentiment ---
    if session_state.get('avg_sentiment') is not None:
        avg_sent = session_state['avg_sentiment']
        sent_desc = "Positive" if avg_sent > 0.05 else "Negative" if avg_sent < -0.05 else "Neutral"
        context['average_news_sentiment'] = f"{avg_sent:.3f} ({sent_desc})"

    # --- Forecast ---
    forecast_data = session_state.get('prophet_forecast_data')
    if forecast_data is not None and not forecast_data.empty:
        last_forecast = forecast_data.iloc[-1]
        context['prophet_forecast_summary'] = {
            'prediction_date': last_forecast['ds'].strftime(config.DATE_FORMAT),
            'predicted_price': f"${last_forecast['yhat']:.2f}",
            'confidence_interval': f"${last_forecast['yhat_lower']:.2f} - ${last_forecast['yhat_upper']:.2f}"
        }

    # --- ML Prediction ---
    ml_results = session_state.get('ml_results')
    if ml_results is not None:
        pred = ml_results.get('next_day_prediction')
        actual = ml_results.get('last_actual_close')
        context['ml_prediction_summary'] = {
            'predicted_next_close': f"${pred:.2f}" if pred is not None else "N/A",
            'last_actual_close': f"${actual:.2f}" if actual is not None else "N/A",
            'test_rmse': f"${ml_results.get('rmse'):.2f}" if ml_results.get('rmse') is not None else "N/A"
        }

    # --- ESG ---
    esg = session_state.get('esg_scores')
    if esg is not None:  # Include even if empty dict, maybe model can say "No scores found"
        context['esg_scores'] = esg

    # --- Earnings (Next Date if available) ---
    earnings = session_state.get('earnings_calendar')
    if earnings is not None and not earnings.empty:
        # Find next future earnings date
        future_earnings = earnings[earnings['Earnings Date']
                                   >= pd.Timestamp.today().normalize()]
        if not future_earnings.empty:
            next_earning = future_earnings.iloc[0]
            context['next_earnings_date'] = next_earning['Earnings Date'].strftime(
                config.DATE_FORMAT)
            # Optionally add estimates if needed context['next_earnings_avg_est'] = next_earning.get('Earnings Average')

    logger.debug(f"Prepared LLM context with keys: {list(context.keys())}")
    return context

# --- Data Validation / Normalization (Example Helper) ---


def normalize_yfinance_data(raw_data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """
    Normalizes downloaded yfinance data to standard columns:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    Handles MultiIndex, 'Adj Close', NaNs, and case variations.
    """
    if raw_data is None or raw_data.empty:
        logger.warning(f"Cannot normalize empty data for {symbol}.")
        return None

    df = raw_data.copy()
    required_set = {'Open', 'High', 'Low', 'Close', 'Volume'}

    # 1. Handle MultiIndex (often happens with multiple tickers, but check anyway)
    if isinstance(df.columns, pd.MultiIndex):
        logger.info(f"Normalizing MultiIndex columns for {symbol}.")
        # Try to flatten by taking the first level if it matches standard names
        level0 = df.columns.get_level_values(0)
        if any(col in required_set for col in level0):
            df.columns = level0
        else:
            # More complex flattening might be needed, or extraction as before
            logger.error(
                f"Cannot reliably flatten MultiIndex for {symbol}: {df.columns}")
            return None  # Cannot proceed if flattening fails

    # 2. Ensure standard column names (case-insensitive)
    rename_map = {col: col.capitalize()
                  for col in df.columns if isinstance(col, str)}
    df.rename(columns=rename_map, inplace=True)

    # 3. Handle 'Adj Close' -> 'Close' if 'Close' is missing
    if 'Close' not in df.columns and 'Adj close' in df.columns:
        logger.warning(f"Using 'Adj close' as 'Close' for {symbol}.")
        df['Close'] = df['Adj close']
    elif 'Close' not in df.columns and 'Adj Close' in df.columns:  # Handle 'Adj Close' too
        logger.warning(f"Using 'Adj Close' as 'Close' for {symbol}.")
        df['Close'] = df['Adj Close']

    # 4. Check if all required columns are present
    missing_cols = required_set - set(df.columns)
    if missing_cols:
        logger.error(
            f"Missing required columns for {symbol} after normalization: {missing_cols}. Available: {list(df.columns)}")
        return None

    # 5. Select and order standard columns + Date index
    df_normalized = df[list(required_set)].copy()

    # 6. Handle Index and Date column
    if not isinstance(df.index, pd.DatetimeIndex):
        # If index is not datetime, try to find a Date column or reset index
        if 'Date' in df.columns:
            try:
                df_normalized['Date'] = pd.to_datetime(
                    df['Date']).dt.tz_localize(None)
                df_normalized.set_index('Date', inplace=True)
            except Exception as e:
                logger.error(
                    f"Failed to set 'Date' column as index for {symbol}: {e}")
                return None
        else:
            try:
                # Assume index might be date-like string/object
                df_normalized['Date'] = pd.to_datetime(
                    df.index).tz_localize(None)
                df_normalized.set_index('Date', inplace=True)
            except Exception as e:
                logger.error(
                    f"Failed to convert index to DatetimeIndex for {symbol}: {e}")
                return None
    else:
        # Index is already datetime, ensure it's timezone naive and name it 'Date'
        df_normalized.index = df.index.tz_localize(None)
        df_normalized.index.name = 'Date'

    # Ensure Date is also a column
    df_normalized.reset_index(inplace=True)

    # 7. Convert columns to numeric, coercing errors
    for col in required_set:
        df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')

    # 8. Handle NaNs (forward fill then drop any remaining)
    initial_len = len(df_normalized)
    cols_to_check_nan = list(required_set)  # Check NaNs in core OHLCV columns
    if df_normalized[cols_to_check_nan].isnull().any().any():
        logger.warning(
            f"NaN values found in {symbol} data. Applying ffill & dropna.")
        df_normalized.ffill(inplace=True)
        df_normalized.dropna(subset=cols_to_check_nan, inplace=True)
        logger.info(
            f"NaN handling removed {initial_len - len(df_normalized)} rows for {symbol}.")

    if df_normalized.empty:
        logger.error(
            f"DataFrame for {symbol} became empty after normalization/cleaning.")
        return None

    # Ensure final column order
    final_cols = ['Date'] + list(required_set)
    logger.info(
        f"Successfully normalized data for {symbol} ({len(df_normalized)} rows).")
    return df_normalized[final_cols]
