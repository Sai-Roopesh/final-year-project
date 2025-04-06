# src/utils.py

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import streamlit as st
from newsapi import NewsApiClient
from openai import OpenAI
import pandas as pd
import numpy as np  # For safe division

# Use relative imports for files within the same package
from . import config  # Use '.' for relative import within the package

# --- Logging Setup ---
# (setup_logging function remains the same)


def setup_logging() -> logging.Logger:
    """Sets up logging with rotating file and console handlers."""
    log_dir = config.LOG_DIRECTORY
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Warning: Could not create log directory '{log_dir}': {e}")
    log_filename = datetime.now().strftime(
        f"{log_dir}/app_log_{config.DATE_FORMAT}.log")
    logger = logging.getLogger("StockAppLogger")
    if logger.hasHandlers():
        logger.handlers.clear()
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
    logger.addHandler(stream_handler)
    return logger


# --- Initialize Logger ---
logger = setup_logging()

# --- API Client Initialization ---
# (initialize_api_clients function remains the same)


@st.cache_resource(show_spinner="Initializing API connections...")
def initialize_api_clients() -> Tuple[Optional[NewsApiClient], Optional[OpenAI]]:
    """Initializes and validates API clients based on config."""
    newsapi_client = None
    gemini_client = None
    if not config.NEWSAPI_KEY:
        logger.error("NewsAPI Key not found.")
    else:
        try:
            newsapi_client = NewsApiClient(api_key=config.NEWSAPI_KEY)
            logger.info("NewsAPI client initialized.")
        except Exception as e:
            logger.exception("Error initializing NewsAPI client.")
            newsapi_client = None
    if not config.GEMINI_API_KEY:
        logger.error("Gemini API Key not found.")
    else:
        try:
            gemini_client = OpenAI(
                api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)
            logger.info(
                "Gemini client potentially initialized (connection test skipped).")
        except Exception as e:
            logger.exception("Error initializing Gemini client.")
            gemini_client = None
    if not newsapi_client:
        logger.warning("NewsAPI client UNAVAILABLE.")
    if not gemini_client:
        logger.warning("Gemini client UNAVAILABLE.")
    return newsapi_client, gemini_client

# --- Helper Function for Formatting Numbers ---
# (format_value function remains the same)


def format_value(value):
    """Formats large numbers into readable strings (e.g., B, M, K)."""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return 'N/A'
    abs_val = abs(value)
    sign = '-' if value < 0 else ''
    if abs_val >= 1e9:
        return f'{sign}{abs_val / 1e9:.2f}B'
    elif abs_val >= 1e6:
        return f'{sign}{abs_val / 1e6:.2f}M'
    elif abs_val >= 1e3:
        return f'{sign}{abs_val / 1e3:.1f}K'
    else:
        return f'{sign}{abs_val:,.2f}' if isinstance(value, float) else f'{sign}{abs_val:,}'

# --- UI Helpers ---
# (create_tooltip_metric function remains the same)


def create_tooltip_metric(label: str, value: Any, tooltip: str) -> str:
    tooltip_safe = str(tooltip).replace('"', '&quot;').replace(
        '<', '&lt;').replace('>', '&gt;')
    value_str = str(value).replace('<', '&lt;').replace('>', '&gt;')
    label_safe = str(label).replace('<', '&lt;').replace('>', '&gt;')
    return f'<span title="{tooltip_safe}"><b>{label_safe}:</b> {value_str}</span>'

# --- LLM Context Preparation (Add Analyst Info) ---


def prepare_llm_context(session_state: Any) -> Dict[str, Any]:
    """ Prepares a structured dictionary of context data for the LLM assistant. """
    context = {'symbol': session_state.get('current_symbol', 'N/A')}
    logger.debug("Preparing LLM context...")

    analysis_results = session_state.get('analysis_results', {})
    if not analysis_results:
        logger.warning(
            "Analysis results not found in session state for LLM context.")
        return context

    # --- Data Extraction (using analysis_results) ---
    stock_info = analysis_results.get('stock_info')
    tech_data = analysis_results.get('tech_data')
    patterns = analysis_results.get('patterns', [])
    avg_sentiment = analysis_results.get('avg_sentiment')
    forecast_data = analysis_results.get('prophet_forecast_data')
    ml_results = analysis_results.get('ml_results')
    financials = analysis_results.get('financials')
    balance_sheet = analysis_results.get('balance_sheet')
    cash_flow = analysis_results.get('cash_flow')
    esg = analysis_results.get('esg_scores')
    earnings = analysis_results.get('earnings_calendar')
    analyst_info = analysis_results.get('analyst_info')  # <-- Get analyst info

    # --- Company Info & Ratios ---
    # (Remains the same)
    if stock_info:
        context['company_summary'] = {'Name': stock_info.get('longName'), 'Symbol': stock_info.get('symbol'), 'Sector': stock_info.get('sector'), 'Industry': stock_info.get(
            'industry'), 'Summary': (stock_info.get('longBusinessSummary', '')[:300] + "...") if stock_info.get('longBusinessSummary') else "N/A"}
        ratios = {'Trailing PE': stock_info.get('trailingPE'), 'Forward PE': stock_info.get('forwardPE'), 'Price/Book': stock_info.get('priceToBook'), 'Price/Sales': stock_info.get(
            'priceToSalesTrailing12Months'), 'Dividend Yield': stock_info.get('dividendYield'), 'Beta': stock_info.get('beta'), 'Profit Margin': stock_info.get('profitMargins')}
        context['key_ratios'] = {k: (f"{v:.2f}" if isinstance(
            v, (int, float)) else "N/A") for k, v in ratios.items() if v is not None}
        for key in ['Dividend Yield', 'Profit Margin']:
            if key in context['key_ratios'] and context['key_ratios'][key] != "N/A":
                try:
                    context['key_ratios'][key] = f"{float(context['key_ratios'][key]):.2%}"
                except:
                    pass

    # --- Latest Technical Data ---
    # (Remains the same)
    latest_technicals = {}
    if tech_data is not None and not tech_data.empty:
        try:
            latest_tech = tech_data.iloc[-1]
            latest_date_str = latest_tech['Date'].strftime(
                config.DATE_FORMAT) if pd.notna(latest_tech.get('Date')) else "Latest"
            latest_technicals = {'Date': latest_date_str, 'Close': f"${latest_tech.get('Close'):,.2f}" if pd.notna(latest_tech.get('Close')) else "N/A", 'Volume': f"{latest_tech.get('Volume'):,.0f}" if pd.notna(latest_tech.get('Volume')) else "N/A", 'SMA_20': f"${latest_tech.get('SMA_20'):,.2f}" if pd.notna(latest_tech.get('SMA_20')) else "N/A", 'SMA_50': f"${latest_tech.get('SMA_50'):,.2f}" if pd.notna(
                latest_tech.get('SMA_50')) else "N/A", 'RSI': f"{latest_tech.get('RSI'):.2f}" if pd.notna(latest_tech.get('RSI')) else "N/A", 'MACD': f"{latest_tech.get('MACD'):.3f}" if pd.notna(latest_tech.get('MACD')) else "N/A", 'Signal': f"{latest_tech.get('Signal_Line'):.3f}" if pd.notna(latest_tech.get('Signal_Line')) else "N/A"}
            context['latest_technicals'] = latest_technicals
        except IndexError:
            logger.warning(
                "Could not access latest technical data (IndexError).")
            context['latest_technicals'] = {
                "Error": "Could not access latest data"}

    # --- Fundamental Data Summary (with YoY Change) ---
    # (Remains the same)
    fundamental_summary = {}

    def get_yoy_change(latest_val, prev_val):
        if pd.notna(latest_val) and pd.notna(prev_val) and isinstance(latest_val, (int, float)) and isinstance(prev_val, (int, float)) and prev_val != 0:
            change = ((latest_val / prev_val) - 1)
            return f"{change:+.1%}"
        return "N/A"
    if financials is not None and len(financials) > 0:
        try:
            latest_fin = financials.iloc[0]
            prev_fin = financials.iloc[1] if len(financials) > 1 else None
            fin_items = {}
            for item in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA']:
                latest_val = latest_fin.get(item)
                prev_val = prev_fin.get(item) if prev_fin is not None else None
                fin_items[item] = f"{format_value(latest_val)} (YoY: {get_yoy_change(latest_val, prev_val)})"
            fundamental_summary['Income Statement (Latest Year vs Prev)'] = fin_items
        except Exception as e:
            logger.warning(
                f"Error processing income statement for LLM context: {e}")
            fundamental_summary['Income Statement (Latest Year vs Prev)'] = {
                "Error": "Data processing failed"}
    if balance_sheet is not None and len(balance_sheet) > 0:
        try:
            latest_bal = balance_sheet.iloc[0]
            bal_items = {'Total Assets': latest_bal.get('Total Assets'), 'Total Liabilities': latest_bal.get('Total Liabilities Net Minority Interest'), 'Total Equity': latest_bal.get(
                'Stockholders Equity'), 'Current Assets': latest_bal.get('Current Assets'), 'Current Liabilities': latest_bal.get('Current Liabilities'), 'Long Term Debt': latest_bal.get('Long Term Debt And Capital Lease Obligation')}
            fundamental_summary['Balance Sheet (Latest Year)'] = {
                k: format_value(v) for k, v in bal_items.items()}
        except Exception as e:
            logger.warning(
                f"Error processing balance sheet for LLM context: {e}")
            fundamental_summary['Balance Sheet (Latest Year)'] = {
                "Error": "Data processing failed"}
    if cash_flow is not None and len(cash_flow) > 0:
        try:
            latest_cf = cash_flow.iloc[0]
            prev_cf = cash_flow.iloc[1] if len(cash_flow) > 1 else None
            cf_items = {}
            for item in ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow', 'Capital Expenditure']:
                latest_val = latest_cf.get(item)
                prev_val = prev_cf.get(item) if prev_cf is not None else None
                cf_items[item] = f"{format_value(latest_val)} (YoY: {get_yoy_change(latest_val, prev_val)})"
            fundamental_summary['Cash Flow (Latest Year vs Prev)'] = cf_items
        except Exception as e:
            logger.warning(f"Error processing cash flow for LLM context: {e}")
            fundamental_summary['Cash Flow (Latest Year vs Prev)'] = {
                "Error": "Data processing failed"}
    if fundamental_summary:
        context['fundamental_summary'] = fundamental_summary

    # --- Forecast Summary ---
    # (Remains the same)
    forecast_summary = {}
    if forecast_data is not None and not forecast_data.empty:
        try:
            last_forecast = forecast_data.iloc[-1]
            future_dates = forecast_data[forecast_data['ds']
                                         > pd.Timestamp.today().normalize()]
            first_future_forecast = future_dates.iloc[0] if not future_dates.empty else last_forecast
            implied_trend = 'Upward' if last_forecast['yhat'] > first_future_forecast['yhat'] else (
                'Downward' if last_forecast['yhat'] < first_future_forecast['yhat'] else 'Flat')
            forecast_summary = {'Horizon': f"{len(future_dates)} days", 'End Date': last_forecast['ds'].strftime(
                config.DATE_FORMAT), 'Predicted Price (End)': f"${last_forecast['yhat']:.2f}", 'Confidence Interval (End)': f"(${last_forecast['yhat_lower']:.2f} - ${last_forecast['yhat_upper']:.2f})", 'Implied Trend': implied_trend}
            context['prophet_forecast_summary'] = forecast_summary
        except Exception as e:
            logger.warning(
                f"Error processing forecast data for LLM context: {e}")
            context['prophet_forecast_summary'] = {
                "Error": "Data processing failed"}

    # --- ML Prediction Summary ---
    # (Remains the same)
    ml_summary = {}
    if ml_results:
        for model_name, result_data in ml_results.items():
            if isinstance(result_data, dict) and not result_data.get("error"):
                pred = result_data.get('next_day_prediction_price')
                ret = result_data.get('next_day_prediction_return')
                ml_summary[f'{model_name} Predicted Next Close'] = f"${pred:.2f}" if pred is not None else "N/A"
                ml_summary[f'{model_name} Predicted Next Return'] = f"{ret:.4%}" if ret is not None else "N/A"
        if ml_summary:
            context['ml_prediction_summary'] = ml_summary
            try:
                first_model_key = list(ml_results.keys())[0]
                if isinstance(ml_results[first_model_key], dict) and ml_results[first_model_key].get('last_actual_close') is not None:
                    context['ml_prediction_summary'][
                        'Last Actual Close'] = f"${ml_results[first_model_key]['last_actual_close']:.2f}"
            except Exception:
                pass

    # --- Sentiment Summary ---
    # (Remains the same)
    sentiment_summary = "N/A"
    if avg_sentiment is not None:
        sent_desc = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < - \
            0.05 else "Neutral"
        sentiment_summary = f"{avg_sentiment:.3f} ({sent_desc})"
    context['average_news_sentiment'] = sentiment_summary

    # --- ESG Summary ---
    # (Remains the same)
    if esg is not None:
        if isinstance(esg, dict):
            context['esg_scores'] = {k: (f"{v:.1f}" if isinstance(
                v, float) else v) for k, v in esg.items()}
        else:
            logger.warning(f"ESG data is not a dictionary: {type(esg)}")
            context['esg_scores'] = {"Error": "Invalid ESG data format"}

    # --- Earnings Summary ---
    # (Remains the same)
    earnings_summary = {}
    if earnings is not None and not earnings.empty:
        try:
            if 'Earnings Date' not in earnings.columns:
                logger.warning(
                    "Earnings calendar missing 'Earnings Date' column.")
            else:
                earnings['Earnings Date'] = pd.to_datetime(
                    earnings['Earnings Date'], errors='coerce')
                future_earnings = earnings[earnings['Earnings Date'] >= pd.Timestamp.today(
                ).normalize()].sort_values('Earnings Date')
                if not future_earnings.empty:
                    next_earning = future_earnings.iloc[0]
                    earnings_summary['Next Earnings Date'] = next_earning['Earnings Date'].strftime(
                        config.DATE_FORMAT)
                    avg_eps = next_earning.get('Earnings Average')
                    avg_rev = next_earning.get('Revenue Average')
                    if pd.notna(avg_eps):
                        earnings_summary['Next EPS Est.'] = f"{avg_eps:.2f}"
                    if pd.notna(avg_rev):
                        earnings_summary['Next Revenue Est.'] = format_value(
                            avg_rev)
            context['earnings_summary'] = earnings_summary
        except Exception as e:
            logger.warning(
                f"Could not process earnings calendar for LLM context: {e}")
            context['earnings_summary'] = {"Error": "Data processing failed"}

    # --- NEW: Analyst Ratings Summary ---
    if analyst_info:
        analyst_summary = {}
        current_rec = analyst_info.get('current_recommendation')
        num_analysts = analyst_info.get('num_analysts')
        targets = analyst_info.get('targets', {})

        if current_rec:
            analyst_summary['Consensus Recommendation'] = current_rec.replace(
                '_', ' ').title()
        if num_analysts:
            analyst_summary['Number of Analysts'] = num_analysts
        if targets:
            # Format target prices nicely
            target_summary = {}
            if 'Low' in targets:
                target_summary['Low Target'] = f"${targets['Low']:.2f}"
            if 'Mean' in targets:
                target_summary['Mean Target'] = f"${targets['Mean']:.2f}"
            if 'Median' in targets:
                target_summary['Median Target'] = f"${targets['Median']:.2f}"
            if 'High' in targets:
                target_summary['High Target'] = f"${targets['High']:.2f}"
            if target_summary:  # Add only if any targets were found
                analyst_summary['Price Targets'] = target_summary

        if analyst_summary:  # Add to main context only if we found something
            context['analyst_ratings_summary'] = analyst_summary
    # --- END NEW ---

    # --- Signal Synthesis ---
    # (Remains the same)
    signals = []
    if patterns:
        for pattern in patterns:
            if "Bullish" in pattern or "Golden Cross" in pattern or "Oversold" in pattern:
                signals.append(
                    f"Technical Pattern: {pattern} (Potential Bullish)")
            elif "Bearish" in pattern or "Death Cross" in pattern or "Overbought" in pattern:
                signals.append(
                    f"Technical Pattern: {pattern} (Potential Bearish)")
            elif "Price above Upper" in pattern:
                signals.append(
                    f"Technical Pattern: {pattern} (Potential Overextension/Strong Momentum)")
            elif "Price below Lower" in pattern:
                signals.append(
                    f"Technical Pattern: {pattern} (Potential Oversold/Strong Downtrend)")
    if avg_sentiment is not None:
        if avg_sentiment > 0.1:
            signals.append("Sentiment: Generally Positive")
        elif avg_sentiment < -0.1:
            signals.append("Sentiment: Generally Negative")
    if forecast_summary and forecast_summary.get('Implied Trend') == 'Upward':
        signals.append("Forecast: Implied Upward Trend")
    elif forecast_summary and forecast_summary.get('Implied Trend') == 'Downward':
        signals.append("Forecast: Implied Downward Trend")
    if ml_summary:
        try:
            first_model_key = list(ml_results.keys())[0]
            ret_str = ml_summary.get(
                f'{first_model_key} Predicted Next Return', "N/A")
            if ret_str != "N/A":
                ret_val = float(ret_str.strip('%')) / 100
                if ret_val > 0.001:
                    signals.append(
                        f"ML ({first_model_key}): Predicts Positive Next Day Return")
                elif ret_val < -0.001:
                    signals.append(
                        f"ML ({first_model_key}): Predicts Negative Next Day Return")
        except Exception as e:
            logger.warning(f"Could not derive signal from ML prediction: {e}")
    context['derived_signals'] = signals if signals else [
        "No specific directional signals derived from the available context data."]

    logger.debug(f"Prepared LLM context with keys: {list(context.keys())}")
    return context


# --- Data Validation / Normalization ---
# (normalize_yfinance_data function remains the same)
def normalize_yfinance_data(raw_data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    if raw_data is None or raw_data.empty:
        logger.warning(f"Cannot normalize empty data for {symbol}.")
        return None
    df = raw_data.copy()
    required_set = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if isinstance(df.columns, pd.MultiIndex):
        logger.info(f"Normalizing MultiIndex columns for {symbol}.")
        level0 = df.columns.get_level_values(0)
        if any(col in required_set for col in level0):
            df.columns = level0
            df = df[[
                col for col in df.columns if col in required_set or col == 'Adj Close']]
        else:
            logger.error(
                f"Cannot reliably flatten MultiIndex for {symbol}: {df.columns}")
            return None
    rename_map = {col: col.capitalize()
                  for col in df.columns if isinstance(col, str)}
    df.rename(columns=rename_map, inplace=True)
    if 'Close' not in df.columns and 'Adj close' in df.columns:
        logger.warning(f"Using 'Adj close' as 'Close' for {symbol}.")
        df.rename(columns={'Adj close': 'Close'}, inplace=True)
    elif 'Close' not in df.columns and 'Adj Close' in df.columns:
        logger.warning(f"Using 'Adj Close' as 'Close' for {symbol}.")
        df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    missing_cols = required_set - set(df.columns)
    if missing_cols:
        logger.error(
            f"Missing required columns for {symbol} after normalization: {missing_cols}. Available: {list(df.columns)}")
        return None
    cols_to_select = [col for col in required_set if col in df.columns]
    df_normalized = df[cols_to_select].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
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
                df_normalized['Date'] = pd.to_datetime(
                    df.index).dt.tz_localize(None)
                df_normalized.set_index('Date', inplace=True)
            except Exception as e:
                logger.error(
                    f"Failed to convert index to DatetimeIndex for {symbol}: {e}")
                return None
    else:
        df_normalized.index = df.index.tz_localize(None)
        df_normalized.index.name = 'Date'
    df_normalized.reset_index(inplace=True)
    for col in required_set:
        if col in df_normalized.columns:
            df_normalized[col] = pd.to_numeric(
                df_normalized[col], errors='coerce')
    initial_len = len(df_normalized)
    cols_to_check_nan = [
        col for col in required_set if col in df_normalized.columns]
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
    final_cols_order = [
        'Date'] + [col for col in required_set if col in df_normalized.columns]
    logger.info(
        f"Successfully normalized data for {symbol} ({len(df_normalized)} rows).")
    return df_normalized[final_cols_order]
