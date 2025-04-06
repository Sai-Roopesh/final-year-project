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
import numpy as np

from . import config

# --- Logging Setup ---


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
    logger_instance = logging.getLogger(
        "StockAppLogger")  # Use a specific name
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()
    logger_instance.setLevel(logging.INFO)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger_instance.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to attach file logger: {e}")
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger_instance.addHandler(stream_handler)
    return logger_instance


logger = setup_logging()

# --- API Client Initialization ---


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
            # Simple test - list models (optional, might fail depending on API key permissions)
            # try:
            #     gemini_client.models.list()
            #     logger.info("Gemini client initialized and connection tested.")
            # except Exception as test_err:
            #     logger.warning(f"Gemini client initialized, but connection test failed: {test_err}. Chat might not work.")
            # Keep this if test skipped/fails
            logger.info("Gemini client potentially initialized.")
        except Exception as e:
            logger.exception("Error initializing Gemini client.")
            gemini_client = None
    if not newsapi_client:
        logger.warning("NewsAPI client UNAVAILABLE.")
    if not gemini_client:
        logger.warning("Gemini client UNAVAILABLE.")
    return newsapi_client, gemini_client

# --- Helper Function for Formatting Numbers ---


def format_value(value):
    """Formats large numbers into readable strings (e.g., B, M, K)."""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return 'N/A'
    abs_val = abs(value)
    sign = '-' if value < 0 else ''
    if abs_val >= 1e12:
        return f'{sign}{abs_val / 1e12:.2f}T'
    if abs_val >= 1e9:
        return f'{sign}{abs_val / 1e9:.2f}B'
    if abs_val >= 1e6:
        return f'{sign}{abs_val / 1e6:.2f}M'
    if abs_val >= 1e3:
        return f'{sign}{abs_val / 1e3:.1f}K'
    # Format floats with 2 decimals, ints with commas
    return f'{sign}{value:,.2f}' if isinstance(value, float) and abs_val < 1000 else f'{sign}{value:,}'


# --- UI Helpers ---
def create_tooltip_metric(label: str, value: Any, tooltip: str) -> str:
    tooltip_safe = str(tooltip).replace('"', '&quot;').replace(
        '<', '&lt;').replace('>', '&gt;')
    value_str = str(value).replace('<', '&lt;').replace('>', '&gt;')
    label_safe = str(label).replace('<', '&lt;').replace('>', '&gt;')
    return f'<span title="{tooltip_safe}"><b>{label_safe}:</b> {value_str}</span>'


# --- LLM Context Preparation (MODIFIED) ---
def prepare_llm_context(analysis_results_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares a structured dictionary of context data for the LLM assistant
    using the provided analysis results dictionary.
    """
    logger.debug("Preparing LLM context...")
    # Use the passed dictionary directly
    if not analysis_results_dict:
        logger.warning(
            "prepare_llm_context called with empty analysis_results_dict.")
        return {'symbol': 'N/A', 'error': 'Analysis results missing'}

    # Start context with symbol from the results dict
    context = {'symbol': analysis_results_dict.get('symbol', 'N/A')}

    # --- Data Extraction (using analysis_results_dict) ---
    stock_info = analysis_results_dict.get('stock_info')
    tech_data = analysis_results_dict.get('tech_data')
    patterns = analysis_results_dict.get('patterns', [])
    avg_sentiment = analysis_results_dict.get('avg_sentiment')
    sentiment_method = analysis_results_dict.get(
        'sentiment_method_used', 'vader')  # Get method used
    forecast_data = analysis_results_dict.get('prophet_forecast_data')
    ml_results = analysis_results_dict.get('ml_results')
    financials = analysis_results_dict.get('financials')
    balance_sheet = analysis_results_dict.get('balance_sheet')
    cash_flow = analysis_results_dict.get('cash_flow')
    esg = analysis_results_dict.get('esg_scores')
    earnings = analysis_results_dict.get('earnings_calendar')
    analyst_info = analysis_results_dict.get('analyst_info')
    economic_data = analysis_results_dict.get('economic_data')

    # --- Company Info & Ratios ---
    if stock_info:
        context['company_summary'] = {
            'Name': stock_info.get('longName'), 'Symbol': stock_info.get('symbol'),
            'Sector': stock_info.get('sector'), 'Industry': stock_info.get('industry'),
            'Summary': (stock_info.get('longBusinessSummary', '')[:300] + "...") if stock_info.get('longBusinessSummary') else "N/A"
        }
        ratios = {
            'Trailing PE': stock_info.get('trailingPE'), 'Forward PE': stock_info.get('forwardPE'),
            'Price/Book': stock_info.get('priceToBook'), 'Price/Sales': stock_info.get('priceToSalesTrailing12Months'),
            'Dividend Yield': stock_info.get('dividendYield'), 'Beta': stock_info.get('beta'),
            'Profit Margin': stock_info.get('profitMargins'), 'ROE': stock_info.get('returnOnEquity')
        }
        # Format ratios nicely
        formatted_ratios = {}
        for k, v in ratios.items():
            if v is not None and pd.notna(v):
                try:
                    if k in ['Dividend Yield', 'Profit Margin', 'ROE']:
                        formatted_ratios[k] = f"{v:.2%}"
                    else:
                        formatted_ratios[k] = f"{v:.2f}"
                except (TypeError, ValueError):
                    formatted_ratios[k] = "N/A"
            else:
                formatted_ratios[k] = "N/A"
        context['key_ratios'] = formatted_ratios

    # --- Latest Technical Data ---
    if tech_data is not None and not tech_data.empty:
        try:
            latest_tech = tech_data.iloc[-1]
            latest_date_str = latest_tech['Date'].strftime(
                config.DATE_FORMAT) if pd.notna(latest_tech.get('Date')) else "Latest"
            latest_technicals = {
                'Date': latest_date_str,
                'Close': f"${latest_tech.get('Close'):,.2f}" if pd.notna(latest_tech.get('Close')) else "N/A",
                'Volume': f"{latest_tech.get('Volume'):,.0f}" if pd.notna(latest_tech.get('Volume')) else "N/A",
                f'SMA_{config.SMA_PERIODS[0]}': f"${latest_tech.get(f'SMA_{config.SMA_PERIODS[0]}'):,.2f}" if pd.notna(latest_tech.get(f'SMA_{config.SMA_PERIODS[0]}')) else "N/A",
                f'SMA_{config.SMA_PERIODS[1]}': f"${latest_tech.get(f'SMA_{config.SMA_PERIODS[1]}'):,.2f}" if pd.notna(latest_tech.get(f'SMA_{config.SMA_PERIODS[1]}')) else "N/A",
                'RSI': f"{latest_tech.get('RSI'):.2f}" if pd.notna(latest_tech.get('RSI')) else "N/A",
                'MACD': f"{latest_tech.get('MACD'):.3f}" if pd.notna(latest_tech.get('MACD')) else "N/A",
                'Signal': f"{latest_tech.get('Signal_Line'):.3f}" if pd.notna(latest_tech.get('Signal_Line')) else "N/A",
                'BB Upper': f"${latest_tech.get('BB_Upper'):.2f}" if pd.notna(latest_tech.get('BB_Upper')) else "N/A",
                'BB Lower': f"${latest_tech.get('BB_Lower'):.2f}" if pd.notna(latest_tech.get('BB_Lower')) else "N/A",
            }
            context['latest_technicals'] = latest_technicals
        except IndexError:
            logger.warning(
                "Could not access latest technical data (IndexError).")
            context['latest_technicals'] = {
                "Error": "Could not access latest data"}
        if patterns:
            # Add patterns if they exist
            context['technical_patterns'] = patterns

    # --- Fundamental Data Summary (YoY) ---
    fundamental_summary = {}

    def get_yoy_change(latest_val, prev_val):
        if pd.notna(latest_val) and pd.notna(prev_val) and isinstance(latest_val, (int, float)) and isinstance(prev_val, (int, float)) and prev_val != 0:
            try:
                return f"{((latest_val / prev_val) - 1):+.1%}"
            except ZeroDivisionError:
                return "N/A"
        return "N/A"

    if financials is not None and len(financials) > 0:
        try:
            latest_fin = financials.iloc[0]
            prev_fin = financials.iloc[1] if len(financials) > 1 else None
            fin_items = {}
            for item in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA']:
                latest_val = latest_fin.get(item)
                prev_val = prev_fin.get(item) if prev_fin is not None else None
                if pd.notna(latest_val):
                    fin_items[item] = f"{format_value(latest_val)} (YoY: {get_yoy_change(latest_val, prev_val)})"
            if fin_items:
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
            formatted_bal_items = {k: format_value(
                v) for k, v in bal_items.items() if pd.notna(v)}
            if formatted_bal_items:
                fundamental_summary['Balance Sheet (Latest Year)'] = formatted_bal_items
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
                if pd.notna(latest_val):
                    cf_items[item] = f"{format_value(latest_val)} (YoY: {get_yoy_change(latest_val, prev_val)})"
            if cf_items:
                fundamental_summary['Cash Flow (Latest Year vs Prev)'] = cf_items
        except Exception as e:
            logger.warning(f"Error processing cash flow for LLM context: {e}")
            fundamental_summary['Cash Flow (Latest Year vs Prev)'] = {
                "Error": "Data processing failed"}

    if fundamental_summary:
        context['fundamental_summary'] = fundamental_summary

    # --- Forecast Summary ---
    if forecast_data is not None and not forecast_data.empty:
        try:
            last_forecast = forecast_data.iloc[-1]
            future_dates = forecast_data[pd.to_datetime(
                forecast_data['ds']) > pd.Timestamp.today().normalize()]
            first_future_forecast = future_dates.iloc[0] if not future_dates.empty else last_forecast
            # Use first and last points of the *future* forecast for trend
            trend_start_price = first_future_forecast['yhat']
            trend_end_price = last_forecast['yhat']
            implied_trend = 'Upward' if trend_end_price > trend_start_price else (
                'Downward' if trend_end_price < trend_start_price else 'Flat')

            context['prophet_forecast_summary'] = {
                'Horizon': f"{len(future_dates)} days",
                'End Date': pd.to_datetime(last_forecast['ds']).strftime(config.DATE_FORMAT),
                'Predicted Price (End)': f"${last_forecast['yhat']:.2f}",
                'Confidence Interval (End)': f"(${last_forecast['yhat_lower']:.2f} - ${last_forecast['yhat_upper']:.2f})",
                'Implied Trend': implied_trend
            }
        except Exception as e:
            logger.warning(
                f"Error processing forecast data for LLM context: {e}")
            context['prophet_forecast_summary'] = {
                "Error": "Data processing failed"}

    # --- ML Prediction Summary ---
    if ml_results:
        ml_summary = {}
        for model_name, result_data in ml_results.items():
            if isinstance(result_data, dict) and not result_data.get("error"):
                pred_price = result_data.get('next_day_prediction_price')
                pred_return = result_data.get('next_day_prediction_return')
                ml_summary[f'{model_name} Predicted Next Close'] = f"${pred_price:.2f}" if pred_price is not None else "N/A"
                ml_summary[f'{model_name} Predicted Next Return'] = f"{pred_return:.4%}" if pred_return is not None else "N/A"
        if ml_summary:
            context['ml_prediction_summary'] = ml_summary
            try:  # Add last actual close for context
                first_model_key = list(ml_results.keys())[0]
                if isinstance(ml_results[first_model_key], dict) and ml_results[first_model_key].get('last_actual_close') is not None:
                    context['ml_prediction_summary'][
                        'Last Actual Close'] = f"${ml_results[first_model_key]['last_actual_close']:.2f}"
            except Exception:
                pass  # Ignore if fails

    # --- Sentiment Summary ---
    sentiment_summary = "N/A"
    if avg_sentiment is not None:
        sent_desc = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < - \
            0.05 else "Neutral"
        sentiment_summary = f"{avg_sentiment:.3f} ({sent_desc}) using {sentiment_method.upper()}"
    context['average_news_sentiment'] = sentiment_summary

    # --- ESG Summary ---
    if esg is not None:
        context['esg_scores'] = {k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in esg.items(
        )} if isinstance(esg, dict) else {"Error": "Invalid ESG data format"}

    # --- Earnings Summary ---
    if earnings is not None and not earnings.empty:
        try:
            if 'Earnings Date' in earnings.columns:
                earnings['Earnings Date'] = pd.to_datetime(
                    earnings['Earnings Date'], errors='coerce')
                future_earnings = earnings[earnings['Earnings Date'] >= pd.Timestamp.today(
                ).normalize()].sort_values('Earnings Date')
                if not future_earnings.empty:
                    next_earning = future_earnings.iloc[0]
                    earnings_summary = {}
                    earnings_summary['Next Earnings Date'] = next_earning['Earnings Date'].strftime(
                        config.DATE_FORMAT)
                    avg_eps = next_earning.get('Earnings Average')
                    avg_rev = next_earning.get('Revenue Average')
                    if pd.notna(avg_eps):
                        earnings_summary['Next EPS Est.'] = f"{avg_eps:.2f}"
                    if pd.notna(avg_rev):
                        earnings_summary['Next Revenue Est.'] = format_value(
                            avg_rev)
                    if earnings_summary:
                        context['earnings_summary'] = earnings_summary
            else:
                logger.warning(
                    "Earnings calendar missing 'Earnings Date' column.")
        except Exception as e:
            logger.warning(
                f"Could not process earnings calendar for LLM context: {e}")
            context['earnings_summary'] = {"Error": "Data processing failed"}

    # --- Analyst Ratings Summary ---
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
            target_summary = {k: f"${v:.2f}" for k,
                              v in targets.items() if v is not None}
            if target_summary:
                analyst_summary['Price Targets'] = target_summary
        if analyst_summary:
            context['analyst_ratings_summary'] = analyst_summary

    # --- Economic Data Summary ---
    if economic_data is not None and not economic_data.empty:
        try:
            latest_econ = economic_data.iloc[-1]
            context['latest_economic_data'] = {col: f"{latest_econ[col]:,.2f}" if pd.notna(
                latest_econ[col]) else "N/A" for col in economic_data.columns}
            context['latest_economic_data']['Date'] = latest_econ.name.strftime(
                config.DATE_FORMAT) if isinstance(latest_econ.name, pd.Timestamp) else "Latest"
        except Exception as e:
            logger.warning(
                f"Could not process economic data for LLM context: {e}")
            context['latest_economic_data'] = {
                "Error": "Data processing failed"}

    # --- Derived Signals (Simplified) ---
    # This section could be made more sophisticated
    signals = []
    if patterns:
        signals.extend(
            [f"Pattern: {p}" for p in patterns if "No significant" not in p])
    if avg_sentiment is not None:
        if avg_sentiment > 0.15:
            signals.append("Sentiment: Strongly Positive")
        elif avg_sentiment > 0.05:
            signals.append("Sentiment: Leaning Positive")
        elif avg_sentiment < -0.15:
            signals.append("Sentiment: Strongly Negative")
        elif avg_sentiment < -0.05:
            signals.append("Sentiment: Leaning Negative")
    if 'prophet_forecast_summary' in context and context['prophet_forecast_summary'].get('Implied Trend') != 'Flat':
        signals.append(
            f"Forecast Trend: {context['prophet_forecast_summary']['Implied Trend']}")
    if 'ml_prediction_summary' in context:
        # Example: Check Random Forest prediction if available
        rf_ret_str = context['ml_prediction_summary'].get(
            'Random Forest Predicted Next Return', "N/A")
        if rf_ret_str != "N/A":
            try:
                ret_val = float(rf_ret_str.strip('%')) / 100
                if ret_val > 0.005:
                    signals.append("ML (RF): Predicts Strong Positive Return")
                elif ret_val > 0.001:
                    signals.append("ML (RF): Predicts Positive Return")
                elif ret_val < -0.005:
                    signals.append("ML (RF): Predicts Strong Negative Return")
                elif ret_val < -0.001:
                    signals.append("ML (RF): Predicts Negative Return")
            except ValueError:
                pass  # Ignore if conversion fails
    # Add analyst consensus signal
    if 'analyst_ratings_summary' in context and 'Consensus Recommendation' in context['analyst_ratings_summary']:
        rec = context['analyst_ratings_summary']['Consensus Recommendation'].lower()
        if 'buy' in rec:
            signals.append("Analyst Consensus: Leaning Positive")
        elif 'sell' in rec:
            signals.append("Analyst Consensus: Leaning Negative")
        elif 'hold' in rec:
            signals.append("Analyst Consensus: Neutral")

    context['derived_signals'] = signals if signals else [
        "Neutral/Mixed signals from available data."]

    logger.debug(f"LLM context prepared with keys: {list(context.keys())}")
    return context


# --- Data Validation / Normalization ---
def normalize_yfinance_data(raw_data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """Normalizes yfinance data columns and index."""
    if raw_data is None or raw_data.empty:
        logger.warning(f"Cannot normalize empty data for {symbol}.")
        return None
    df = raw_data.copy()
    required_set = {'Open', 'High', 'Low', 'Close', 'Volume'}
    # Handle potential MultiIndex columns if auto_adjust=False was used elsewhere
    if isinstance(df.columns, pd.MultiIndex):
        logger.info(f"Normalizing MultiIndex columns for {symbol}.")
        level0_cols = df.columns.get_level_values(0).unique()
        # Check if required columns are present at level 0
        if any(col in required_set for col in level0_cols):
            # Attempt to select the first level if it contains the required cols
            try:
                df.columns = df.columns.get_level_values(0)
                # Remove duplicate columns if any resulted from flattening
                df = df.loc[:, ~df.columns.duplicated()]
            except Exception as multi_err:
                logger.error(
                    f"Failed to flatten MultiIndex for {symbol}: {multi_err}")
                return None
        else:
            logger.error(
                f"Required columns not found in MultiIndex level 0 for {symbol}: {level0_cols}")
            return None

    # Standardize column names (case-insensitive)
    rename_map = {col: col.capitalize()
                  for col in df.columns if isinstance(col, str)}
    df.rename(columns=rename_map, inplace=True)

    # Handle 'Adj Close' specifically if 'Close' is missing (common with auto_adjust=False)
    if 'Close' not in df.columns:
        if 'Adj close' in df.columns:  # yfinance sometimes uses lowercase 'c'
            logger.warning(f"Using 'Adj close' as 'Close' for {symbol}.")
            df.rename(columns={'Adj close': 'Close'}, inplace=True)
        elif 'Adj Close' in df.columns:  # More standard capitalization
            logger.warning(f"Using 'Adj Close' as 'Close' for {symbol}.")
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)

    # Check if all required columns are now present
    missing_cols = required_set - set(df.columns)
    if missing_cols:
        logger.error(
            f"Missing required columns for {symbol} after normalization: {missing_cols}. Available: {list(df.columns)}")
        return None

    # Select only required columns
    cols_to_select = list(required_set)
    df_normalized = df[cols_to_select].copy()

    # Ensure index is DatetimeIndex and timezone-naive, named 'Date'
    if not isinstance(df.index, pd.DatetimeIndex):
        # If original index wasn't datetime, try resetting and finding 'Date' column
        df_reset = df.reset_index()
        date_col_found = None
        if 'Date' in df_reset.columns:
            date_col_found = 'Date'
        elif 'Datetime' in df_reset.columns:
            date_col_found = 'Datetime'
        elif 'index' in df_reset.columns and pd.api.types.is_datetime64_any_dtype(df_reset['index']):
            date_col_found = 'index'

        if date_col_found:
            try:
                df_normalized['Date'] = pd.to_datetime(
                    df_reset[date_col_found]).dt.tz_localize(None)
                df_normalized.set_index('Date', inplace=True)
            except Exception as e:
                logger.error(
                    f"Failed to set '{date_col_found}' column as DatetimeIndex for {symbol}: {e}")
                return None
        else:  # If no suitable date column found
            logger.error(
                f"Could not find or convert a suitable Date index for {symbol}.")
            return None
    else:  # If original index was already DatetimeIndex
        df_normalized.index = df.index.tz_localize(None)  # Make timezone-naive
        df_normalized.index.name = 'Date'  # Ensure index name is 'Date'

    # Convert columns to numeric, coercing errors
    for col in required_set:
        if col in df_normalized.columns:
            df_normalized[col] = pd.to_numeric(
                df_normalized[col], errors='coerce')

    # Handle NaNs (forward fill first, then drop any remaining)
    initial_len = len(df_normalized)
    if df_normalized.isnull().any().any():
        logger.warning(
            f"NaN values found in {symbol} data after numeric conversion. Applying ffill & dropna.")
        df_normalized.ffill(inplace=True)
        df_normalized.dropna(inplace=True)  # Drop rows with any remaining NaNs
        logger.info(
            f"NaN handling removed {initial_len - len(df_normalized)} rows for {symbol}.")

    if df_normalized.empty:
        logger.error(
            f"DataFrame for {symbol} became empty after normalization/cleaning.")
        return None

    # Reset index to make 'Date' a column again for consistency
    df_normalized.reset_index(inplace=True)
    # Ensure standard column order
    final_cols_order = [
        'Date'] + [col for col in required_set if col in df_normalized.columns]

    logger.info(
        f"Successfully normalized data for {symbol} ({len(df_normalized)} rows).")
    return df_normalized[final_cols_order]
