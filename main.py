# final2.py
# --------------------------------------------------------------------------------
# A fully functional Streamlit application providing advanced stock analysis.
# This version fixes the missing Prophet forecast integration, ensures Prophet
# predictions are non-negative, and verifies news/sentiment analysis flow.
# --------------------------------------------------------------------------------

# --- Standard Library Imports ---
import os
import logging
import time
from json.decoder import JSONDecodeError
from requests.exceptions import RequestException
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# --- Third-party Library Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Correct import

from prophet import Prophet  # Import Prophet
from newsapi import NewsApiClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# NOTE: Using the OpenAI library as a stand-in for Gemini’s endpoint.
from openai import OpenAI

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------
LOG_DIRECTORY = 'logs'
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_STOCK_SYMBOL = "AAPL"
DEFAULT_COMPANY_NAME = "Apple Inc."
MIN_INVESTMENT = 1000
DEFAULT_INVESTMENT = 10000
DEFAULT_PORTFOLIO_STOCKS = "AAPL, GOOGL, MSFT"
DEFAULT_CORRELATION_STOCKS = "AAPL, GOOGL, MSFT"
DEFAULT_ML_STOCK = "AAPL"
DEFAULT_ESG_STOCK = "AAPL"
MAX_CHAT_TOKENS = 2048  # Increased token limit for LLM response

# Ensure NLTK data is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    try:
        nltk.download('vader_lexicon')
    except Exception as e:
        st.error(f"Failed to download NLTK 'vader_lexicon': {e}")
except Exception as e:  # Catch other potential errors during find
    st.error(f"Error checking for NLTK 'vader_lexicon': {e}")


# --- Load Environment Variables ---
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Sets up logging with rotating file and console handlers."""
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    log_filename = datetime.now().strftime(
        f"{LOG_DIRECTORY}/app_log_{DATE_FORMAT}.log")
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        # Clear existing handlers if any (useful for Streamlit's rerun behavior)
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    # Use utf-8 encoding for file handler
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logging()
logger.info("Application session started.")

# -----------------------------------------------------------------------------
# API Client Initialization
# -----------------------------------------------------------------------------


def initialize_api_clients() -> Tuple[Optional[NewsApiClient], Optional[OpenAI]]:
    """Initializes and validates API clients."""
    newsapi_client = None
    gemini_client = None

    if not NEWSAPI_KEY:
        logger.error("NewsAPI Key not found in environment variables.")
        st.error(
            "NewsAPI Key not found. Please set it in the .env file or environment variables.")
    else:
        try:
            newsapi_client = NewsApiClient(api_key=NEWSAPI_KEY)
            logger.info("NewsAPI client initialized successfully.")
        except Exception as e:
            logger.exception("Error initializing NewsAPI client.")
            st.error(f"Error initializing NewsAPI client: {e}")

    if not GEMINI_API_KEY:
        logger.error("Gemini API Key not found in environment variables.")
        st.error(
            "Gemini API Key not found. Please set it in the .env file or environment variables.")
    else:
        try:
            # Ensure correct base URL for Gemini via OpenAI library compatibility layer
            # Check the latest documentation for the correct base URL if issues arise.
            # As of early 2024, this often involved models like 'gemini-pro'.
            # The URL provided might be specific to a beta or certain setup.
            # Let's assume the provided URL is correct for the user's setup for now.
            gemini_client = OpenAI(api_key=GEMINI_API_KEY,
                                   # Corrected Base URL common structure
                                   base_url="https://generativelanguage.googleapis.com/v1beta/")
            # Test connection (Optional but recommended)
            # try:
            #     gemini_client.models.list() # Example: List available models
            #     logger.info("Gemini client initialized and connection tested successfully.")
            # except Exception as test_e:
            #     logger.error(f"Gemini client initialized, but connection test failed: {test_e}")
            #     st.warning(f"Gemini client initialized, but connection test failed. Chat features might not work. Error: {test_e}")
            #     # Keep gemini_client object, maybe partial functionality works
            logger.info(
                "Gemini client potentially initialized (connection test skipped).")

        except Exception as e:
            logger.exception("Error initializing Gemini client.")
            st.error(f"Error initializing Gemini client: {e}")
            gemini_client = None  # Ensure client is None if init fails

    if not newsapi_client or not gemini_client:
        st.warning(
            "One or more API clients failed to initialize. Some features may be unavailable.")
        if not gemini_client:
            st.error(
                "Gemini client failed to initialize. AI Chat Assistant and Symbol Lookup by Name features disabled.")
            logger.error("Gemini client initialization failed.")
        if not newsapi_client:
            st.error(
                "NewsAPI client failed to initialize. News and Sentiment features disabled.")
            logger.error("NewsAPI client initialization failed.")

    return newsapi_client, gemini_client


newsapi, gemini_client = initialize_api_clients()

# -----------------------------------------------------------------------------
# Enhanced Stock Analyzer Class
# -----------------------------------------------------------------------------


class EnhancedStockAnalyzer:
    """
    Encapsulates stock data fetching, analysis, prediction, and visualization.
    """

    def __init__(self, newsapi_client: Optional[NewsApiClient], gemini_client: Optional[OpenAI], logger_instance: logging.Logger):
        self.newsapi = newsapi_client
        self.gemini = gemini_client  # Use the potentially None gemini_client
        self.logger = logger_instance
        self.sia = None  # Initialize as None
        self.sentiment_available = False  # Default to False
        try:
            # Check if VADER lexicon is available before initializing
            nltk.data.find('sentiment/vader_lexicon.zip')
            self.sia = SentimentIntensityAnalyzer()
            self.sentiment_available = True
            self.logger.info(
                "VADER lexicon found. Sentiment analysis enabled.")
        except LookupError:
            self.logger.warning(
                "VADER lexicon not found (LookupError). Sentiment analysis disabled. Run nltk.download('vader_lexicon') if needed.")
            st.warning(
                "NLTK VADER lexicon not found. Sentiment analysis will be disabled.")
        except Exception as e:
            self.logger.error(
                f"Error initializing SentimentIntensityAnalyzer or finding lexicon: {e}")
            st.error(f"Could not initialize sentiment analysis: {e}")

    @st.cache_data(show_spinner="Fetching stock symbol...")
    def get_stock_symbol(_self, company_name_or_symbol: str) -> Optional[str]:
        """Resolves a company name or validates a symbol using Gemini or yfinance."""
        # Input basic validation
        if not company_name_or_symbol or not isinstance(company_name_or_symbol, str):
            _self.logger.warning("get_stock_symbol called with invalid input.")
            return None

        cleaned_input = company_name_or_symbol.strip().upper()

        # 1. Direct Check: If it looks like a symbol, try yfinance first
        # Simple check: All caps, no spaces, relatively short. Adjust regex for more complex symbols if needed.
        is_likely_symbol = cleaned_input.isalnum(
        ) and cleaned_input.isupper() and len(cleaned_input) <= 6
        if is_likely_symbol:
            try:
                _self.logger.info(
                    f"Input '{cleaned_input}' looks like a symbol. Verifying with yfinance...")
                test_ticker = yf.Ticker(cleaned_input)
                # Fetching info is a good way to check validity. History might be empty for valid tickers sometimes.
                if test_ticker.info and test_ticker.info.get('regularMarketPrice') is not None:
                    _self.logger.info(
                        f"yfinance verified '{cleaned_input}' as a valid symbol.")
                    return cleaned_input
                else:
                    _self.logger.warning(
                        f"yfinance did not return valid info for likely symbol '{cleaned_input}'.")
            except Exception as e:
                _self.logger.warning(
                    f"yfinance check for '{cleaned_input}' failed: {e}. Will try Gemini if available.")
                # Don't return None yet, proceed to Gemini if available

        # 2. Gemini Lookup: If it doesn't look like a symbol OR yfinance failed/didn't verify
        if not _self.gemini:
            _self.logger.error(
                "Gemini client not available for symbol lookup by name.")
            if not is_likely_symbol:  # Only show error if it wasn't likely a symbol to begin with
                st.error(
                    "AI Chat Assistant is not configured. Cannot look up symbol by name.")
            return None  # Cannot proceed without Gemini for name lookup

        _self.logger.info(
            f"Attempting to resolve symbol for '{company_name_or_symbol}' using Gemini.")
        prompt = (f"What is the primary stock ticker symbol for the company named '{company_name_or_symbol}'? "
                  f"Return ONLY the stock ticker symbol itself (e.g., AAPL, GOOGL). "
                  f"If you cannot find a clear, unambiguous primary symbol, return 'NOT_FOUND'.")
        try:
            # Make sure to use a model compatible with the base URL and API key
            # Using gemini-pro as a common example, adjust if needed.
            response = _self.gemini.chat.completions.create(
                # model="gemini-2.0-flash-lite-001", # Use the model provided by user if it works
                # A generally available model via v1beta URL
                model="models/gemini-1.5-flash-latest",
                messages=[
                    {"role": "system", "content": "You are a financial assistant. Provide only the stock ticker symbol."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,  # Symbols are short
                temperature=0.0  # Be precise
            )
            symbol = response.choices[0].message.content.strip().upper()
            _self.logger.info(
                f"Gemini returned: '{symbol}' for '{company_name_or_symbol}'")

            # Validate Gemini's response
            # Basic check
            if not symbol or 'NOT_FOUND' in symbol or len(symbol) > 10 or not symbol.isalnum():
                _self.logger.warning(
                    f"Gemini returned invalid or 'NOT_FOUND' symbol: '{symbol}' for '{company_name_or_symbol}'.")
                return None

            # Final Verification with yfinance for Gemini's result
            try:
                _self.logger.info(
                    f"Verifying Gemini's symbol '{symbol}' with yfinance...")
                test_ticker = yf.Ticker(symbol)
                if test_ticker.info and test_ticker.info.get('regularMarketPrice') is not None:
                    _self.logger.info(
                        f"yfinance verified Gemini's symbol '{symbol}'.")
                    return symbol
                else:
                    _self.logger.warning(
                        f"yfinance did not return valid info for Gemini's symbol '{symbol}'.")
                    return None  # Gemini's symbol wasn't valid according to yfinance
            except Exception as e:
                _self.logger.warning(
                    f"yfinance verification failed for Gemini's symbol '{symbol}': {e}")
                return None  # Error during final check

        except Exception as e:
            _self.logger.error(
                f"Error during Gemini symbol lookup for '{company_name_or_symbol}': {e}", exc_info=True)
            st.error(f"Error looking up stock symbol via AI: {e}")
            return None

        # Should not be reached if logic is correct, but as a fallback
        _self.logger.error(
            f"Symbol resolution failed for '{company_name_or_symbol}' after all checks.")
        return None

    @st.cache_data(show_spinner="Fetching historical stock data...")
    def load_stock_data(_self, symbol: str, start_date: datetime, end_date: datetime, retries: int = 3, delay: int = 2) -> Optional[pd.DataFrame]:
        _self.logger.info(
            f"Fetching data for '{symbol}' from {start_date.strftime(DATE_FORMAT)} to {end_date.strftime(DATE_FORMAT)}")
        start_str = start_date.strftime(DATE_FORMAT)
        # yfinance end is exclusive
        end_str = (end_date + timedelta(days=1)).strftime(DATE_FORMAT)

        for attempt in range(1, retries + 1):
            processed_df = None  # Initialize DataFrame for processed data
            try:
                _self.logger.info(
                    f"Attempt {attempt}/{retries} for '{symbol}'")
                raw_data = yf.download(
                    # Keep auto_adjust=True for now
                    symbol, start=start_str, end=end_str, progress=False, auto_adjust=True)

                # --- Log raw columns immediately ---
                if not raw_data.empty:
                    _self.logger.info(
                        f"Raw columns downloaded for {symbol} (attempt {attempt}): {raw_data.columns}")
                else:
                    _self.logger.warning(
                        f"No data returned by yfinance for '{symbol}' on attempt {attempt}.")
                    if attempt == retries:
                        st.warning(
                            f"Could not retrieve data for {symbol} after {retries} attempts for the specified period.")
                        return None
                    time.sleep(delay)
                    continue  # Go to next retry attempt

                # --- Data Processing Logic ---
                required_set = {'Open', 'High', 'Low', 'Close', 'Volume'}
                temp_processed_data = {}

                if isinstance(raw_data.columns, pd.MultiIndex):
                    _self.logger.warning(
                        f"Unexpected MultiIndex columns received for {symbol} despite auto_adjust=True. Attempting extraction.")
                    # Iterate through tuples in MultiIndex, e.g., ('Close', 'GOOGL')
                    for col_tuple in raw_data.columns:
                        matched_metric = None
                        # Check elements within the tuple for a standard metric name
                        for level_val in col_tuple:
                            if isinstance(level_val, str) and level_val.capitalize() in required_set:
                                matched_metric = level_val.capitalize()
                                break  # Found standard metric

                        if matched_metric and matched_metric not in temp_processed_data:
                            # Assign the data series to the standard metric name
                            temp_processed_data[matched_metric] = raw_data[col_tuple]
                            _self.logger.debug(
                                f"Mapped MultiIndex {col_tuple} to {matched_metric} for {symbol}")

                    # If all required columns were extracted from MultiIndex
                    if required_set.issubset(temp_processed_data.keys()):
                        _self.logger.info(
                            f"Successfully extracted required columns from MultiIndex for {symbol}.")
                        processed_df = pd.DataFrame(temp_processed_data)
                    else:
                        missing_in_multi = required_set - \
                            set(temp_processed_data.keys())
                        _self.logger.error(
                            f"Failed to extract required columns {missing_in_multi} from MultiIndex for {symbol}. Available extracted: {list(temp_processed_data.keys())}")
                        # Don't return yet, maybe simple processing works or Adj Close fallback

                # If not MultiIndex OR MultiIndex extraction failed but we have raw_data
                if processed_df is None:
                    _self.logger.info(
                        f"Processing {symbol} assuming simple columns or as fallback.")
                    # Make a copy to avoid modifying original download cache
                    simple_df = raw_data.copy()
                    # Rename columns case-insensitively first
                    rename_map = {col: col.capitalize()
                                  for col in simple_df.columns if isinstance(col, str)}
                    simple_df.rename(columns=rename_map, inplace=True)
                    _self.logger.debug(
                        f"Columns after capitalization for {symbol}: {simple_df.columns}")

                    # Check if required columns are now present
                    if required_set.issubset(simple_df.columns):
                        _self.logger.info(
                            f"Found required columns in simple structure for {symbol}.")
                        # Select only required cols in standard order
                        processed_df = simple_df[list(required_set)]
                    else:
                        missing_simple = required_set - set(simple_df.columns)
                        _self.logger.warning(
                            f"Required columns {missing_simple} still missing in simple structure for {symbol}.")
                        # --- Fallback: Check for 'Adj Close' ---
                        # yfinance sometimes uses 'Adj close' (lowercase c)
                        if 'Close' in missing_simple and 'Adj close' in simple_df.columns:
                            _self.logger.warning(
                                f"Found 'Adj close' for {symbol}. Attempting to use it as 'Close'.")
                            simple_df['Close'] = simple_df['Adj close']
                            # Check again if required set is now met (excluding the original 'Close')
                            current_cols_set = set(simple_df.columns)
                            if required_set.issubset(current_cols_set):
                                _self.logger.info(
                                    f"Using 'Adj close' successfully substituted 'Close' for {symbol}.")
                                processed_df = simple_df[list(required_set)]
                            else:
                                _self.logger.error(
                                    f"Tried using 'Adj close', but still missing columns for {symbol}. Final available: {list(current_cols_set)}")

                # --- Final Checks on processed_df ---
                if processed_df is None:
                    # If after all attempts, we couldn't get the required columns
                    _self.logger.error(
                        f"Failed to construct DataFrame with required columns for {symbol} after all processing attempts.")
                    st.error(
                        f"Data processing error for {symbol}: Could not obtain required columns (Open, High, Low, Close, Volume). Check logs for details.")
                    return None  # Critical failure

                # --- Post-processing: Index, Date, Numerics ---
                # Ensure index is reset and 'Date' column exists
                processed_df.reset_index(inplace=True)
                date_col_name = 'Date'
                if 'Date' not in processed_df.columns:
                    if 'index' in processed_df.columns:
                        processed_df.rename(
                            columns={'index': 'Date'}, inplace=True)
                    elif 'Datetime' in processed_df.columns:
                        processed_df.rename(
                            columns={'Datetime': 'Date'}, inplace=True)
                    else:
                        _self.logger.error(
                            f"Date column missing in final processed_df for {symbol}. Columns: {processed_df.columns}")
                        st.error(f"Data date processing error for {symbol}.")
                        return None

                # Convert Date column
                try:
                    processed_df["Date"] = pd.to_datetime(
                        processed_df["Date"]).dt.tz_localize(None)
                except Exception as date_err:
                    _self.logger.error(
                        f"Error converting final Date column for {symbol}: {date_err}")
                    st.error(f"Data date format error for {symbol}.")
                    return None

                # Convert numeric columns
                for col in required_set:  # Use the set for iteration
                    processed_df[col] = pd.to_numeric(
                        processed_df[col], errors='coerce')

                # Handle NaNs
                if processed_df[list(required_set)].isnull().any().any():
                    _self.logger.warning(
                        f"NaN values found in final columns for {symbol}. Filling/dropping.")
                    processed_df.ffill(inplace=True)
                    processed_df.dropna(inplace=True)

                if processed_df.empty:
                    _self.logger.error(
                        f"Final DataFrame for {symbol} empty after cleaning.")
                    st.error(
                        f"Data processing error: No valid data rows remained for {symbol}.")
                    return None

                _self.logger.info(
                    f"Successfully loaded and processed {len(processed_df)} rows for '{symbol}'.")
                # Ensure standard column order
                return processed_df[['Date'] + list(required_set)]

            except (JSONDecodeError, RequestException) as net_err:
                _self.logger.error(
                    f"Network error for '{symbol}' on attempt {attempt}: {net_err}")
            except Exception as e:
                _self.logger.exception(
                    f"General error loading/processing data for '{symbol}' on attempt {attempt}: {e}")
                # Show generic error in UI
                st.error(
                    f"An unexpected error occurred fetching data for {symbol}: {e}")

            if attempt < retries:
                _self.logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)

        _self.logger.error(
            f"Failed to load valid data for '{symbol}' after {retries} attempts.")
        st.error(f"Failed to load data for {symbol} after multiple retries.")
        return None

    @st.cache_data(show_spinner="Fetching company information...")
    def load_stock_info(_self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            _self.logger.info(f"Fetching company info for '{symbol}'.")
            stock = yf.Ticker(symbol)
            # .info can sometimes raise errors for obscure tickers or during high traffic
            info = stock.info
            # Added check for market price
            if not info or not isinstance(info, dict) or info.get('regularMarketPrice') is None:
                _self.logger.warning(
                    f"No valid company info returned by yfinance for '{symbol}'. Info: {info}")
                # Don't show Streamlit warning yet, might be okay if only history is needed
                # st.warning(f"Could not retrieve complete company information for {symbol}.")
                # Return info even if partial, let caller decide if it's sufficient
                return info if isinstance(info, dict) else None
            # Log first few keys
            _self.logger.info(
                f"Fetched company info for '{symbol}'. Keys: {list(info.keys())[:5]}...")
            return info
        except Exception as e:
            _self.logger.error(
                f"Error fetching info for '{symbol}': {e}", exc_info=True)
            st.error(f"Error fetching company info for {symbol}: {e}")
            return None

    @st.cache_data(show_spinner="Calculating technical indicators...")
    def calculate_technical_indicators(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates various technical indicators."""
        _self.logger.info("Calculating technical indicators.")
        if df is None or df.empty:
            _self.logger.warning(
                "Input DataFrame is empty. Cannot calculate indicators.")
            return pd.DataFrame()  # Return empty DataFrame
        if not all(col in df.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
            _self.logger.error(
                f"Missing required columns in DataFrame for indicators. Got: {df.columns}")
            return df  # Return original df if columns missing

        df_tech = df.copy()
        # Ensure Date is index for some rolling calculations if needed, but keep it as column too
        # df_tech.set_index('Date', inplace=True, drop=False) # Keep Date column

        # --- SMAs ---
        for window in [20, 50, 200]:
            if len(df_tech) >= window:
                df_tech[f'SMA_{window}'] = df_tech['Close'].rolling(
                    window=window, min_periods=window).mean()
            else:
                df_tech[f'SMA_{window}'] = np.nan

        # --- EMAs ---
        # EMA calculation doesn't strictly require min_periods like rolling
        df_tech['EMA_12'] = df_tech['Close'].ewm(span=12, adjust=False).mean()
        df_tech['EMA_26'] = df_tech['Close'].ewm(span=26, adjust=False).mean()

        # --- RSI ---
        delta = df_tech['Close'].diff()
        gain = delta.where(delta > 0, 0.0)  # Use 0.0 instead of 0
        loss = -delta.where(delta < 0, 0.0)  # Use 0.0 instead of 0

        # Use exponential moving average for RSI calculation for smoother results
        avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
        avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
        df_tech['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        # Fill initial NaNs with neutral 50
        df_tech['RSI'].fillna(50, inplace=True)

        # --- MACD ---
        df_tech['MACD'] = df_tech['EMA_12'] - df_tech['EMA_26']
        df_tech['Signal_Line'] = df_tech['MACD'].ewm(
            span=9, adjust=False).mean()
        df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['Signal_Line']

        # --- Bollinger Bands ---
        bb_window = 20
        if len(df_tech) >= bb_window:
            df_tech['BB_Middle'] = df_tech['Close'].rolling(
                window=bb_window, min_periods=bb_window).mean()
            bb_std = df_tech['Close'].rolling(
                window=bb_window, min_periods=bb_window).std()
            df_tech['BB_Upper'] = df_tech['BB_Middle'] + 2 * bb_std
            df_tech['BB_Lower'] = df_tech['BB_Middle'] - 2 * bb_std
        else:
            df_tech[['BB_Middle', 'BB_Upper', 'BB_Lower']] = np.nan

        # --- Volatility (Standard Deviation) ---
        vol_window = 20
        if len(df_tech) >= vol_window:
            df_tech['Volatility_20'] = df_tech['Close'].rolling(
                # Annualized
                window=vol_window, min_periods=vol_window).std() * np.sqrt(252)
        else:
            df_tech['Volatility_20'] = np.nan

        # --- ATR (Average True Range) ---
        atr_window = 14
        if len(df_tech) >= atr_window:
            high_low = df_tech['High'] - df_tech['Low']
            high_close = np.abs(df_tech['High'] - df_tech['Close'].shift())
            low_close = np.abs(df_tech['Low'] - df_tech['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            # Use exponential moving average for ATR
            df_tech['ATR_14'] = true_range.ewm(
                alpha=1/atr_window, adjust=False).mean()
            # df_tech['ATR_14'] = true_range.rolling(window=atr_window).mean() # Alternative: Simple Moving Average
        else:
            df_tech['ATR_14'] = np.nan

        _self.logger.info("Technical indicators calculated.")
        return df_tech

    def analyze_patterns(_self, df: pd.DataFrame) -> List[str]:
        _self.logger.info("Analyzing technical patterns.")
        patterns = []
        # Need at least 50 days for reliable SMA_50, 200 etc. and comparisons
        min_days_for_patterns = 50
        if df is None or df.empty or len(df) < min_days_for_patterns:
            _self.logger.warning(
                f"Insufficient data ({len(df) if df is not None else 0} rows) for pattern analysis (need {min_days_for_patterns}).")
            return ["Not enough historical data for reliable pattern analysis."]

        # Check required columns exist and have recent data
        required_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI',
                         'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']
        # Check the *last* row for NaN in required cols
        latest = df.iloc[-1]
        missing_latest = [
            col for col in required_cols if pd.isna(latest.get(col))]
        if missing_latest:
            _self.logger.warning(
                f"Latest data missing for required indicator columns: {missing_latest}. Pattern analysis might be incomplete.")
            # Proceed with available data but maybe add a warning pattern?
            # patterns.append("Warning: Some recent indicator data missing, analysis might be incomplete.")

        # Use last two points for crossover checks if available
        previous = df.iloc[-2] if len(df) > 1 else latest

        # --- Pattern Logic ---

        # SMA Golden/Death Cross (Requires SMA_20 and SMA_50)
        if pd.notna(latest.get('SMA_20')) and pd.notna(latest.get('SMA_50')) and \
           pd.notna(previous.get('SMA_20')) and pd.notna(previous.get('SMA_50')):
            if latest['SMA_20'] > latest['SMA_50'] and previous['SMA_20'] <= previous['SMA_50']:
                patterns.append(
                    "Golden Cross (SMA 20 crossed above SMA 50) - Potential Bullish Signal")
            elif latest['SMA_20'] < latest['SMA_50'] and previous['SMA_20'] >= previous['SMA_50']:
                patterns.append(
                    "Death Cross (SMA 20 crossed below SMA 50) - Potential Bearish Signal")

        # RSI Overbought/Oversold (Requires RSI)
        if pd.notna(latest.get('RSI')):
            if latest['RSI'] > 70:
                patterns.append(
                    f"Overbought (RSI = {latest['RSI']:.2f}) - Potential for Price Pullback")
            elif latest['RSI'] < 30:
                patterns.append(
                    f"Oversold (RSI = {latest['RSI']:.2f}) - Potential for Price Bounce")

        # MACD Crossover (Requires MACD and Signal_Line)
        if pd.notna(latest.get('MACD')) and pd.notna(latest.get('Signal_Line')) and \
           pd.notna(previous.get('MACD')) and pd.notna(previous.get('Signal_Line')):
            if latest['MACD'] > latest['Signal_Line'] and previous['MACD'] <= previous['Signal_Line']:
                patterns.append(
                    "MACD Bullish Crossover (MACD crossed above Signal Line)")
            elif latest['MACD'] < latest['Signal_Line'] and previous['MACD'] >= previous['Signal_Line']:
                patterns.append(
                    "MACD Bearish Crossover (MACD crossed below Signal Line)")

        # Bollinger Bands Breach (Requires Close, BB_Upper, BB_Lower)
        if pd.notna(latest.get('Close')) and pd.notna(latest.get('BB_Upper')) and pd.notna(latest.get('BB_Lower')):
            if latest['Close'] > latest['BB_Upper']:
                patterns.append(
                    "Price above Upper Bollinger Band - Potential Overextension or Strong Momentum")
            elif latest['Close'] < latest['BB_Lower']:
                patterns.append(
                    "Price below Lower Bollinger Band - Potential Oversold or Strong Downtrend")

        # --- Add more patterns as needed ---
        # Example: Volume Spike
        # avg_volume_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        # if pd.notna(latest.get('Volume')) and pd.notna(avg_volume_20) and latest['Volume'] > avg_volume_20 * 2:
        #     patterns.append(f"High Volume Spike ({latest['Volume']:,.0f} vs avg {avg_volume_20:,.0f}) - Indicates Strong Interest")

        if not patterns:
            patterns.append(
                "No significant standard technical patterns detected in the recent data.")

        _self.logger.info(f"Patterns identified: {len(patterns)}")
        return patterns

    @st.cache_data(show_spinner="Fetching news articles...")
    def load_news(_self, query: str, from_date: datetime, to_date: datetime, page_size: int = 50) -> List[Dict[str, Any]]:
        if not _self.newsapi:
            _self.logger.warning(
                "NewsAPI client not available. Cannot fetch news.")
            st.error("NewsAPI client is not configured. News feature disabled.")
            return []  # Return empty list

        # NewsAPI 'everything' endpoint limitation: max 30 days back for free/dev plans
        max_days_back = 30
        earliest_allowed_from_date = datetime.now() - timedelta(days=max_days_back)

        # Adjust from_date if it's too far back
        adjusted_from_date = max(from_date, earliest_allowed_from_date)
        if adjusted_from_date > from_date:
            _self.logger.warning(
                f"News search 'from_date' adjusted from {from_date.strftime(DATE_FORMAT)} to {adjusted_from_date.strftime(DATE_FORMAT)} due to API limitations.")
            st.info(
                f"Note: News search is limited to the last {max_days_back} days.")

        # Ensure to_date is not in the future and from_date is before to_date
        adjusted_to_date = min(to_date, datetime.now())
        if adjusted_from_date >= adjusted_to_date:
            _self.logger.warning(
                f"Invalid date range for news: from={adjusted_from_date}, to={adjusted_to_date}. Using last 2 days.")
            adjusted_from_date = datetime.now() - timedelta(days=2)
            adjusted_to_date = datetime.now()

        _self.logger.info(
            f"Fetching news for '{query}' from {adjusted_from_date.strftime(DATE_FORMAT)} to {adjusted_to_date.strftime(DATE_FORMAT)}")
        try:
            all_articles = _self.newsapi.get_everything(
                q=query,
                # Use adjusted dates
                from_param=adjusted_from_date.strftime(DATE_FORMAT),
                to=adjusted_to_date.strftime(DATE_FORMAT),
                language='en',
                sort_by='relevancy',  # 'publishedAt' or 'popularity' are other options
                page_size=min(page_size, 100)  # API max is 100
            )
            articles = all_articles.get('articles', [])
            _self.logger.info(
                f"Fetched {len(articles)} news articles for '{query}'. Total results: {all_articles.get('totalResults')}")
            # Basic filtering for relevance (optional)
            # articles = [a for a in articles if query.lower() in (a.get('title','')+a.get('description','')).lower()]
            return articles
        except Exception as e:
            _self.logger.error(
                f"Error fetching news for '{query}': {e}", exc_info=True)
            st.error(f"Error fetching news from NewsAPI: {e}")
            return []  # Return empty list on error

    @st.cache_data(show_spinner="Analyzing news sentiment...")
    def analyze_sentiment(_self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, Optional[pd.DataFrame]]:
        if not _self.sentiment_available or _self.sia is None:
            _self.logger.warning(
                "Sentiment analysis skipped; VADER analyzer not available.")
            st.warning(
                "Sentiment analysis tool (VADER) is not available. Skipping sentiment calculation.")
            # Return structure consistent with success, but with neutral/empty values
            return articles, 0.0, None

        if not articles:
            _self.logger.info("No articles provided for sentiment analysis.")
            return articles, 0.0, None

        _self.logger.info(f"Analyzing sentiment for {len(articles)} articles.")
        sentiments = []
        analyzed_articles = []
        daily_sentiment_scores = {}  # date_str -> list of compound scores

        for article in articles:
            # Prioritize title and description for VADER's effectiveness (often better than full content)
            title = article.get('title', '') or ""
            description = article.get('description', '') or ""
            # Use content only if title and description are both missing
            # Ensure content is string
            content = article.get('content', '') or ""

            text_to_analyze = f"{title}. {description}".strip()
            if not text_to_analyze and content:
                # Use first ~280 chars of content if title/desc empty (VADER limit context)
                text_to_analyze = content[:280].strip()

            # Avoid modifying original list elements directly
            article_with_sentiment = article.copy()

            if text_to_analyze:
                try:
                    sentiment_scores = _self.sia.polarity_scores(
                        text_to_analyze)
                    compound_score = sentiment_scores['compound']
                    article_with_sentiment['sentiment'] = sentiment_scores
                    sentiments.append(compound_score)

                    # Store daily sentiment
                    pub_date_str = article.get('publishedAt', '')[
                        :10]  # YYYY-MM-DD
                    if pub_date_str:
                        try:
                            # Validate date format before using
                            datetime.strptime(pub_date_str, DATE_FORMAT)
                            daily_sentiment_scores.setdefault(
                                pub_date_str, []).append(compound_score)
                        except ValueError:
                            _self.logger.warning(
                                f"Invalid date format '{pub_date_str}' in article, skipping for daily sentiment.")

                except Exception as e:
                    _self.logger.warning(
                        f"Sentiment analysis failed for article titled '{title[:50]}...': {e}")
                    article_with_sentiment['sentiment'] = {
                        'compound': 0.0, 'error': str(e)}
                    sentiments.append(0.0)  # Append neutral score on error
            else:
                # No text found to analyze
                article_with_sentiment['sentiment'] = {
                    'compound': 0.0, 'error': 'No text content available'}
                sentiments.append(0.0)

            analyzed_articles.append(article_with_sentiment)

        # Calculate overall average sentiment
        avg_sentiment = sum(sentiments) / \
            len(sentiments) if sentiments else 0.0

        # Prepare daily sentiment DataFrame
        daily_data = []
        if daily_sentiment_scores:
            # Sort by date string
            for date_str in sorted(daily_sentiment_scores.keys()):
                scores = daily_sentiment_scores[date_str]
                daily_avg = sum(scores) / len(scores) if scores else 0.0
                daily_data.append({'Date': pd.to_datetime(
                    date_str), 'Daily_Sentiment': daily_avg})

        sentiment_df = pd.DataFrame(daily_data) if daily_data else None
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df.set_index('Date', inplace=True)
            # Optional: Calculate rolling average of daily sentiment
            # sentiment_df['Rolling_Sentiment_3D'] = sentiment_df['Daily_Sentiment'].rolling(window=3).mean()

        _self.logger.info(
            f"Sentiment analysis complete. Avg score: {avg_sentiment:.3f}. Daily points: {len(daily_data)}")
        return analyzed_articles, avg_sentiment, sentiment_df

    # --- FIX: Integrate Prophet Forecast Method ---

    # Inside the EnhancedStockAnalyzer class:

    # Updated spinner text slightly
    @st.cache_data(show_spinner="Generating Prophet forecast with sentiment...")
    def prophet_forecast(_self, df: pd.DataFrame, days_to_predict: int) -> Optional[pd.DataFrame]:
        _self.logger.info(
            # Keep original log
            f"Starting Prophet forecast for {days_to_predict} days.")

        # --- Basic Input Checks ---
        if df is None or df.empty:
            _self.logger.warning("Input DataFrame empty for Prophet forecast.")
            st.warning("Cannot generate forecast: Input data is missing.")
            return None
        if 'Date' not in df.columns or 'Close' not in df.columns:
            _self.logger.error(
                f"Prophet requires 'Date'/'Close' columns. Found: {df.columns}")
            st.error("Cannot generate forecast: Missing 'Date' or 'Close' column.")
            return None
        min_data_points = 30
        if len(df) < min_data_points:
            _self.logger.warning(
                f"Insufficient data ({len(df)} points) for Prophet (need {min_data_points}).")
            st.warning(
                f"Not enough historical data ({len(df)} days) for reliable forecasting. Need at least {min_data_points}.")
            return None
        if not isinstance(days_to_predict, int) or days_to_predict <= 0:
            _self.logger.warning(
                f"Invalid 'days_to_predict': {days_to_predict}. Must be positive.")
            st.error(
                "Invalid number of days to forecast. Please select a positive number.")
            return None

        try:
            # --- Prepare base prophet_df (ds, y) ---
            prophet_df = df[['Date', 'Close']].copy()
            prophet_df.rename(
                columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            # Ensure 'ds' is datetime without timezone for merging
            prophet_df['ds'] = pd.to_datetime(
                prophet_df['ds']).dt.tz_localize(None)

            # --- SENTIMENT REGRESSOR: Retrieve and Prepare Sentiment Data ---
            sentiment_df = st.session_state.get('sentiment_df')
            sentiment_regressor_added = False
            last_sentiment = 0.0  # Default to neutral if no sentiment data

            if sentiment_df is not None and isinstance(sentiment_df, pd.DataFrame) and not sentiment_df.empty:
                _self.logger.info(
                    "Sentiment data found, preparing as regressor.")
                # Ensure sentiment_df index is datetime and matches 'ds' format
                sentiment_prep = sentiment_df[['Daily_Sentiment']].copy()
                sentiment_prep.index = pd.to_datetime(
                    sentiment_prep.index).tz_localize(None)
                sentiment_prep.rename(
                    columns={'Daily_Sentiment': 'sentiment_score'}, inplace=True)
                sentiment_prep.index.name = 'ds'  # Rename index for merging

                # Left merge sentiment onto the historical price data
                prophet_df = pd.merge(
                    prophet_df, sentiment_prep, on='ds', how='left')

                # Handle NaNs in the sentiment score
                # 1. Forward fill: Assume sentiment persists until new news comes
                prophet_df['sentiment_score'].ffill(inplace=True)
                # 2. Fill any remaining NaNs at the beginning with neutral (0.0)
                prophet_df['sentiment_score'].fillna(0.0, inplace=True)

                # Check if sentiment score column has non-NaN values after processing
                if prophet_df['sentiment_score'].notna().any():
                    # Get the last known sentiment score for future prediction
                    last_sentiment = prophet_df['sentiment_score'].iloc[-1]
                    sentiment_regressor_added = True
                    _self.logger.info(
                        f"Sentiment regressor added. Last sentiment value: {last_sentiment:.3f}")
                else:
                    _self.logger.warning(
                        "Sentiment data merged but resulted in all NaNs after processing. Proceeding without sentiment regressor.")
                    # Remove the column if it's all NaN to avoid issues
                    if 'sentiment_score' in prophet_df.columns:
                        prophet_df.drop(
                            columns=['sentiment_score'], inplace=True)

            else:
                _self.logger.warning(
                    "Sentiment data (sentiment_df) not available or empty in session state. Proceeding without sentiment regressor.")
            # --- END SENTIMENT REGRESSOR PREP ---

            # --- Log Transform (Optional - Keep or remove as per previous tests) ---
            use_log_transform = True  # Set to False if you want to test without it
            if use_log_transform:
                _self.logger.info("Applying log transform to 'y' data.")
                epsilon = 1e-9
                prophet_df['y_orig'] = prophet_df['y']
                prophet_df['y'] = np.log(prophet_df['y'] + epsilon)

            # --- Initialize Prophet Model ---
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                interval_width=0.95,
                changepoint_prior_scale=0.005  # Keeping the tuned value from previous suggestion
            )

            # --- SENTIMENT REGRESSOR: Add to Model ---
            if sentiment_regressor_added:
                model.add_regressor('sentiment_score')
                _self.logger.info(
                    "Added 'sentiment_score' regressor to Prophet model.")

            # --- Fit Model ---
            # Prophet automatically uses columns added via add_regressor if they exist in the fitting df
            # Pass the DataFrame containing ds, y, and potentially sentiment_score
            model.fit(prophet_df)

            # --- Create Future DataFrame ---
            future = model.make_future_dataframe(periods=days_to_predict)

            # --- SENTIMENT REGRESSOR: Add Future Values ---
            if sentiment_regressor_added:
                # Assume the last known sentiment persists into the future
                future['sentiment_score'] = last_sentiment
                _self.logger.info(
                    f"Added future sentiment values ({last_sentiment:.3f}) to prediction frame.")

            # --- Make Prediction ---
            forecast = model.predict(future)

            # --- Inverse Transform (if log transform was used) ---
            if use_log_transform:
                _self.logger.info(
                    "Applying inverse log transform to forecast.")
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    if col in forecast.columns:
                        forecast[col] = np.exp(forecast[col]) - epsilon
                        forecast[col] = forecast[col].clip(
                            lower=0.0)  # Ensure non-negative
            else:
                # Still clip if not using log transform
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    if col in forecast.columns:
                        forecast[col] = forecast[col].clip(lower=0.0)

            _self.logger.info(
                "Prophet forecast generated successfully (with sentiment integration attempt).")
            # Return relevant columns
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        except Exception as e:
            _self.logger.exception(
                f"Error during Prophet forecasting (with sentiment attempt): {e}")
            st.error(f"Failed to generate Prophet forecast: {e}")
            return None

    @st.cache_data(show_spinner="Generating Random Forest prediction...")
    def machine_learning_prediction(_self, symbol: str, features: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']) -> Optional[Dict[str, Any]]:
        _self.logger.info(f"Starting ML prediction for {symbol}.")
        try:
            # Fetch 5 years of data for more robust training
            data = yf.download(symbol, period='5y',
                               auto_adjust=True, progress=False)

            if data.empty:
                _self.logger.warning(
                    f"No data downloaded for ML prediction for {symbol}.")
                st.warning(
                    f"Could not download data for {symbol} for ML prediction.")
                return None

            # Standardize column names
            data.columns = [col.capitalize() for col in data.columns]

            # Basic Feature Engineering (Examples)
            # Add more relevant features: lagged values, rolling means, volatility, etc.
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['Close_lag1'] = data['Close'].shift(1)
            data['Volume_lag1'] = data['Volume'].shift(1)
            # Example: Price change %
            data['Pct_Change'] = data['Close'].pct_change()

            # Define features to use (including engineered ones)
            # Ensure engineered features are calculated before this list is used
            features_with_eng = features + \
                ['SMA_20', 'SMA_50', 'Close_lag1', 'Volume_lag1', 'Pct_Change']
            # Filter out features that might not exist (e.g., if download failed partially)
            features_present = [
                f for f in features_with_eng if f in data.columns]

            # Target variable: Predict the *next* day's close price
            data['Target'] = data['Close'].shift(-1)

            # Drop rows with NaNs resulting from shifts or rolling calculations
            data_clean = data.dropna()

            if len(data_clean) < 60:  # Need enough data after cleaning
                _self.logger.warning(
                    f"Insufficient data ({len(data_clean)} rows) after feature engineering/cleaning for {symbol}.")
                st.warning(
                    "Not enough data remains after feature engineering for robust ML prediction.")
                return None

            X = data_clean[features_present]
            y = data_clean['Target']
            original_index = data_clean.index  # Keep track of dates for plotting

            # Split data chronologically (important for time series)
            test_size = 0.2
            split_index = int(len(X) * (1 - test_size))

            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            test_dates = original_index[split_index:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest model
            # Hyperparameters can be tuned (e.g., using GridSearchCV)
            rf_model = RandomForestRegressor(
                n_estimators=150,         # More trees
                random_state=42,
                n_jobs=-1,                # Use all available cores
                max_depth=10,             # Limit tree depth to prevent overfitting
                min_samples_split=10,     # Min samples to split a node
                min_samples_leaf=5        # Min samples in a leaf node
            )
            rf_model.fit(X_train_scaled, y_train)
            _self.logger.info(f"Trained Random Forest model for {symbol}.")

            # Make predictions on the test set
            predictions = rf_model.predict(X_test_scaled)

            # Evaluate the model
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))  # Mean Absolute Error

            # Feature Importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,  # Use columns from X_train
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Predict the *next* day's price based on the *last available* data row
            # Use original 'data' before dropna
            last_row_features_unscaled = data[features_present].iloc[-1:]
            next_day_prediction = None
            if not last_row_features_unscaled.isnull().any().any():
                last_scaled = scaler.transform(
                    last_row_features_unscaled.values)  # Scale the last row
                next_day_prediction = rf_model.predict(last_scaled)[0]
                _self.logger.info(
                    f"Predicted next day close for {symbol}: {next_day_prediction}")
            else:
                _self.logger.warning(
                    f"Could not make next day prediction for {symbol} due to NaNs in last feature row.")

            results = {
                'test_dates': test_dates,
                'y_test': y_test,
                'predictions': predictions,
                'rmse': rmse,
                'mae': mae,  # Added MAE
                'feature_importance': feature_importance,
                'next_day_prediction': next_day_prediction,
                # Get the very last actual close
                'last_actual_close': data['Close'].iloc[-1],
                'features_used': features_present  # Record features used
            }
            _self.logger.info("ML prediction completed successfully.")
            return results

        except Exception as e:
            _self.logger.exception(
                f"Error during ML prediction for {symbol}: {e}")
            st.error(f"Error during Machine Learning prediction: {e}")
            return None

    def _fetch_portfolio_data(_self, symbols: List[str], period: str = '1y') -> Dict[str, pd.Series]:
        """ Helper to fetch adjusted close data for multiple symbols, handling errors."""
        valid_data = {}
        _self.logger.info(
            f"Fetching portfolio data for: {symbols} over period: {period}")

        if not symbols:
            return {}

        # Fetch data for all symbols at once
        # Use auto_adjust=True for simplicity (handles splits/dividends in 'Close')
        all_data = yf.download(symbols, period=period,
                               auto_adjust=True, progress=False)

        if all_data.empty:
            st.warning(
                f"No data returned by yfinance for symbols: {', '.join(symbols)}")
            _self.logger.warning(f"No portfolio data returned for: {symbols}")
            return {}

        # Determine if the result is for single or multiple tickers
        if len(symbols) == 1:
            # If only one symbol, columns might not be MultiIndex
            if 'Close' in all_data.columns:
                close_data_single = all_data[['Close']].rename(
                    columns={'Close': symbols[0]})
                if not close_data_single[symbols[0]].isnull().all():
                    valid_data[symbols[0]
                               ] = close_data_single[symbols[0]].dropna()
                else:
                    st.warning(
                        f"No valid 'Close' data for {symbols[0]} in single download.")
                    _self.logger.warning(
                        f"No 'Close' data for {symbols[0]} in single download.")
            else:
                st.warning(
                    f"'Close' column missing for single symbol download: {symbols[0]}")
                _self.logger.warning(
                    f"'Close' column missing for single symbol download: {symbols[0]}")
        elif isinstance(all_data.columns, pd.MultiIndex) and 'Close' in all_data.columns.levels[0]:
            # Multiple symbols usually result in MultiIndex columns ('Close', 'AAPL'), ('Close', 'GOOG') etc.
            close_data_multi = all_data['Close']  # Select the 'Close' level
            for symbol in symbols:
                if symbol in close_data_multi.columns:
                    series = close_data_multi[symbol].dropna()
                    if not series.empty:
                        valid_data[symbol] = series
                    else:
                        st.warning(
                            f"No valid 'Close' data for {symbol} after dropping NaNs.")
                        _self.logger.warning(
                            f"No 'Close' data for {symbol} after dropping NaNs.")
                else:
                    st.warning(
                        f"Symbol {symbol} missing from downloaded 'Close' data columns.")
                    _self.logger.warning(
                        f"Symbol {symbol} missing in portfolio 'Close' data columns.")
        else:
            # Fallback/Error case: Unexpected structure
            st.warning(
                f"Unexpected data structure returned by yfinance for multiple symbols: {symbols}. Columns: {all_data.columns}")
            _self.logger.warning(
                f"Unexpected data structure for portfolio: {symbols}. Columns: {all_data.columns}")

        _self.logger.info(
            f"Successfully fetched data for {len(valid_data)} out of {len(symbols)} portfolio symbols.")
        return valid_data

    @st.cache_data(show_spinner="Simulating portfolio performance...")
    def portfolio_simulation(_self, symbols: List[str], initial_investment: float, investment_strategy: str = 'equal_weight') -> Optional[Dict[str, Any]]:
        _self.logger.info(
            f"Starting portfolio simulation for {symbols} using strategy '{investment_strategy}'.")

        # 1. Fetch Data
        valid_data_dict = _self._fetch_portfolio_data(symbols, period='1y')

        if not valid_data_dict:
            st.error(
                "No valid historical data found for any portfolio symbols. Cannot run simulation.")
            _self.logger.error(
                "Portfolio simulation failed: no valid data fetched.")
            return None

        actual_symbols_used = list(valid_data_dict.keys())
        if len(actual_symbols_used) < len(symbols):
            missing = set(symbols) - set(actual_symbols_used)
            st.warning(
                f"Could not retrieve data for: {', '.join(missing)}. Simulation run with: {', '.join(actual_symbols_used)}.")
            _self.logger.warning(
                f"Portfolio simulation excluded symbols: {', '.join(missing)}.")

        # Should be caught by valid_data_dict check, but belt-and-suspenders
        if len(actual_symbols_used) < 1:
            st.error("No symbols remain after data fetch. Cannot run simulation.")
            return None

        # 2. Calculate Returns DataFrame
        # Align data by date index and calculate percentage change
        portfolio_df = pd.DataFrame(valid_data_dict)
        # Drop first row with NaNs from pct_change
        portfolio_returns = portfolio_df.pct_change().dropna()

        if portfolio_returns.empty:
            st.error(
                "Could not calculate portfolio returns (possibly insufficient overlapping data).")
            _self.logger.error(
                "Portfolio simulation failed: empty returns DataFrame after pct_change/dropna.")
            return None

        # 3. Determine Weights
        num_stocks = len(actual_symbols_used)
        weights = np.ones(num_stocks) / num_stocks  # Default: Equal weight

        if investment_strategy == 'market_cap_weighted':
            _self.logger.info("Calculating Market Cap weights.")
            market_caps = []
            temp_weights = {}  # Store valid caps temporarily
            missing_caps_symbols = []

            with st.spinner("Fetching market caps for weighting..."):
                for stock in actual_symbols_used:
                    try:
                        # Use cached info if possible, otherwise fetch
                        stock_info_local = _self.load_stock_info(stock)
                        cap = stock_info_local.get(
                            'marketCap') if stock_info_local else None

                        if cap is None or not isinstance(cap, (int, float)) or cap <= 0:
                            _self.logger.warning(
                                f"Missing or invalid marketCap for {stock}. Will use equal weight fallback for this stock.")
                            missing_caps_symbols.append(stock)
                            # Assign a placeholder weight of 0 for now, will adjust later
                            temp_weights[stock] = 0
                        else:
                            market_caps.append(cap)
                            temp_weights[stock] = cap
                    except Exception as e:
                        _self.logger.warning(
                            f"Error fetching market cap for {stock}: {e}. Using fallback.")
                        missing_caps_symbols.append(stock)
                        temp_weights[stock] = 0

            # Sum of caps for stocks where it was found
            total_valid_cap = sum(temp_weights.values())

            if total_valid_cap > 0:
                # Calculate weights based on market cap for valid stocks
                final_weights_dict = {
                    stock: cap / total_valid_cap for stock, cap in temp_weights.items() if cap > 0}
                # If some caps were missing, distribute the remaining weight equally among them
                # This part seems overly complex in the original - let's simplify:
                # If caps missing, maybe fall back to equal weight for ALL stocks? Or warn user?

                # Simpler approach: If *any* market cap is missing, maybe fall back to equal weight entirely?
                if missing_caps_symbols:
                    st.warning(
                        f"Market cap data missing/invalid for: {', '.join(missing_caps_symbols)}. Falling back to EQUAL WEIGHT strategy for all stocks in the portfolio.")
                    _self.logger.warning(
                        f"Market cap missing for {missing_caps_symbols}, falling back to equal weight.")
                    # Reset to equal weight
                    weights = np.ones(num_stocks) / num_stocks
                else:
                    # Use calculated market cap weights if all were found
                    weights_array = np.array(
                        [final_weights_dict.get(s, 0) for s in actual_symbols_used])
                    weights = weights_array / weights_array.sum()  # Ensure sums to 1
                    _self.logger.info("Using market cap weighted strategy.")

            else:
                # All market caps were missing or invalid
                st.warning(
                    "Could not retrieve valid market caps for any stock. Falling back to EQUAL WEIGHT strategy.")
                _self.logger.warning(
                    "All market caps missing, falling back to equal weight.")
                # Reset to equal weight
                weights = np.ones(num_stocks) / num_stocks

        _self.logger.info(
            f"Final portfolio weights ({investment_strategy}): {dict(zip(actual_symbols_used, weights))}")

        # 4. Calculate Portfolio Metrics
        daily_portfolio_returns = (portfolio_returns * weights).sum(axis=1)

        # Cumulative Value (starts from initial investment)
        cumulative_value = (
            1 + daily_portfolio_returns).cumprod() * initial_investment
        # Handle potential initial NaNs if cumprod starts with NaN
        cumulative_value.fillna(initial_investment, inplace=True)

        # Ensure we have enough data points for annualization (e.g., > 1)
        if len(daily_portfolio_returns) > 1:
            annual_volatility = daily_portfolio_returns.std() * np.sqrt(252)  # 252 trading days
            annual_return = daily_portfolio_returns.mean() * 252
            # Sharpe Ratio (assuming risk-free rate = 0)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0.0
        else:
            _self.logger.warning(
                "Not enough return data points (<2) to calculate annualized metrics.")
            annual_volatility = 0.0
            annual_return = 0.0
            sharpe_ratio = 0.0

        _self.logger.info("Portfolio simulation calculations complete.")
        return {
            'daily_returns': daily_portfolio_returns,
            'cumulative_value': cumulative_value,
            'annualized_volatility': annual_volatility,
            'annualized_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            # Store final weights used
            'weights': dict(zip(actual_symbols_used, weights)),
            'symbols_used': actual_symbols_used  # Store symbols actually included
        }

    @st.cache_data(show_spinner="Calculating correlation matrix...")
    def advanced_correlation_analysis(_self, symbols: List[str]) -> Optional[pd.DataFrame]:
        _self.logger.info(f"Starting correlation analysis for {symbols}.")

        # 1. Fetch Data
        valid_data_dict = _self._fetch_portfolio_data(symbols, period='1y')

        if not valid_data_dict:
            st.error("No valid historical data found for correlation analysis.")
            _self.logger.error(
                "Correlation analysis failed: no valid data fetched.")
            return None

        actual_symbols_used = list(valid_data_dict.keys())
        if len(actual_symbols_used) < 2:
            st.warning(
                "Correlation analysis requires at least two stocks with valid data.")
            _self.logger.warning(
                f"Correlation analysis skipped: only {len(actual_symbols_used)} stock(s) had data.")
            return None

        if len(actual_symbols_used) < len(symbols):
            missing = set(symbols) - set(actual_symbols_used)
            st.warning(
                f"Correlation excluded symbols with missing data: {', '.join(missing)}.")

        # 2. Calculate Returns and Correlation
        portfolio_df = pd.DataFrame(valid_data_dict)
        returns_df = portfolio_df.pct_change().dropna()

        # Need >1 row for correlation
        if returns_df.empty or len(returns_df) < 2:
            st.error(
                "Insufficient overlapping data points to calculate correlations.")
            _self.logger.error(
                "Correlation analysis failed: empty returns DF or <2 rows.")
            return None

        correlation_matrix = returns_df.corr()
        _self.logger.info("Correlation matrix computed successfully.")
        return correlation_matrix

    @st.cache_data(show_spinner="Fetching ESG scores...")
    def esg_scoring(_self, symbol: str) -> Optional[Dict[str, float]]:
        _self.logger.info(f"Fetching ESG scores for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            # Access sustainability data
            esg_data = ticker.sustainability

            if esg_data is not None and not esg_data.empty and isinstance(esg_data, pd.DataFrame):
                # Convert index to lowercase for consistent matching
                esg_data.index = esg_data.index.str.lower()

                _self.logger.info(
                    f"DEBUG ESG: Columns found: {esg_data.columns}")
                _self.logger.info(
                    f"DEBUG ESG: Index values: {esg_data.index.tolist()}")
                _self.logger.info(
                    f"DEBUG ESG: DataFrame head:\n{esg_data.head().to_string()}")

                scores = {}
                # Define potential keys from yfinance (these can change)
                score_map = {
                    'totalesg': 'Total ESG Score',
                    'environmentscore': 'Environmental Score',
                    'socialscore': 'Social Score',
                    'governancescore': 'Governance Score',
                    'esgperformance': 'ESG Performance',  # Sometimes used
                    # Add other potential keys if observed: 'controversylevel', etc.
                }

                # Iterate through the map and extract scores if present
                for key, label in score_map.items():
                    if key in esg_data.index:
                        # --- START REPLACEMENT ---
                        correct_column_name = 'esgScores'  # The column name identified from logs

                        if correct_column_name in esg_data.columns:
                            # Get value using the correct column name
                            value = esg_data.loc[key, correct_column_name]

                            if pd.api.types.is_number(value) and pd.notna(value):
                                scores[label] = float(value)
                            else:
                                _self.logger.warning(
                                    f"Found ESG key '{key}' for {symbol}, but value '{value}' in column '{correct_column_name}' is not numeric.")
                        else:
                            # This case shouldn't happen based on logs, but good practice
                            _self.logger.warning(
                                f"Score column '{correct_column_name}' not found in ESG data for {symbol}, though expected. Columns: {esg_data.columns}")
                        # --- END REPLACEMENT ---

                if scores:
                    _self.logger.info(
                        f"Actual ESG scores found for {symbol}: {scores}")
                    st.info(f"ESG scores found for {symbol}.")
                    return scores
                else:
                    _self.logger.warning(
                        f"Sustainability data structure found for {symbol}, but no recognized/valid ESG score keys matched. Index: {list(esg_data.index)}")
                    st.warning(
                        f"Could not extract standard ESG scores for {symbol}, although some sustainability data might exist.")
                    # Optionally display raw data: st.dataframe(esg_data)
                    return None  # Indicate that standard scores weren't found

            else:
                # ticker.sustainability was None or empty
                _self.logger.info(
                    f"No ESG / sustainability data returned by yfinance for {symbol}.")
                st.warning(
                    f"No ESG / sustainability data found for {symbol} via yfinance.")
                return None  # Indicate no data found

        except Exception as e:
            # Catch errors during yfinance call or data processing
            _self.logger.error(
                f"Error fetching or processing ESG data for {symbol}: {e}", exc_info=True)
            st.error(
                f"An error occurred while trying to retrieve ESG data for {symbol}: {e}")
            return None  # Indicate error

    @st.cache_data(show_spinner="Fetching earnings calendar...")
    def get_earnings_calendar(_self, symbol: str) -> Optional[pd.DataFrame]:
        _self.logger.info(f"Fetching earnings calendar for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            # .calendar can return a DataFrame or sometimes a Dict
            cal_data = ticker.calendar

            if cal_data is None:
                _self.logger.info(
                    f"No earnings calendar data structure returned by yfinance for {symbol}.")
                # st.info(f"No upcoming earnings data found for {symbol}.") # Display handled later
                return None

            cal_df = None

            # --- Handle DataFrame response ---
            if isinstance(cal_data, pd.DataFrame) and not cal_data.empty:
                _self.logger.info(
                    f"Received earnings calendar as DataFrame for {symbol}.")
                cal_df = cal_data.copy()
                # Ensure 'Earnings Date' exists and format it
                if 'Earnings Date' in cal_df.columns:
                    cal_df['Earnings Date'] = pd.to_datetime(
                        cal_df['Earnings Date'], errors='coerce')
                    cal_df.dropna(subset=['Earnings Date'], inplace=True)
                    # Convert other numeric columns safely
                    num_cols = ['Earnings Average', 'Earnings Low', 'Earnings High',
                                'Revenue Average', 'Revenue Low', 'Revenue High']
                    for col in num_cols:
                        if col in cal_df.columns:
                            cal_df[col] = pd.to_numeric(
                                cal_df[col], errors='coerce')
                else:
                    _self.logger.warning(
                        f"'Earnings Date' column missing in DataFrame calendar for {symbol}.")
                    cal_df = None  # Invalidate if date is missing

            # --- Handle Dictionary response (less common now, but good fallback) ---
            elif isinstance(cal_data, dict) and cal_data:
                _self.logger.info(
                    f"Received earnings calendar as Dictionary for {symbol}.")
                # Try to parse dictionary structure (this might need adjustment based on actual dict format)
                earnings_section = cal_data.get('Earnings')
                if earnings_section and isinstance(earnings_section, dict):
                    dates_ts = earnings_section.get(
                        'earningsDate', [])  # List of timestamps
                    processed_data = []
                    if dates_ts:
                        for ts in dates_ts:
                            try:
                                date_obj = pd.to_datetime(
                                    ts, unit='s').date()  # Assuming timestamps
                                row = {'Earnings Date': date_obj}
                                # Extract other fields if they exist
                                row['Earnings Average'] = pd.to_numeric(
                                    earnings_section.get('earningsAverage'), errors='coerce')
                                row['Earnings Low'] = pd.to_numeric(
                                    earnings_section.get('earningsLow'), errors='coerce')
                                row['Earnings High'] = pd.to_numeric(
                                    earnings_section.get('earningsHigh'), errors='coerce')
                                row['Revenue Average'] = pd.to_numeric(
                                    earnings_section.get('revenueAverage'), errors='coerce')
                                # Add Revenue Low/High if available in dict structure
                                processed_data.append(row)
                            except Exception as date_err:
                                _self.logger.warning(
                                    f"Error parsing earnings timestamp '{ts}' for {symbol} from dict: {date_err}")
                        if processed_data:
                            cal_df = pd.DataFrame(processed_data)
                            cal_df['Earnings Date'] = pd.to_datetime(
                                cal_df['Earnings Date'])  # Ensure datetime
                    else:
                        _self.logger.info(
                            f"Dictionary calendar found for {symbol}, but 'earningsDate' list is missing or empty.")
                else:
                    _self.logger.info(
                        f"Dictionary calendar found for {symbol}, but 'Earnings' section is missing or invalid.")

            # --- Final Processing for valid DataFrame ---
            if cal_df is not None and not cal_df.empty:
                # Keep only relevant columns that actually exist
                potential_cols = ['Earnings Date', 'Earnings Average', 'Earnings Low',
                                  'Earnings High', 'Revenue Average', 'Revenue Low', 'Revenue High']
                existing_cols = [
                    c for c in potential_cols if c in cal_df.columns]
                cal_df = cal_df[existing_cols]
                # Sort by date
                cal_df = cal_df.sort_values(
                    'Earnings Date').reset_index(drop=True)
                _self.logger.info(
                    f"Successfully processed earnings calendar for {symbol} ({len(cal_df)} entries).")
                return cal_df
            else:
                # If no valid data was extracted from either DataFrame or Dict
                _self.logger.info(
                    f"No valid earnings entries found after processing for {symbol}.")
                # st.info(f"No upcoming earnings data found for {symbol}.") # Display handled later
                return None

        except Exception as e:
            _self.logger.error(
                f"Error retrieving or processing earnings calendar for {symbol}: {e}", exc_info=True)
            st.error(f"Error retrieving earnings calendar for {symbol}: {e}")
            return None

    @st.cache_data(show_spinner="Fetching dividend history...")
    def get_dividend_history(_self, symbol: str) -> Optional[pd.Series]:
        _self.logger.info(f"Fetching dividend history for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends  # Returns a Series with Date index

            if dividends is None or dividends.empty:
                _self.logger.info(f"No dividend data found for {symbol}.")
                # st.info(f"No dividend history found for {symbol}.") # Display handled later
                return None

            # Filter out zero dividends if necessary (usually not needed, yfinance handles it)
            dividends = dividends[dividends > 0]

            if dividends.empty:
                _self.logger.info(
                    f"Dividend data found for {symbol}, but all values were zero.")
                return None

            _self.logger.info(
                f"Dividend history for {symbol} fetched ({len(dividends)} entries). Latest: {dividends.index[-1].strftime(DATE_FORMAT)} - ${dividends.iloc[-1]:.4f}")
            # Return the Series, sorted by date (default from yfinance)
            return dividends

        except Exception as e:
            _self.logger.error(
                f"Error retrieving dividend history for {symbol}: {e}", exc_info=True)
            st.error(f"Error retrieving dividend history for {symbol}: {e}")
            return None

    # --- Static Methods (Utility/Display) ---

    @staticmethod
    # @st.cache_data(ttl=3600, show_spinner="Fetching sector performance...") # Cache for 1 hour
    def display_sector_performance():
        """ Fetches and displays recent sector performance using ETFs. """
        logger.info("Fetching sector performance data.")
        # SPDR Sector ETFs (common benchmark)
        sectors = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Energy": "XLE",
        }
        performance = {}
        # Define period (e.g., 1 month)
        end_date = datetime.now()
        # Go back slightly further to ensure we capture ~30 days of trading
        start_date = end_date - timedelta(days=45)
        start_str = start_date.strftime(DATE_FORMAT)
        end_str = end_date.strftime(DATE_FORMAT)

        etf_symbols = list(sectors.values())

        try:
            with st.spinner("Downloading sector ETF data..."):
                # Download all ETFs at once
                data = yf.download(etf_symbols, start=start_str, end=end_str,
                                   progress=False, auto_adjust=True)  # Use auto_adjust

            if data.empty or 'Close' not in data:
                logger.warning("Failed to download valid sector ETF data.")
                st.warning(
                    "Could not retrieve sector performance data at this time.")
                return

            # Should be DataFrame with ETFs as columns
            close_prices = data['Close']

            # Calculate performance for each sector
            for sector, etf in sectors.items():
                if etf in close_prices.columns:
                    etf_series = close_prices[etf].dropna()
                    # Ensure enough data points for calculation (e.g., at least 2)
                    if len(etf_series) >= 2:
                        # Find first and last available price in the period
                        start_price = etf_series.iloc[0]
                        end_price = etf_series.iloc[-1]
                        # Calculate percentage change
                        pct_change = ((end_price - start_price) /
                                      start_price) * 100 if start_price != 0 else 0.0
                        performance[sector] = pct_change
                    else:
                        logger.warning(
                            f"Insufficient data points for {sector} ({etf}) in the period.")
                        # Mark as None if not enough data
                        performance[sector] = None
                else:
                    logger.warning(
                        f"Data for sector ETF {etf} not found in download.")
                    # Mark as None if ETF data missing
                    performance[sector] = None

        except Exception as e:
            logger.error(
                f"Error fetching or processing sector data: {e}", exc_info=True)
            st.warning(
                "An error occurred while fetching sector performance data.")
            return

        # Filter out sectors where calculation failed
        valid_performance = {k: v for k,
                             v in performance.items() if v is not None}

        if not valid_performance:
            st.warning(
                "No valid sector performance data could be calculated for the period.")
            return

        # Create DataFrame and plot
        df_perf = pd.DataFrame(list(valid_performance.items()), columns=[
                               "Sector", "1-Month % Change"])
        df_perf = df_perf.sort_values("1-Month % Change", ascending=False)

        fig = px.bar(df_perf,
                     x="Sector",
                     y="1-Month % Change",
                     title="Sector Performance (Approx. Last 30 Days % Change)",
                     color="1-Month % Change",
                     color_continuous_scale='RdYlGn',  # Red -> Yellow -> Green
                     labels={"1-Month % Change": "% Change"},
                     height=400)
        fig.update_layout(xaxis_tickangle=-45)  # Improve label readability
        st.plotly_chart(fig, use_container_width=True, key="sector_perf_chart")
        logger.info("Displayed sector performance chart.")

    @staticmethod
    def create_tooltip_metric(label: str, value: Any, tooltip: str) -> str:
        """ Creates an HTML span with a tooltip for Streamlit metrics. """
        # Basic sanitation for HTML attributes/content
        tooltip_safe = str(tooltip).replace('"', '&quot;').replace(
            '<', '&lt;').replace('>', '&gt;')
        # Ensure value is string and also sanitized
        value_str = str(value).replace('<', '&lt;').replace('>', '&gt;')
        label_safe = str(label).replace('<', '&lt;').replace('>', '&gt;')
        # Use markdown's title attribute for tooltips
        return f'<span title="{tooltip_safe}"><b>{label_safe}:</b> {value_str}</span>'

    # --- Chatbot Function ---

    def generate_chat_response(self, user_query: str, context_data: Dict[str, Any]) -> str:
        """ Generates a response from Gemini based on provided context and user query. """
        if not self.gemini:
            self.logger.error(
                "generate_chat_response called but Gemini client is unavailable.")
            return "Sorry, the AI Chat Assistant is currently unavailable. Please check API configuration."

        self.logger.info(
            f"Generating chat response for query: '{user_query[:50]}...'")

        # --- Build Context String ---
        symbol = context_data.get('symbol', 'N/A')
        system_prompt = (
            f"You are a helpful financial analyst assistant integrated into a stock analysis dashboard.\n"
            f"You have been provided with the latest available data and analysis for the stock: {symbol}.\n"
            f"Answer the user's query concisely and accurately, strictly using ONLY the provided context below.\n"
            f"Do NOT use any external data, real-time information, or prior knowledge.\n"
            f"If the context doesn't contain the answer, state that the information is not available in the provided data.\n"
            f"Format responses clearly. Use bullet points for lists where appropriate.\n"
            f"\n--- START CONTEXT DATA FOR {symbol} ---\n"
        )

        context_lines = [system_prompt]

        # Company Info
        if context_data.get('stock_info'):
            info = context_data['stock_info']
            context_lines.append(f"\n**Company Information:**")
            context_lines.append(
                f"- Name: {info.get('longName', 'N/A')} ({symbol})")
            context_lines.append(f"- Sector: {info.get('sector', 'N/A')}")
            context_lines.append(f"- Industry: {info.get('industry', 'N/A')}")
            context_lines.append(
                # Truncate summary
                f"- Summary: {info.get('longBusinessSummary', 'N/A')[:200]}...")

        # Latest Technical Data
        if context_data.get('tech_data') is not None and not context_data['tech_data'].empty:
            latest_tech = context_data['tech_data'].iloc[-1]
            latest_date = latest_tech.get('Date')
            latest_date_str = latest_date.strftime(DATE_FORMAT) if pd.notna(
                latest_date) else "Latest Available"
            context_lines.append(
                f"\n**Latest Technical Data ({latest_date_str}):**")
            context_lines.append(f"- Close Price: ${latest_tech.get('Close', 'N/A'):,.2f}" if pd.notna(
                latest_tech.get('Close')) else "- Close Price: N/A")
            context_lines.append(f"- Volume: {latest_tech.get('Volume', 'N/A'):,.0f}" if pd.notna(
                latest_tech.get('Volume')) else "- Volume: N/A")
            # Key Indicators
            indicators_context = []
            for key in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility_20', 'ATR_14']:
                value = latest_tech.get(key)
                if pd.notna(value):
                    formatted_value = f"{value:,.2f}"  # Default formatting
                    if key.startswith('SMA') or key.startswith('BB'):
                        formatted_value = f"${value:,.2f}"
                    if key == 'RSI':
                        formatted_value = f"{value:.2f}"
                    if key in ['MACD', 'Signal_Line']:
                        formatted_value = f"{value:.3f}"
                    if key == 'Volatility_20':
                        # Assuming annualized %
                        formatted_value = f"{value:.2f}%"
                    if key == 'ATR_14':
                        formatted_value = f"${value:.2f}"  # ATR is price units
                    indicators_context.append(f"- {key}: {formatted_value}")
            if indicators_context:
                context_lines.append("**Key Indicators:**")
                context_lines.extend(indicators_context)

        # Technical Patterns
        if context_data.get('patterns'):
            context_lines.append("\n**Detected Technical Patterns:**")
            context_lines.extend([f"- {p}" for p in context_data['patterns']])

        # Sentiment
        if context_data.get('avg_sentiment') is not None:
            avg_sent = context_data['avg_sentiment']
            sent_desc = "Positive" if avg_sent > 0.05 else "Negative" if avg_sent < -0.05 else "Neutral"
            context_lines.append(
                f"\n**Average News Sentiment (VADER):** {avg_sent:.3f} ({sent_desc})")

        # Prophet Forecast (Include prediction for the last day forecasted)
        if context_data.get('prophet_forecast_data') is not None and not context_data['prophet_forecast_data'].empty:
            forecast_df = context_data['prophet_forecast_data']
            last_forecast = forecast_df.iloc[-1]
            pred_date = last_forecast['ds'].strftime(DATE_FORMAT)
            pred_value = last_forecast['yhat']
            pred_lower = last_forecast['yhat_lower']
            pred_upper = last_forecast['yhat_upper']
            context_lines.append(f"\n**Prophet Forecast:**")
            context_lines.append(
                f"- Predicted price on {pred_date}: ${pred_value:.2f} (Range: ${pred_lower:.2f} - ${pred_upper:.2f})")

        # ML Prediction (Next day)
        if context_data.get('ml_results'):
            ml_pred = context_data['ml_results'].get('next_day_prediction')
            last_actual = context_data['ml_results'].get('last_actual_close')
            context_lines.append("\n**Random Forest Prediction (Next Day):**")
            pred_text = f"${ml_pred:.2f}" if ml_pred is not None else "N/A"
            context_lines.append(f"- Predicted Close: {pred_text}")
            if ml_pred is not None and last_actual is not None:
                context_lines.append(
                    f"- Last Actual Close: ${last_actual:.2f}")

        # Financial Ratios (from stock_info)
        if context_data.get('stock_info'):
            info = context_data['stock_info']
            financials = []
            ratio_map = {'trailingPE': 'P/E (Trailing)', 'forwardPE': 'P/E (Forward)',
                         'priceToBook': 'P/B Ratio', 'dividendYield': 'Dividend Yield',
                         'payoutRatio': 'Payout Ratio', 'beta': 'Beta',
                         'profitMargins': 'Profit Margin'}
            for key, label in ratio_map.items():
                value = info.get(key)
                if value is not None and pd.notna(value):
                    formatted_val = f"{value:.2f}"
                    if key in ['dividendYield', 'payoutRatio', 'profitMargins']:
                        formatted_val = f"{value:.2%}"
                    financials.append(f"- {label}: {formatted_val}")
            if financials:
                context_lines.append("\n**Key Financial Ratios:**")
                context_lines.extend(financials)

        context_lines.append("\n--- END CONTEXT DATA ---")
        full_context = "\n".join(context_lines)

        # Truncate context if too long (though MAX_CHAT_TOKENS is quite large)
        # A simple truncation might remove vital info. Better rely on model's context window if possible.
        # if len(full_context) > MAX_CONTEXT_LENGTH: # Define MAX_CONTEXT_LENGTH based on model
        #     full_context = full_context[:MAX_CONTEXT_LENGTH] + "\n... (Context Truncated)"

        messages = [
            {"role": "system", "content": full_context},
            {"role": "user", "content": user_query}
        ]

        try:
            # Ensure model name matches the one used for initialization/availability
            response = self.gemini.chat.completions.create(
                # model="gemini-2.0-flash-lite-001", # If using this specific model endpoint
                model="models/gemini-1.5-flash-latest",  # General purpose model via v1beta
                messages=messages,
                max_tokens=MAX_CHAT_TOKENS,  # Max tokens for the *response*
                temperature=0.5  # Balance creativity and factuality
                # Add other parameters like top_p, top_k if needed
            )
            generated_text = response.choices[0].message.content.strip()
            self.logger.info(
                f"Chat response received, length: {len(generated_text)}")

            # Basic check if the model failed to follow instructions
            if "information is not available in the provided data" not in generated_text and "context doesn't contain" not in generated_text:
                # Optional: Check if response seems related to context (complex)
                pass

            return generated_text

        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            # Check for specific API errors if possible (e.g., rate limits, auth issues)
            st.error(f"Error communicating with the AI assistant: {e}")
            return "Sorry, I encountered an error while processing your request. Please try again later or check the application logs."

# -----------------------------------------------------------------------------
# Streamlit UI Layout and Logic Functions
# -----------------------------------------------------------------------------


def configure_streamlit_page():
    """Configures the Streamlit page with a dark theme."""
    st.set_page_config(
        layout="wide",
        page_title="Advanced Stock Insights Dashboard",
        page_icon="🌃"  # Dark theme icon
    )
    # Apply custom CSS for Dark Theme
    st.markdown("""
    <style>
        /* --- Main App Dark Theme --- */
        html, body, [class*="st-"] {
          color: #E0E0E0; /* Light grey text for readability */
        }
        .stApp {
             /* background-color: #0E1117; */ /* Default Streamlit dark */
             background-color: #1C1C1E; /* Slightly different dark shade */
        }
        .main .block-container {
             padding-top: 2rem;
             padding-bottom: 2rem;
             padding-left: 1.5rem;
             padding-right: 1.5rem;
        }

        /* --- Headers --- */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF; /* White headers */
        }
        .gradient-header {
            background: linear-gradient(45deg, #3a418a, #2d67a1); /* Slightly lighter blue gradient for dark */
            color: white;
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); /* Darker shadow */
        }
         .gradient-header h1, .gradient-header h2, .gradient-header h3 {
              margin: 0;
              color: white !important;
         }

        /* --- Cards & Containers --- */
        .metric-card {
            background: #2E2E2E; /* Darker card background */
            padding: 1rem 1.2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
            border: 1px solid #444444; /* Darker border */
            height: 100%;
            color: #E0E0E0; /* Ensure text inside card is light */
        }
        .stExpander {
             border: 1px solid #444444;
             border-radius: 8px;
             background: #2E2E2E; /* Dark background for expander body */
             margin-bottom: 1rem;
             color: #E0E0E0;
        }
         .stExpander header {
              font-weight: bold;
              background-color: #3a3a3c; /* Slightly lighter dark for header */
              border-radius: 8px 8px 0 0;
              color: #FFFFFF !important; /* White header text */
         }
         .stExpander header:hover {
             background-color: #4a4a4c; /* Lighter on hover */
         }

        /* --- Metrics --- */
        .stMetric {
             border-radius: 6px;
        }
         .stMetric > label {
              font-weight: bold;
              color: #AAAAAA; /* Lighter grey label */
         }
         .stMetric > div[data-testid="stMetricValue"] {
              font-size: 1.6em;
              font-weight: 600;
              color: #FFFFFF; /* White value */
         }
         .stMetric > div[data-testid="stMetricDelta"] {
               font-size: 0.9em;
               /* Color handled by positive/negative classes potentially */
         }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 18px;
            border-bottom: 1px solid #444444; /* Darker border under tabs */
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            white-space: pre-wrap;
            background-color: transparent; /* Tabs transparent until selected */
            border-radius: 8px 8px 0 0;
            gap: 8px;
            color: #AAAAAA; /* Default tab text color */
            border-bottom: 2px solid transparent; /* Prepare for selected border */
            margin-bottom: -1px; /* Align bottom border */
        }
        .stTabs [aria-selected="true"] {
            background-color: #2E2E2E; /* Selected tab background matches card */
            font-weight: bold;
            color: #FFFFFF; /* White text for selected tab */
            border-bottom: 2px solid #3a418a; /* Accent color border */
        }

        /* --- Sentiment & Delta Colors --- */
        .positive { color: #4CAF50; } /* Brighter Green */
        .negative { color: #F44336; } /* Brighter Red */
        .neutral { color: #90A4AE; } /* Blue Grey */

        /* Specific for Metric Delta */
         .stMetric > div[data-testid="stMetricDelta"] .positive { color: #4CAF50 !important; }
         .stMetric > div[data-testid="stMetricDelta"] .negative { color: #F44336 !important; }


        /* --- Plotly Chart Styling (Handled by template, but ensure container bg is ok) --- */
        .plotly-chart {
             border-radius: 8px;
             background-color: #1E1E1E; /* Match app background */
        }

        /* --- Sidebar --- */
        .stSidebar {
             background-color: #242426; /* Slightly different dark for sidebar */
             border-right: 1px solid #444444;
        }
         .stSidebar .stButton button {
             width: 100%;
             background-color: #3a418a; /* Match header accent */
             color: white;
             border: none;
         }
         .stSidebar .stButton button:hover {
              background-color: #4a519a;
              color: white;
         }
          .stSidebar .stButton button:active {
               background-color: #2d317a;
               color: white;
          }
          .stSidebar .stTextInput input,
          .stSidebar .stSelectbox select,
          .stSidebar .stDateInput input,
          .stSidebar .stNumberInput input {
              background-color: #3a3a3c; /* Dark input fields */
              color: #E0E0E0;
              border: 1px solid #555555;
          }
          .stSidebar label {
              color: #E0E0E0; /* Light labels */
          }


        /* --- Dataframes --- */
         .stDataFrame {
              background-color: #2E2E2E;
              color: #E0E0E0;
         }
          .stDataFrame thead th {
              background-color: #3a3a3c;
              color: #FFFFFF;
          }
           .stDataFrame tbody tr:nth-child(even) {
                background-color: #3a3a3c; /* Banded rows */
           }


         /* --- Chat --- */
         .stChatInput {
              background-color: #2E2E2E;
         }
          .stChatMessage {
               background-color: #3a3a3c;
               border-radius: 8px;
               padding: 10px 15px;
               margin: 5px 0;
          }

    </style>
    """, unsafe_allow_html=True)


def display_sidebar(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the sidebar for user inputs."""
    st.sidebar.header("Analysis Parameters")

    # --- Stock Input ---
    company_input = st.sidebar.text_input(
        "Company Name or Stock Symbol",
        # Use 'company_input' from state
        defaults.get('company_input', DEFAULT_COMPANY_NAME),
        help="Enter the company name (e.g., Apple Inc.) or its stock symbol (e.g., AAPL)."
    )

    st.sidebar.markdown("---")

    # --- Time Range ---
    st.sidebar.subheader("Time Range")
    # Define periods relative to today
    date_options = {
        '1 Month': 30, '3 Months': 90, '6 Months': 180,
        '1 Year': 365, '2 Years': 730, '5 Years': 1825,
        'Max': 99999,  # Use a large number for 'Max'
        'Custom': -1
    }
    selected_range_key = st.sidebar.selectbox(
        "Select Period",
        list(date_options.keys()),
        index=list(date_options.keys()).index(defaults.get(
            'selected_range_key', '1 Year')),  # Default to 1 Year
        help="Select the historical data period for analysis."
    )

    end_date_default = defaults.get('end_date', datetime.today())
    start_date_default = defaults.get('start_date', datetime.today(
    ) - timedelta(days=date_options.get(selected_range_key, 365)))

    end_date = datetime.today()  # Generally, end date is today
    start_date = None

    if selected_range_key == 'Custom':
        start_date_input = st.sidebar.date_input(
            "Start Date", start_date_default, max_value=datetime.today() - timedelta(days=1))
        end_date_input = st.sidebar.date_input(
            "End Date", end_date_default, max_value=datetime.today())

        if start_date_input >= end_date_input:
            st.sidebar.error("Start date must be before end date.")
            # Keep defaults or adjust slightly
            start_date = end_date_input - timedelta(days=1)
            end_date = end_date_input
        else:
            start_date = start_date_input
            end_date = end_date_input
    elif selected_range_key == 'Max':
        # yfinance often goes back far, start early
        start_date = datetime(1970, 1, 1)
    else:
        days = date_options[selected_range_key]
        start_date = end_date - timedelta(days=days)

    st.sidebar.markdown("---")

    # --- Technical Indicators ---
    st.sidebar.subheader("Technical Indicators")
    tech_indicators = {
        'show_sma': st.sidebar.checkbox("Show SMA (20, 50, 200)", defaults.get('show_sma', True)),
        'show_rsi': st.sidebar.checkbox("Show RSI (14)", defaults.get('show_rsi', True)),
        'show_macd': st.sidebar.checkbox("Show MACD (12, 26, 9)", defaults.get('show_macd', True)),
        'show_bollinger': st.sidebar.checkbox("Show Bollinger Bands (20, 2)", defaults.get('show_bollinger', True))
    }

    st.sidebar.markdown("---")

    # --- Prediction ---
    st.sidebar.subheader("Forecast")
    # Renamed from "Prediction" to "Forecast" for clarity (Prophet)
    prediction_days = st.sidebar.slider(
        "Days to Forecast (Prophet)",
        min_value=1, max_value=30,  # Allow up to 1 year forecast
        value=defaults.get('prediction_days', 7),
        step=1,
        help="Number of future days to predict using the Prophet time series model."
    )

    st.sidebar.markdown("---")

    # --- Advanced Analyses ---
    st.sidebar.subheader("Advanced Analyses")

    # Portfolio Simulation
    run_portfolio = st.sidebar.checkbox(
        "Portfolio Simulation", value=defaults.get('run_portfolio', False))
    portfolio_input = ""
    initial_investment = 0.0
    strategy = ""
    if run_portfolio:
        portfolio_input = st.sidebar.text_input(
            "Portfolio Stocks (comma-separated symbols)",
            defaults.get('portfolio_input', DEFAULT_PORTFOLIO_STOCKS),
            help="Enter stock symbols (e.g., AAPL, GOOGL, MSFT)."
        )
        initial_investment = st.sidebar.number_input(
            "Initial Investment ($)",
            min_value=float(MIN_INVESTMENT),
            value=float(defaults.get(
                'initial_investment', DEFAULT_INVESTMENT)),
            step=1000.0, format="%.2f",
            help="The starting value of the simulated portfolio."
        )
        strategy_options = ["Equal Weight", "Market Cap Weighted"]
        strategy = st.sidebar.selectbox(
            "Investment Strategy",
            strategy_options,
            index=strategy_options.index(defaults.get(
                'strategy_display', "Equal Weight")),  # Use display name
            help="How to allocate the initial investment among stocks."
        )

    # Correlation Analysis
    run_correlation = st.sidebar.checkbox(
        "Correlation Analysis", value=defaults.get('run_correlation', False))
    correlation_input = ""
    if run_correlation:
        correlation_input = st.sidebar.text_input(
            "Correlation Stocks (comma-separated symbols)",
            defaults.get('correlation_input', DEFAULT_CORRELATION_STOCKS),
            help="Enter stock symbols (at least 2) to analyze return correlations."
        )

    # Random Forest Prediction
    run_ml = st.sidebar.checkbox(
        "Random Forest Prediction", value=defaults.get('run_ml', False))
    # ML prediction now runs on the main selected stock, no separate input needed
    # ml_input_stock = st.sidebar.text_input("Stock for ML Prediction", defaults.get('ml_input_stock', DEFAULT_ML_STOCK))

    # ESG Performance
    run_esg = st.sidebar.checkbox(
        "ESG Performance", value=defaults.get('run_esg', False))
    # ESG analysis now runs on the main selected stock

    st.sidebar.markdown("---")

    # --- Additional Data ---
    st.sidebar.subheader("Financial Data")
    show_dividends = st.sidebar.checkbox(
        "Dividend History", value=defaults.get('show_dividends', True))
    show_sector = st.sidebar.checkbox(
        "Sector Performance", value=defaults.get('show_sector', True))

    st.sidebar.markdown("---")

    # --- UI Options ---
    st.sidebar.subheader("UI Options")
    show_tooltips = st.sidebar.checkbox(
        "Enable Metric Tooltips", defaults.get('show_tooltips', True))

    st.sidebar.markdown("---")

    # --- Submit Button ---
    submitted = st.sidebar.button("Analyze Stock")

    logger.info("Sidebar inputs gathered.")

    # Return dictionary of parameters
    return {
        "company_input": company_input,
        "start_date": start_date,
        "end_date": end_date,
        "selected_range_key": selected_range_key,  # Store selected range key
        **tech_indicators,
        "prediction_days": prediction_days,
        "run_portfolio": run_portfolio,
        "portfolio_input": portfolio_input,
        "initial_investment": initial_investment,
        # Store internal strategy key, and display name for sidebar default
        "strategy": strategy.lower().replace(" ", "_") if strategy else "",
        "strategy_display": strategy,
        "run_correlation": run_correlation,
        "correlation_input": correlation_input,
        "run_ml": run_ml,
        # "ml_input_stock": ml_input_stock, # Removed, uses main stock
        "run_esg": run_esg,
        # "esg_input_stock": esg_input_stock, # Removed, uses main stock
        "show_dividends": show_dividends,
        "show_sector": show_sector,
        "show_tooltips": show_tooltips,
        "submitted": submitted
    }


def plot_stock_data(df: pd.DataFrame, symbol: str, indicators: Dict[str, bool]):
    """ Plots the main stock chart with OHLC, Volume, and selected indicators. """
    st.subheader("Price Chart & Technical Analysis")
    if df is None or df.empty:
        st.warning("No historical data available to plot.")
        return

    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        # Enable secondary axis
                        specs=[[{"secondary_y": True}]])

    # 1. Candlestick Trace (Primary Y-axis)
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name=f'{symbol} Price',
        increasing_line_color='#2e7d32', decreasing_line_color='#c62828'
    ), secondary_y=False)  # Assign to primary axis

    # 2. Volume Trace (Secondary Y-axis)
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'], name='Volume',
        # Light purple/blue volume bars
        marker_color='rgba(100, 100, 180, 0.3)',
    ), secondary_y=True)  # Assign to secondary axis

    # 3. Add Selected Technical Indicators (Primary Y-axis)
    sma_colors = {'SMA_20': '#ffa726', 'SMA_50': '#fb8c00',
                  'SMA_200': '#ef6c00'}  # Orange shades
    if indicators.get('show_sma'):
        for sma, color in sma_colors.items():
            if sma in df.columns and df[sma].notna().any():
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df[sma],
                    line=dict(color=color, width=1.5),
                    name=sma.replace('_', ' ')
                ), secondary_y=False)

    if indicators.get('show_bollinger'):
        if 'BB_Upper' in df.columns and df['BB_Upper'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['BB_Upper'],
                line=dict(color='#1565c0', width=1,
                          dash='dash'),  # Blue dashed
                name='BB Upper'
            ), secondary_y=False)
        if 'BB_Lower' in df.columns and df['BB_Lower'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['BB_Lower'],
                line=dict(color='#1565c0', width=1, dash='dash'),
                fill='tonexty',  # Fill area between upper and lower bands
                fillcolor='rgba(21, 101, 192, 0.1)',  # Light blue fill
                name='BB Lower'
            ), secondary_y=False)

    # --- Layout Configuration ---
    fig.update_layout(
        # title=f"{symbol} Price Analysis", # Title moved outside plot function
        template="plotly_dark",  # Dark theme
        height=550,  # Increase height slightly
        xaxis_rangeslider_visible=False,  # Hide rangeslider
        xaxis_title="Date",


        yaxis=dict(
            title="Price (USD)",
            # Lighter grid lines
            showgrid=True
        ),
        yaxis2=dict(
            title="Volume",
            showgrid=False,  # Hide grid for volume axis
            overlaying='y',  # Ensure it overlays the primary y-axis
            side='right',
            showticklabels=False  # Optionally hide volume tick labels if cluttered
        ),

        hovermode='x unified',  # Show tooltips for all traces at a given x-value
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom", y=1.01,  # Position above plot
            xanchor="left", x=0,
            font=dict(
                color='white'
            )
            # Semi-transparent legend background
        ),
        margin=dict(l=50, r=50, t=30, b=50)  # Adjust margins
    )

    # --- Plot Indicator Subplots (RSI, MACD) ---
    indicator_subplots = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,  # Space between RSI and MACD
        row_heights=[0.5, 0.5]  # Equal height for RSI and MACD
    )
    added_indicator_subplot = False

    # RSI Subplot
    if indicators.get('show_rsi') and 'RSI' in df.columns and df['RSI'].notna().any():
        indicator_subplots.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'], name='RSI',
            line=dict(color='#1565c0')  # Blue line for RSI
        ), row=1, col=1)
        # Add Overbought/Oversold lines
        indicator_subplots.add_hline(
            y=70, line_dash='dash', line_color='rgba(200, 0, 0, 0.5)', row=1, col=1)
        indicator_subplots.add_hline(
            y=30, line_dash='dash', line_color='rgba(0, 150, 0, 0.5)', row=1, col=1)
        indicator_subplots.update_yaxes(title_text="RSI", row=1, col=1, range=[
                                        0, 100])  # Fix RSI range
        added_indicator_subplot = True

    # MACD Subplot
    if indicators.get('show_macd') and 'MACD' in df.columns and df['MACD'].notna().any():
        # MACD Line
        indicator_subplots.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD'], name='MACD',
            line=dict(color='#ff7f0e')  # Orange for MACD
        ), row=2, col=1)
        # Signal Line
        indicator_subplots.add_trace(go.Scatter(
            x=df['Date'], y=df['Signal_Line'], name='Signal Line',
            line=dict(color='#1f77b4')  # Blue for Signal
        ), row=2, col=1)
        # Histogram
        histogram_colors = ['rgba(46, 125, 50, 0.6)' if v >=
                            # Green/Red bars
                            0 else 'rgba(198, 40, 40, 0.6)' for v in df['MACD_Histogram']]
        indicator_subplots.add_trace(go.Bar(
            x=df['Date'], y=df['MACD_Histogram'], name='Histogram',
            marker_color=histogram_colors
        ), row=2, col=1)
        indicator_subplots.update_yaxes(title_text="MACD", row=2, col=1)
        added_indicator_subplot = True

    # Update layout for the indicator subplots
    indicator_subplots.update_layout(
        template="plotly_dark",  # Dark theme
        height=350 if added_indicator_subplot else 0,  # Adjust height based on content
        xaxis2_title="Date",  # Title for the bottom x-axis
        hovermode='x unified',
        showlegend=False,  # Hide legend for subplots if main plot has it
        margin=dict(l=50, r=50, t=20, b=50)
    )
    # Hide x-axis labels on top plot
    indicator_subplots.update_xaxes(showticklabels=False, row=1, col=1)
    # Show x-axis labels on bottom plot
    indicator_subplots.update_xaxes(showticklabels=True, row=2, col=1)

    # Display the charts
    st.plotly_chart(fig, use_container_width=True, key="main_stock_chart")
    if added_indicator_subplot:
        st.plotly_chart(indicator_subplots,
                        use_container_width=True, key="indicator_subplots")

    # Display Technical Pattern Analysis
    st.subheader("Technical Pattern Analysis")
    # Get patterns from session state
    patterns = st.session_state.get('patterns', [])
    if patterns:
        # Use columns for better layout if many patterns
        num_patterns = len(patterns)
        cols = st.columns(min(num_patterns, 3))  # Max 3 columns
        for i, pattern in enumerate(patterns):
            with cols[i % 3]:
                st.info(f"- {pattern}")  # Use st.info for visual separation
    else:
        st.info(
            "No significant standard technical patterns detected in the recent data.")
    st.caption(
        "Pattern analysis is based on standard indicator readings and is illustrative, not investment advice.")


def display_company_info(info: Dict[str, Any], show_tooltips: bool, analyzer: EnhancedStockAnalyzer):
    st.subheader(
        f"Company Overview: {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")

    # --- Key Metrics Row ---
    cols_metrics = st.columns(4)
    with cols_metrics[0]:
        value = info.get('currentPrice') or info.get(
            'regularMarketPrice') or info.get('previousClose')
        st.metric("Last Price", f"${value:,.2f}" if pd.notna(
            value) else "N/A", help="Most recent available price (current or previous close).")
    with cols_metrics[1]:
        st.metric("Market Cap", f"${info.get('marketCap', 0):,}" if info.get(
            'marketCap') else "N/A", help="Total market value of outstanding shares.")
    with cols_metrics[2]:
        st.metric("Sector", info.get('sector', 'N/A'),
                  help="The sector the company belongs to.")
    with cols_metrics[3]:
        st.metric("Industry", info.get('industry', 'N/A'),
                  help="The specific industry the company operates in.")

    st.markdown("---")

    # --- Business Summary ---
    st.subheader("Business Summary")
    summary = info.get('longBusinessSummary', 'No business summary available.')
    st.info(summary)  # Use st.info for visual distinction

    st.markdown("---")

    # --- Financial Metrics Expander ---
    with st.expander("Key Financial Metrics & Ratios", expanded=False):
        metric_definitions = {
            # Valuation
            'trailingPE': ("Trailing P/E", "Price-to-Earnings ratio based on past 12 months' earnings.", ".2f"),
            'forwardPE': ("Forward P/E", "Price-to-Earnings ratio based on future earnings estimates.", ".2f"),
            'priceToBook': ("Price/Book (P/B)", "Compares market value to the company's book value.", ".2f"),
            'enterpriseValue': ("Enterprise Value", "Total company value (market cap + debt - cash).", ",.0f"),
            'enterpriseToRevenue': ("EV/Revenue", "Enterprise value compared to total revenue.", ".2f"),
            'enterpriseToEbitda': ("EV/EBITDA", "Enterprise value compared to Earnings Before Interest, Taxes, Depreciation, Amortization.", ".2f"),
            # Dividends & Payout
            'dividendYield': ("Dividend Yield", "Annual dividend payout as a percentage of share price.", ".2%"),
            'payoutRatio': ("Payout Ratio", "Proportion of earnings paid out as dividends.", ".2%"),
            # Note: .2f for percentage value * 100
            'fiveYearAvgDividendYield': ("5Y Avg Div Yield", "Average dividend yield over the last 5 years.", ".2f"),
            # Profitability & Margins
            'profitMargins': ("Profit Margin", "Net income as a percentage of revenue.", ".2%"),
            'grossMargins': ("Gross Margin", "Gross profit as a percentage of revenue.", ".2%"),
            'operatingMargins': ("Operating Margin", "Operating income as a percentage of revenue.", ".2%"),
            'ebitdaMargins': ("EBITDA Margin", "EBITDA as a percentage of revenue.", ".2%"),
            # Per Share
            'trailingEps': ("Trailing EPS", "Earnings per share over the past 12 months.", ".2f"),
            'forwardEps': ("Forward EPS", "Estimated earnings per share for the next fiscal year.", ".2f"),
            # Volatility & Other
            'beta': ("Beta", "Stock's volatility relative to the overall market (S&P 500).", ".2f"),
            'volume': ("Volume", "Number of shares traded in the latest session.", ",.0f"),
            'averageVolume': ("Avg Volume (10 Day)", "Average daily trading volume over 10 days.", ",.0f"),
            'fiftyTwoWeekHigh': ("52 Week High", "Highest price in the past 52 weeks.", ",.2f"),
            'fiftyTwoWeekLow': ("52 Week Low", "Lowest price in the past 52 weeks.", ",.2f"),
        }

        metrics_data = {}
        available_metrics_count = 0
        for key, (label, tooltip, fmt) in metric_definitions.items():
            value = info.get(key)
            # Check value is meaningful
            if value is not None and pd.notna(value) and value != 0:
                try:
                    if fmt == ".2%":
                        value_str = f"{value:.2%}"
                    elif fmt == ",.0f":
                        value_str = f"{value:,.0f}"
                        if key == 'enterpriseValue':
                            value_str = f"${value_str}"  # Add $ for EV
                    elif fmt == ",.2f":
                        # Assume currency if 2 decimals
                        value_str = f"${value:,.2f}"
                    elif fmt == ".2f":
                        # Non-currency 2 decimals (P/E, P/B, Beta, etc)
                        value_str = f"{value:.2f}"
                    else:  # Default
                        value_str = str(value)

                    metrics_data[label] = (value_str, tooltip)
                    available_metrics_count += 1
                except (ValueError, TypeError):
                    # Handle formatting errors
                    metrics_data[label] = ("Error", tooltip)
            # else:
                # Optionally include N/A for missing values
                # metrics_data[label] = ("N/A", tooltip)

        if not metrics_data:
            st.write("No detailed financial metrics available.")
        else:
            cols_per_row = 3  # Adjust number of columns
            num_metrics = len(metrics_data)
            labels_ordered = list(metrics_data.keys())  # Keep order consistent
            num_rows = (num_metrics + cols_per_row - 1) // cols_per_row

            for i in range(num_rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    metric_index = i * cols_per_row + j
                    if metric_index < num_metrics:
                        label = labels_ordered[metric_index]
                        value_str, tooltip = metrics_data[label]
                        with cols[j]:
                            # Use st.metric for better visual separation
                            if show_tooltips:
                                st.metric(
                                    label=label, value=value_str, help=tooltip)
                            else:
                                st.metric(label=label, value=value_str)


def display_sentiment_analysis(articles: List[Dict[str, Any]], avg_sentiment: float, sentiment_df: Optional[pd.DataFrame]):
    st.markdown("<div class='gradient-header'><h3>📰 News & Sentiment Analysis</h3></div>",
                unsafe_allow_html=True)

    if not articles and sentiment_df is None:
        st.warning("No news articles or sentiment data found for analysis.")
        return

    # --- Sentiment Score Card & Trend ---
    st.subheader("Overall Sentiment")
    with st.container():
        col1, col2 = st.columns([1, 3])  # Ratio for score vs chart

        with col1:
            sentiment_color_class = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
            sentiment_label = "Positive 😊" if avg_sentiment > 0.05 else "Negative 😞" if avg_sentiment < - \
                0.05 else "Neutral 😐"

            st.markdown(f"""
                 <div class='metric-card' style='text-align: center;'>
                     <p style='font-size: 0.9em; color: #555; margin-bottom: 0.2em;'>Avg. Sentiment Score</p>
                     <p class='{sentiment_color_class}' style='font-size: 2em; font-weight: bold; margin-bottom: 0.2em;'>{avg_sentiment:.2f}</p>
                     <p class='{sentiment_color_class}' style='font-size: 1em;'>({sentiment_label})</p>
                 </div>
             """, unsafe_allow_html=True)

        with col2:
            if sentiment_df is not None and not sentiment_df.empty:
                st.markdown("<div class='metric-card'>",
                            unsafe_allow_html=True)  # Card styling
                st.markdown(
                    "<p style='font-weight: bold; text-align: center; margin-bottom: 0.5em;'>Daily Sentiment Trend (VADER)</p>", unsafe_allow_html=True)
                fig = px.area(sentiment_df, x=sentiment_df.index, y='Daily_Sentiment',
                              labels={'Daily_Sentiment': 'Avg. Score', 'Date': 'Date'}, height=200)

                line_color = '#2e7d32' if avg_sentiment > 0.05 else '#c62828' if avg_sentiment < - \
                    0.05 else '#546e7a'
                fig.update_traces(line_color=line_color, fillcolor=line_color.replace(
                    ')', ', 0.1)').replace('rgb', 'rgba'))

                fig.add_hline(y=0, line_dash='dash', line_color='grey')

                # --- CORRECTED LAYOUT UPDATE ---
                fig.update_layout(
                    template="plotly_dark",  # Dark theme
                    margin=dict(l=10, r=10, t=5, b=10),  # Tight margins
                    showlegend=False,
                    # White plot background
                    xaxis_showgrid=False,
                    # Correct way to set y-axis grid color
                    yaxis=dict(showgrid=True)
                )
                # --- END CORRECTION ---

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='metric-card'><p style='text-align: center; padding: 3rem 0;'>No daily sentiment data to plot trend.</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- News Articles Display ---
    st.subheader("Recent News Headlines")
    if not articles:
        st.info("No recent news articles found in the selected period.")
        return

    for i, article in enumerate(articles[:10]):
        sentiment_score = article.get('sentiment', {}).get('compound', 0)
        border_color = "green" if sentiment_score > 0.05 else "red" if sentiment_score < -0.05 else "grey"
        sentiment_label = f"{sentiment_score:.2f}"

        source = article.get('source', {}).get('name', 'Unknown Source')
        published_at_str = article.get('publishedAt', '')
        published_date = "Unknown Date"
        if published_at_str:
            try:
                published_date = pd.to_datetime(
                    published_at_str).strftime('%b %d, %Y %H:%M')
            except:
                published_date = published_at_str

        st.markdown(f"""
            <div style='border-left: 5px solid {border_color}; padding: 10px 15px; margin-bottom: 12px; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                <p style='font-size: 0.8em; color: #666; margin-bottom: 5px;'>
                    {source} | {published_date} | Sentiment: <span style='font-weight: bold; color: {border_color};'>{sentiment_label}</span>
                </p>
                <h5 style='margin-bottom: 8px;'>
                    <a href="{article.get('url', '#')}" target="_blank" style='text-decoration: none; color: #0d47a1;'>
                        {article.get('title', 'No Title Available')}
                    </a>
                </h5>
                <p style='font-size: 0.9em; color: #333;'>{article.get('description', 'No description available.')[:250]}...</p>
            </div>
        """, unsafe_allow_html=True)

# --- FIX: Add display function for Prophet ---


def display_prophet_forecast(forecast_df: pd.DataFrame, historical_df: pd.DataFrame, symbol: str, days_predicted: int):
    """ Displays the Prophet forecast plot. """
    st.subheader(f"Prophet Forecast ({days_predicted} Days)")

    if forecast_df is None or forecast_df.empty:
        st.warning("No forecast data available to display.")
        return
    if historical_df is None or historical_df.empty:
        st.warning("Historical data needed for context is missing.")
        return  # Cannot plot forecast without history

    fig = go.Figure()

    # Plot historical actual data
    fig.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['Close'],
        mode='lines',
        name='Actual Price',
        # Solid dark line for actual
        line=dict(color='#1E90FF', width=2)
    ))

    # Plot forecast line (yhat)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='#FFA500', width=2)  # Blue for forecast
    ))

    # Plot confidence interval (yhat_lower, yhat_upper)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines', name='Upper Bound',
        line=dict(width=4),  # Don't draw line
        showlegend=True  # Hide from legend
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines', name='Lower Bound',
        line=dict(width=0),
        fill='tonexty',  # Fill area between upper and lower bounds
        fillcolor='rgba(0, 100, 200, 0.15)',  # Light blue fill
        showlegend=True
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0  # Optional: adjust legend font color
        ))

    st.plotly_chart(fig, use_container_width=True,
                    key="prophet_forecast_chart")

    # Display tail of forecast data
    st.write("Forecasted Values (Last 10 Days):")
    st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).rename(columns={
        'ds': 'Date', 'yhat': 'Predicted', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'
    }).set_index('Date').style.format("${:.2f}"))
# --- End Prophet Display Fix ---


def display_portfolio_analysis(result: Dict[str, Any], initial_investment: float):
    st.subheader("Portfolio Simulation (1 Year Performance)")

    if not result:
        st.warning("Portfolio simulation data is not available or failed.")
        return

    symbols_used = result.get('symbols_used', [])
    st.caption(f"Simulation based on: {', '.join(symbols_used)}")

    # --- Performance Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Value", f"${result['cumulative_value'].iloc[-1]:,.2f}",
                  f"{((result['cumulative_value'].iloc[-1] / initial_investment) - 1):.2%} vs Initial ${initial_investment:,.0f}",
                  help="Total value of the portfolio at the end of the 1-year simulation period.")
    with col2:
        st.metric("Annualized Return", f"{result['annualized_return']:.2%}",
                  help="The geometric average annual rate of return.")
    with col3:
        st.metric("Annualized Volatility", f"{result['annualized_volatility']:.2%}",
                  help="Standard deviation of returns (risk measure).")

    # --- Sharpe Ratio ---
    # Display Sharpe Ratio separately or with more context if needed
    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}",
              help="Risk-adjusted return (higher is generally better). Assumes 0% risk-free rate.")

    st.markdown("---")

    # --- Portfolio Value Chart ---
    st.write("**Portfolio Value Over Time**")
    value_df = result['cumulative_value']
    if not pd.api.types.is_datetime64_any_dtype(value_df.index):
        try:
            value_df.index = pd.to_datetime(value_df.index)
        except Exception as e:
            st.error(
                f"Error converting index to datetime for portfolio value chart: {e}")
            return

    fig_value = px.area(value_df, title=None,  # "Portfolio Value Growth",
                        labels={'value': 'Portfolio Value ($)', 'index': 'Date'}, height=400)
    fig_value.update_traces(line_color='#0d47a1',
                            fillcolor='rgba(13, 71, 161, 0.1)')  # Blue theme
    fig_value.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig_value, use_container_width=True,
                    key="portfolio_value_chart")

    # --- Portfolio Allocation Pie Chart ---
    st.write("**Portfolio Allocation (Weights)**")
    weights_dict = result.get('weights')
    if weights_dict:
        weights_df = pd.DataFrame(
            list(weights_dict.items()), columns=['Stock', 'Weight'])
        # Exclude zero-weight stocks if any
        weights_df = weights_df[weights_df['Weight'] > 0]

        if not weights_df.empty:
            fig_weights = px.pie(weights_df, values='Weight', names='Stock',
                                 # title='Portfolio Weights',
                                 hole=0.3,  # Donut chart
                                 height=350)
            fig_weights.update_traces(textposition='outside', textinfo='percent+label', pull=[
                                      # Explode slices slightly
                                      0.05]*len(weights_df))
            fig_weights.update_layout(
                template="plotly_dark",
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10)

            )
            st.plotly_chart(fig_weights, use_container_width=True,
                            key="portfolio_weights_pie")
        else:
            st.warning(
                "Could not display weights chart (no stocks with >0 weight).")
    else:
        st.warning("Portfolio weights data not available.")


def display_correlation_analysis(matrix: pd.DataFrame):
    st.subheader("Stock Correlation Analysis (1 Year Returns)")

    if matrix is None or matrix.empty:
        st.warning("No correlation data available or calculation failed.")
        return

    if len(matrix.columns) < 2:
        st.warning("Correlation matrix requires at least two stocks.")
        return

    fig = px.imshow(matrix,
                    labels=dict(color="Correlation Coefficient"),
                    x=matrix.columns,
                    y=matrix.index,
                    template="plotly_dark",
                    # title="Correlation Heatmap",
                    color_continuous_scale='RdBu',  # Red-White-Blue scale is good for correlation
                    zmin=-1, zmax=1,  # Ensure scale covers full range
                    text_auto='.2f',  # Display values on heatmap, formatted to 2 decimals
                    aspect="auto",  # Adjust aspect ratio
                    height=max(400, len(matrix.columns)*50)  # Dynamic height
                    )
    # Custom hover text
    fig.update_traces(
        hovertemplate='Correlation(%{x}, %{y}) = %{z:.2f}<extra></extra>')
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_tickangle=-45  # Angle labels if many stocks
    )
    st.plotly_chart(fig, use_container_width=True, key='correlation_heatmap')
    st.caption("Correlation measures how stock returns move relative to each other. 1 means perfect positive correlation, -1 perfect negative, 0 no linear correlation.")


def display_ml_prediction(results: Dict[str, Any], symbol: str):
    st.subheader(f"Random Forest Price Prediction for {symbol}")

    if not results:
        st.warning(
            "Machine Learning prediction results are not available or failed.")
        return

    pred_value = results.get('next_day_prediction')
    last_actual = results.get('last_actual_close')
    rmse = results.get('rmse')
    mae = results.get('mae')
    feature_importance = results.get('feature_importance')
    test_dates = results.get('test_dates')
    y_test = results.get('y_test')
    predictions = results.get('predictions')
    features_used = results.get('features_used', [])

    st.markdown("**Prediction & Model Performance**")
    col1, col2 = st.columns(2)
    with col1:
        delta_str = "N/A"
        delta_help = ""
        if pred_value is not None and last_actual is not None and last_actual != 0:
            delta = (pred_value / last_actual) - 1
            delta_str = f"{delta:.2%}"
            delta_help = f"Change vs last actual close of ${last_actual:.2f}"

        st.metric("Predicted Next Day Close",
                  f"${pred_value:.2f}" if pred_value is not None else "N/A",
                  delta_str if delta_str != "N/A" else None,
                  help=f"Model's forecast for the next trading day's close price. {delta_help} (Illustrative only, not investment advice)."
                  )
    with col2:
        st.metric("Model Test RMSE", f"${rmse:.2f}" if rmse is not None else "N/A",
                  help="Root Mean Squared Error on the test data set (lower is better). Represents typical prediction error in $.")
        # st.metric("Model Test MAE", f"${mae:.2f}" if mae is not None else "N/A",
        #           help="Mean Absolute Error on the test data set (lower is better). Average absolute prediction error in $.")

    st.markdown("---")

    # --- Prediction vs Actual Plot ---
    if test_dates is not None and y_test is not None and predictions is not None:
        st.write("**Model Performance on Test Data**")
        plot_df = pd.DataFrame(
            {'Actual': y_test, 'Predicted': predictions}, index=test_dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual Price', line=dict(color='#eeeeee', width=2)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted Price', line=dict(
            color='#ff7f0e', dash='dot', width=2)))  # Orange dotted line

        fig.update_layout(
            template="plotly_dark",
            # title=f"{symbol} Actual vs. Predicted Close (Test Set)",
            xaxis_title='Date', yaxis_title='Price (USD)',
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key="ml_test_plot")
    else:
        st.warning(
            "Could not plot model performance: Missing test data or predictions.")

    # --- Feature Importance ---
    if feature_importance is not None and not feature_importance.empty:
        st.write("**Feature Importance**")
        st.caption(
            f"Indicates which features were most influential for the Random Forest model's predictions. Features used: {', '.join(features_used)}")
        # Display as bar chart
        fig_feat = px.bar(feature_importance.head(10),  # Show top 10 features
                          x='Importance', y='Feature', orientation='h',
                          # title="Top Feature Importances",
                          labels={'Importance': 'Relative Importance',
                                  'Feature': 'Feature'},
                          height=350, text_auto='.3f')
        fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'},  # Show most important at top
                               margin=dict(l=10, r=10, t=30, b=20))
        fig_feat.update_traces(marker_color='#1f77b4')  # Blue bars
        st.plotly_chart(fig_feat, use_container_width=True,
                        key="ml_feature_importance")

        # Optionally display the full table in an expander
        # with st.expander("View Full Feature Importance Data"):
        #     st.dataframe(feature_importance.style.format({'Importance': "{:.3%}"}))

    else:
        st.warning("Feature importance data not available.")


def display_esg_scores(scores: Optional[Dict[str, float]], symbol: str):
    st.subheader(f"ESG Performance for {symbol}")

    if scores is None:
        st.warning(
            f"ESG scores could not be retrieved or are not available for {symbol}.")
        return
    if not scores:  # Empty dictionary
        st.info(f"No specific ESG scores found for {symbol}.")
        return

    # Check if using placeholders (simple check)
    if any("Placeholder" in lbl for lbl in scores.keys()):
        st.caption(
            "Displaying placeholder ESG scores as actual data was unavailable.")

    score_labels = list(scores.keys())
    score_values = list(scores.values())

    # Use a Gauge chart or Bar chart
    # Gauge chart example (requires Plotly >= 4.7)
    # Let's use Bar chart for simplicity and broader compatibility
    fig = go.Figure()

    # Add bars for each score
    fig.add_trace(go.Bar(
        x=score_labels,
        y=score_values,
        text=[f"{val:.1f}" for val in score_values],  # Display value on bar
        textposition='auto',
        marker_color=['#1f77b4', '#2ca02c', '#ff7f0e',
                      '#d62728'][:len(score_labels)]  # Example colors
    ))

    fig.update_layout(
        template="plotly_dark",
        # title=f"{symbol} ESG Scores",
        yaxis_title='Score (0-100 scale typical, lower can be better)',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_tickangle=-45  # Angle labels if they overlap
    )
    st.plotly_chart(fig, use_container_width=True, key="esg_chart")
    st.caption("ESG scores indicate Environmental, Social, and Governance performance. Interpretation varies by provider; lower scores sometimes indicate lower risk.")


def display_earnings_calendar(calendar_df: Optional[pd.DataFrame], symbol: str):
    st.subheader(f"Earnings Calendar for {symbol}")

    if calendar_df is None or calendar_df.empty:
        st.info(f"No upcoming earnings data found or available for {symbol}.")
        return

    display_df = calendar_df.copy()

    # Format date
    display_df['Earnings Date'] = pd.to_datetime(
        display_df['Earnings Date']).dt.strftime('%b %d, %Y')

    # Format numeric columns as currency or number, handling N/A
    currency_cols = ['Earnings Average', 'Earnings Low', 'Earnings High']
    revenue_cols = ['Revenue Average', 'Revenue Low', 'Revenue High']

    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else 'N/A')
    for col in revenue_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(
                x) else 'N/A')  # Revenue often large, no decimals

    # Select and order columns for display
    potential_cols = ['Earnings Date', 'Earnings Average', 'Revenue Average',
                      'Earnings Low', 'Earnings High', 'Revenue Low', 'Revenue High']
    display_cols = [c for c in potential_cols if c in display_df.columns]

    # Set Date as index for better table display
    if 'Earnings Date' in display_cols:
        display_df.set_index('Earnings Date', inplace=True)
        display_cols.remove('Earnings Date')

    st.dataframe(display_df[display_cols])
    st.caption(
        "Upcoming and recent earnings report dates and estimates. Data from yfinance.")


def display_dividend_history(dividends: Optional[pd.Series], symbol: str):
    st.subheader(f"Dividend History for {symbol}")

    if dividends is None or dividends.empty:
        st.info(f"No dividend history found or available for {symbol}.")
        return

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(  # Use Bar chart for discrete dividend payments
        x=dividends.index,
        y=dividends.values,
        name='Dividend Paid',
        marker_color='#2ca02c'  # Green for dividends
    ))

    fig.update_layout(
        template="plotly_dark",
        # title=f"{symbol} Dividend History",
        xaxis_title="Date",
        yaxis_title="Dividend Amount (USD per Share)",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key="dividend_history_chart")
    st.caption("Historical dividend payments per share. Data from yfinance.")


def prepare_llm_context(session_state: st.session_state) -> Dict[str, Any]:
    """ Prepares a dictionary of context data for the LLM assistant. """
    context = {'symbol': session_state.get('current_symbol', 'N/A')}

    # Only include data if it exists and is not None
    if session_state.get('stock_info'):
        context['stock_info'] = session_state['stock_info']
    if session_state.get('tech_data') is not None:
        context['tech_data'] = session_state['tech_data']
    if session_state.get('patterns'):
        context['patterns'] = session_state['patterns']
    if session_state.get('avg_sentiment') is not None:
        context['avg_sentiment'] = session_state['avg_sentiment']
    # Add Prophet forecast data to context
    if session_state.get('prophet_forecast_data') is not None:
        context['prophet_forecast_data'] = session_state['prophet_forecast_data']
    if session_state.get('ml_results') is not None:
        context['ml_results'] = session_state['ml_results']
    # Add ESG context if available
    if session_state.get('esg_scores') is not None:
        context['esg_scores'] = session_state['esg_scores']
    # Add Earnings context if available (maybe just next date?)

    logger.debug(f"Prepared LLM context with keys: {list(context.keys())}")
    return context


# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------

def main():
    configure_streamlit_page()
    analyzer = EnhancedStockAnalyzer(newsapi, gemini_client, logger)

    # --- Initialize Session State ---
    # Define keys and their default values
    default_session_state = {
        'last_inputs': {}, 'analysis_done': False, 'current_symbol': None,
        'current_company_input': None, 'stock_data': None, 'stock_info': None,
        'tech_data': None, 'patterns': [], 'analyzed_articles': [],
        'avg_sentiment': 0.0, 'sentiment_df': None, 'prophet_forecast_data': None,
        'portfolio_result': None, 'correlation_matrix': None, 'ml_results': None,
        'esg_scores': None, 'dividend_history': None,
        'messages': []  # For chatbot
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Sidebar Inputs ---
    input_params = display_sidebar(st.session_state['last_inputs'])

    # --- Main Area Header ---
    st.markdown("<div class='gradient-header'><h1>Advanced Stock Insights Dashboard</h1></div>",
                unsafe_allow_html=True)

    # --- Display Sector Performance (Optional) ---
    if input_params['show_sector']:
        with st.expander("Sector Performance Snapshot (Last 30 Days)", expanded=False):
            EnhancedStockAnalyzer.display_sector_performance()
        # st.markdown("---") # Optional separator

    # --- Process Inputs and Run Analysis ---
    if input_params['submitted']:
        # Clear previous analysis results on new submission
        for key in default_session_state.keys():
            if key not in ['last_inputs', 'messages']:  # Keep inputs and chat history
                st.session_state[key] = default_session_state[key]

        st.session_state['analysis_done'] = False  # Reset analysis flag
        st.session_state.messages = []  # Clear chat history on new analysis

        if not input_params['company_input']:
            st.warning("Please enter a company name or stock symbol.")
            st.stop()

        # Store current inputs
        st.session_state['last_inputs'] = input_params.copy()

        with st.spinner("Resolving symbol and fetching data..."):
            symbol = analyzer.get_stock_symbol(input_params['company_input'])

            if not symbol:
                st.error(
                    f"Could not find or validate a stock symbol for '{input_params['company_input']}'. Please check the name/symbol or try again.")
                logger.error(
                    f"Symbol resolution failed for '{input_params['company_input']}'.")
                st.stop()
            else:
                logger.info(
                    f"Analysis requested for symbol: {symbol} (Input: '{input_params['company_input']}')")
                st.session_state['current_symbol'] = symbol
                st.session_state['current_company_input'] = input_params['company_input']

                # --- Load Core Data ---
                st.session_state['stock_data'] = analyzer.load_stock_data(
                    symbol, input_params['start_date'], input_params['end_date'])
                st.session_state['stock_info'] = analyzer.load_stock_info(
                    symbol)

        # Check if core data loading failed
        if st.session_state['stock_data'] is None or st.session_state['stock_data'].empty:
            st.error(
                f"Failed to load historical stock data for {symbol}. Cannot proceed with analysis.")
            logger.error(
                f"Stopping analysis for {symbol} due to missing historical data.")
            st.session_state['analysis_done'] = False  # Mark as not done
            st.stop()
        # Info might be partial, proceed but log warning if missing essential parts later
        if st.session_state['stock_info'] is None:
            logger.warning(
                f"Company info fetch failed for {symbol}, some details might be missing.")
            # st.warning("Could not fetch complete company information.")

        # --- Perform Calculations (if core data is available) ---
        with st.spinner("Calculating indicators and performing analyses..."):
            # Technical Analysis
            st.session_state['tech_data'] = analyzer.calculate_technical_indicators(
                st.session_state['stock_data'])
            st.session_state['patterns'] = analyzer.analyze_patterns(
                st.session_state['tech_data'])

            # News & Sentiment (Use a reasonable date range, e.g., last 90 days or selected range, whichever is shorter)
            news_days_back = min(
                90, (input_params['end_date'] - input_params['start_date']).days + 1)
            news_start_date = input_params['end_date'] - \
                timedelta(days=news_days_back)
            news_articles = analyzer.load_news(
                symbol, news_start_date, input_params['end_date'])
            analyzed_articles, avg_sentiment, sentiment_df = analyzer.analyze_sentiment(
                news_articles)
            st.session_state['analyzed_articles'] = analyzed_articles
            st.session_state['avg_sentiment'] = avg_sentiment
            st.session_state['sentiment_df'] = sentiment_df

            # --- FIX: Call Prophet Forecast ---
            st.session_state['prophet_forecast_data'] = analyzer.prophet_forecast(
                st.session_state['stock_data'], input_params['prediction_days'])

            # Optional Analyses based on sidebar choices
            if input_params['run_portfolio'] and input_params['portfolio_input']:
                portfolio_symbols = [s.strip().upper(
                ) for s in input_params['portfolio_input'].split(',') if s.strip()]
                if portfolio_symbols:
                    st.session_state['portfolio_result'] = analyzer.portfolio_simulation(
                        portfolio_symbols, input_params['initial_investment'], input_params['strategy'])

            if input_params['run_correlation'] and input_params['correlation_input']:
                correlation_symbols = [s.strip().upper(
                ) for s in input_params['correlation_input'].split(',') if s.strip()]
                if len(correlation_symbols) >= 2:
                    st.session_state['correlation_matrix'] = analyzer.advanced_correlation_analysis(
                        correlation_symbols)
                elif correlation_symbols:
                    st.warning(
                        "Correlation analysis requires at least two valid stock symbols.")

            if input_params['run_ml']:
                # Run ML on the main symbol
                st.session_state['ml_results'] = analyzer.machine_learning_prediction(
                    symbol)

            if input_params['run_esg']:
                # Run ESG on the main symbol
                st.session_state['esg_scores'] = analyzer.esg_scoring(symbol)

            if input_params['show_dividends']:
                st.session_state['dividend_history'] = analyzer.get_dividend_history(
                    symbol)

            # Mark analysis as complete
            st.session_state['analysis_done'] = True
            logger.info(
                f"All analysis calculations complete for symbol: {symbol}")
            st.rerun()  # Rerun to update the display with new session state data

    # --- Display Results ---
    if st.session_state.get('analysis_done', False):
        symbol = st.session_state.get('current_symbol', 'N/A')
        company_input_display = st.session_state.get(
            'current_company_input', symbol)  # Show original input
        st.header(f"Analysis Results for: {company_input_display} ({symbol})")
        logger.info(f"Displaying analysis results for {symbol}")

        # Retrieve results from session state
        stock_info = st.session_state.get('stock_info')
        tech_data = st.session_state.get('tech_data')
        # patterns = st.session_state.get('patterns', []) # Patterns displayed within plot_stock_data
        prophet_forecast_data = st.session_state.get('prophet_forecast_data')
        analyzed_articles = st.session_state.get('analyzed_articles', [])
        avg_sentiment = st.session_state.get('avg_sentiment', 0.0)
        sentiment_df = st.session_state.get('sentiment_df')
        portfolio_result = st.session_state.get('portfolio_result')
        correlation_matrix = st.session_state.get('correlation_matrix')
        ml_results = st.session_state.get('ml_results')
        esg_scores = st.session_state.get('esg_scores')
        dividend_history = st.session_state.get('dividend_history')

        # Get indicator toggles from last inputs
        indicator_toggles = {k: st.session_state['last_inputs'].get(k, False)
                             for k in ['show_sma', 'show_rsi', 'show_macd', 'show_bollinger']}
        show_tooltips = st.session_state['last_inputs'].get(
            'show_tooltips', True)
        days_predicted = st.session_state['last_inputs'].get(
            'prediction_days', 30)
        initial_investment_disp = st.session_state['last_inputs'].get(
            'initial_investment', DEFAULT_INVESTMENT)

        # --- Define Tabs ---
        tab_titles = ["📊 Overview & Chart", "📰 News & Sentiment"]
        optional_tabs_map = {
            "📈 Forecast": prophet_forecast_data is not None,
            "💼 Portfolio Sim": portfolio_result is not None,
            "🔗 Correlation": correlation_matrix is not None,
            "🤖 ML Prediction": ml_results is not None,
            "🌱 ESG": esg_scores is not None,  # Display even if empty, function handles it
            "💰 Financials": (dividend_history is not None)
        }
        enabled_optional_tabs = [title for title,
                                 enabled in optional_tabs_map.items() if enabled]
        tabs = st.tabs(tab_titles + enabled_optional_tabs)

        # --- Tab 1: Overview & Chart ---
        with tabs[0]:
            if stock_info:
                display_company_info(stock_info, show_tooltips, analyzer)
            else:
                st.warning("Company information could not be fully retrieved.")

            st.markdown("---")
            if tech_data is not None and not tech_data.empty:
                # This function now also displays patterns
                plot_stock_data(tech_data, symbol, indicator_toggles)
            else:
                st.error(
                    "Historical data or technical indicators missing; charts cannot be displayed.")

        # --- Tab 2: News & Sentiment ---
        with tabs[1]:
            # Pass data from session state to display function
            display_sentiment_analysis(
                analyzed_articles, avg_sentiment, sentiment_df)

        # --- Optional Tabs ---
        tab_index_offset = 2  # Start index for optional tabs
        current_tab_index = tab_index_offset

        # Forecast Tab
        if optional_tabs_map["📈 Forecast"]:
            with tabs[current_tab_index]:
                # --- FIX: Call Prophet display function ---
                display_prophet_forecast(
                    prophet_forecast_data, st.session_state['stock_data'], symbol, days_predicted)
            current_tab_index += 1

        # Portfolio Tab
        if optional_tabs_map["💼 Portfolio Sim"]:
            with tabs[current_tab_index]:
                display_portfolio_analysis(
                    portfolio_result, initial_investment_disp)
            current_tab_index += 1

        # Correlation Tab
        if optional_tabs_map["🔗 Correlation"]:
            with tabs[current_tab_index]:
                display_correlation_analysis(correlation_matrix)
            current_tab_index += 1

        # ML Prediction Tab
        if optional_tabs_map["🤖 ML Prediction"]:
            with tabs[current_tab_index]:
                display_ml_prediction(ml_results, symbol)
            current_tab_index += 1

        # ESG Tab
        if optional_tabs_map["🌱 ESG"]:
            with tabs[current_tab_index]:
                display_esg_scores(esg_scores, symbol)
            current_tab_index += 1

        # Financials Tab (Earnings & Dividends)
        # Condition now only depends on dividends
        if optional_tabs_map["💰 Financials"]:
            with tabs[current_tab_index]:
                # We only display dividends now if the tab exists
                if dividend_history is not None:  # Still good practice to check
                    display_dividend_history(dividend_history, symbol)
            current_tab_index += 1

        # --- AI Chat Assistant ---
        st.markdown("---")
        st.subheader("💬 AI Financial Assistant")
        if not analyzer.gemini:
            st.warning(
                "AI Assistant is unavailable (Gemini client not configured or failed to initialize).")
        else:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Get user input
            if prompt := st.chat_input(f"Ask a question about {symbol} based on the analyzed data..."):
                # Add user message to state and display
                st.session_state.messages.append(
                    {"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Thinking..."):
                        context_for_llm = prepare_llm_context(st.session_state)
                        response = analyzer.generate_chat_response(
                            prompt, context_for_llm)
                        st.markdown(response)
                # Add assistant message to state
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})

    else:
        # Initial state before analysis is run
        st.info(
            "👋 Welcome! Please configure parameters in the sidebar and click 'Analyze Stock' to begin.")
        st.markdown(
            "This dashboard provides technical analysis, news sentiment, forecasting, and more.")


if __name__ == "__main__":
    try:
        main()
        logger.info("Application session finished normally.")
    except Exception as main_err:
        logger.critical(
            f"Unhandled critical exception in main: {main_err}", exc_info=True)
        st.error(f"A critical error occurred: {main_err}")
        # st.exception(main_err) # Optionally display full traceback in Streamlit
