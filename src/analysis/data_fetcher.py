# src/analysis/data_fetcher.py

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Any, List
from json.decoder import JSONDecodeError
from requests.exceptions import RequestException
from newsapi import NewsApiClient
from openai import OpenAI

from src import config, utils  # For logger and normalize_yfinance_data

logger = utils.logger


class DataFetcher:
    """Handles fetching data from external APIs (yfinance, NewsAPI, Gemini)."""

    def __init__(self, newsapi_client: Optional[NewsApiClient], gemini_client: Optional[OpenAI]):
        self.newsapi = newsapi_client
        self.gemini = gemini_client

    @st.cache_data(show_spinner="Fetching stock symbol...")
    def get_stock_symbol(_self, company_name_or_symbol: str) -> Optional[str]:
        """Resolves a company name or validates a symbol using Gemini or yfinance."""
        # Input basic validation
        if not company_name_or_symbol or not isinstance(company_name_or_symbol, str):
            _self.logger.warning("get_stock_symbol called with invalid input.")
            return None

        cleaned_input = company_name_or_symbol.strip().upper()

        # 1. Direct Check: If it looks like a symbol, try yfinance first
        is_likely_symbol = cleaned_input.isalnum(
        ) and cleaned_input.isupper() and len(cleaned_input) <= 6
        if is_likely_symbol:
            try:
                logger.info(
                    f"Input '{cleaned_input}' looks like symbol. Verifying with yfinance...")
                test_ticker = yf.Ticker(cleaned_input)
                if test_ticker.info and test_ticker.info.get('regularMarketPrice') is not None:
                    logger.info(
                        f"yfinance verified '{cleaned_input}' as a valid symbol.")
                    return cleaned_input
                else:
                    logger.warning(
                        f"yfinance did not return valid info for likely symbol '{cleaned_input}'.")
            except Exception as e:
                logger.warning(
                    f"yfinance check for '{cleaned_input}' failed: {e}. Will try Gemini.")

        # 2. Gemini Lookup (if available and needed)
        if not _self.gemini:
            logger.error(
                "Gemini client not available for symbol lookup by name.")
            # Only show UI error if it wasn't likely a symbol
            if not is_likely_symbol:
                st.error(
                    "AI Assistant not configured. Cannot look up symbol by name.")
            return None  # Cannot proceed

        logger.info(
            f"Attempting to resolve symbol for '{company_name_or_symbol}' using Gemini.")
        prompt = (f"What is the primary stock ticker symbol for the company named '{company_name_or_symbol}'? "
                  f"Return ONLY the stock ticker symbol itself (e.g., AAPL, GOOGL). "
                  f"If ambiguous or not found, return 'NOT_FOUND'.")
        try:
            response = _self.gemini.chat.completions.create(
                model=config.GEMINI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Provide only the stock ticker symbol."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20, temperature=0.0
            )
            symbol = response.choices[0].message.content.strip().upper()
            logger.info(
                f"Gemini returned: '{symbol}' for '{company_name_or_symbol}'")

            # Allow '.' in symbols
            if not symbol or 'NOT_FOUND' in symbol or len(symbol) > 10 or not symbol.replace('.', '').isalnum():
                logger.warning(
                    f"Gemini returned invalid or 'NOT_FOUND' symbol: '{symbol}'.")
                return None

            # Final Verification with yfinance
            try:
                logger.info(
                    f"Verifying Gemini's symbol '{symbol}' with yfinance...")
                test_ticker = yf.Ticker(symbol)
                if test_ticker.info and test_ticker.info.get('regularMarketPrice') is not None:
                    logger.info(
                        f"yfinance verified Gemini's symbol '{symbol}'.")
                    return symbol
                else:
                    logger.warning(
                        f"yfinance did not return valid info for Gemini's symbol '{symbol}'.")
                    return None
            except Exception as e:
                logger.warning(
                    f"yfinance verification failed for Gemini's symbol '{symbol}': {e}")
                return None

        except Exception as e:
            logger.error(
                f"Error during Gemini symbol lookup: {e}", exc_info=True)
            st.error(f"Error looking up stock symbol via AI: {e}")
            return None

        logger.error(
            f"Symbol resolution failed for '{company_name_or_symbol}' after all checks.")
        return None

    @st.cache_data(show_spinner="Fetching historical stock data...")
    def load_stock_data(_self, symbol: str, start_date: datetime, end_date: datetime, retries: int = 3, delay: int = 2) -> Optional[pd.DataFrame]:
        """Fetches and normalizes historical stock data from yfinance with retries."""
        logger.info(
            f"Fetching data for '{symbol}' from {start_date.strftime(config.DATE_FORMAT)} to {end_date.strftime(config.DATE_FORMAT)}")
        start_str = start_date.strftime(config.DATE_FORMAT)
        # yfinance end is exclusive
        end_str = (end_date + timedelta(days=1)).strftime(config.DATE_FORMAT)

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{retries} for '{symbol}'")
                # auto_adjust=False initially, handle Adj Close in normalize
                raw_data = yf.download(
                    symbol, start=start_str, end=end_str, progress=False, auto_adjust=False)

                if raw_data.empty:
                    logger.warning(
                        f"No data returned by yfinance for '{symbol}' on attempt {attempt}.")
                    if attempt == retries:
                        break  # Go to final error after last attempt
                    time.sleep(delay)
                    continue

                # Normalize the data using the utility function
                normalized_df = utils.normalize_yfinance_data(raw_data, symbol)

                if normalized_df is not None:
                    logger.info(
                        f"Successfully loaded and normalized data for '{symbol}'.")
                    return normalized_df
                else:
                    # Normalization failed, log already happened in normalize func. Retry download.
                    logger.warning(
                        f"Data normalization failed for {symbol} on attempt {attempt}.")
                    if attempt == retries:
                        break
                    time.sleep(delay)
                    continue

            except (JSONDecodeError, RequestException) as net_err:
                logger.error(
                    f"Network error for '{symbol}' on attempt {attempt}: {net_err}")
            except Exception as e:
                logger.exception(
                    f"General error loading data for '{symbol}' on attempt {attempt}: {e}")
                # Avoid showing raw error in UI repeatedly, log it.

            if attempt < retries:
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)

        logger.error(
            f"Failed to load valid data for '{symbol}' after {retries} attempts.")
        # Show UI error once
        st.error(
            f"Failed to load data for {symbol} after multiple retries. Check symbol or try again later.")
        return None

    @st.cache_data(show_spinner="Fetching company information...")
    def load_stock_info(_self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches company information from yfinance."""
        try:
            logger.info(f"Fetching company info for '{symbol}'.")
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info or not isinstance(info, dict) or info.get('regularMarketPrice') is None:
                logger.warning(
                    f"No valid company info returned by yfinance for '{symbol}'.")
                # Return partial info if available
                return info if isinstance(info, dict) else None
            logger.info(
                f"Fetched company info for '{symbol}'. Keys: {list(info.keys())[:5]}...")
            return info
        except Exception as e:
            logger.error(
                f"Error fetching info for '{symbol}': {e}", exc_info=True)
            st.error(f"Error fetching company info for {symbol}: {e}")
            return None

    @st.cache_data(show_spinner="Fetching news articles...")
    def load_news(_self, query: str, from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
        """Fetches news articles using NewsAPI."""
        if not _self.newsapi:
            logger.warning("NewsAPI client not available. Cannot fetch news.")
            st.error("NewsAPI client is not configured.")
            return []

        # Adjust dates based on API limits and validity
        earliest_allowed = datetime.now() - timedelta(days=config.NEWS_DAYS_LIMIT)
        adjusted_from_date = max(from_date, earliest_allowed)
        adjusted_to_date = min(to_date, datetime.now())
        if adjusted_from_date >= adjusted_to_date:
            logger.warning(
                f"Invalid date range for news: from={adjusted_from_date}, to={adjusted_to_date}. Using last 2 days.")
            adjusted_from_date = datetime.now() - timedelta(days=2)
            adjusted_to_date = datetime.now()

        from_str = adjusted_from_date.strftime(config.DATE_FORMAT)
        to_str = adjusted_to_date.strftime(config.DATE_FORMAT)
        logger.info(f"Fetching news for '{query}' from {from_str} to {to_str}")

        try:
            all_articles = _self.newsapi.get_everything(
                q=query, from_param=from_str, to=to_str, language='en',
                sort_by='relevancy', page_size=config.NEWS_API_PAGE_SIZE
            )
            articles = all_articles.get('articles', [])
            logger.info(
                f"Fetched {len(articles)} news articles for '{query}'. Total API results: {all_articles.get('totalResults')}")
            return articles
        except Exception as e:
            logger.error(
                f"Error fetching news for '{query}': {e}", exc_info=True)
            st.error(f"Error fetching news from NewsAPI: {e}")
            return []

    @st.cache_data(show_spinner="Fetching ESG scores...")
    def esg_scoring(_self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches ESG scores from yfinance sustainability data."""
        logger.info(f"Fetching ESG scores for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            esg_data = ticker.sustainability

            if esg_data is None or esg_data.empty or not isinstance(esg_data, pd.DataFrame):
                logger.info(
                    f"No ESG/sustainability data returned by yfinance for {symbol}.")
                return None  # Explicitly return None if no data

            esg_data.index = esg_data.index.str.lower()  # Consistent index matching

            scores = {}
            # Define potential keys and target column (often 'Value' or 'Score')
            # The exact column name can vary, 'Value' seems common
            score_column = 'Value' if 'Value' in esg_data.columns else (
                'Score' if 'Score' in esg_data.columns else None)
            if score_column is None and not esg_data.empty:
                # If specific columns aren't found, try the first column if it looks numeric
                first_col_name = esg_data.columns[0]
                if pd.api.types.is_numeric_dtype(esg_data[first_col_name]):
                    score_column = first_col_name
                    logger.warning(
                        f"Using first column '{score_column}' for ESG scores as standard names not found.")
                else:
                    logger.error(
                        f"Could not identify a valid score column in ESG data for {symbol}. Columns: {esg_data.columns}")
                    return None  # Cannot extract scores

            if score_column:
                score_map = {
                    'totalesg': 'Total ESG Score', 'environmentscore': 'Environmental Score',
                    'socialscore': 'Social Score', 'governancescore': 'Governance Score',
                    'highestcontroversy': 'Highest Controversy',  # Level, not score
                    'esgperformance': 'ESG Performance'  # Sometimes used
                }
                for key, label in score_map.items():
                    if key in esg_data.index:
                        value = esg_data.loc[key, score_column]
                        if pd.api.types.is_number(value) and pd.notna(value):
                            scores[label] = float(value)
                        # else: logger.debug(f"ESG key '{key}' found but value '{value}' is not numeric.")
            else:
                # Handle case where score_column was None even after checks
                logger.error(
                    f"No score column identified for ESG data {symbol}")
                return None

            if scores:
                logger.info(f"Extracted ESG scores for {symbol}: {scores}")
                return scores
            else:
                logger.warning(
                    f"Sustainability data found for {symbol}, but no standard ESG scores extracted.")
                # Return the raw DataFrame maybe? Or None? Let's return None for consistency.
                return None

        except Exception as e:
            logger.error(
                f"Error fetching/processing ESG data for {symbol}: {e}", exc_info=True)
            st.error(f"Error retrieving ESG data for {symbol}: {e}")
            return None

    @st.cache_data(show_spinner="Fetching earnings calendar...")
    def get_earnings_calendar(_self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches earnings calendar data from yfinance."""
        logger.info(f"Fetching earnings calendar for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            # .calendar often returns DataFrame directly now
            cal_df = ticker.calendar
            # Check type first before accessing attributes like .empty
            if isinstance(cal_df, dict):
                if not cal_df:  # Check if the dictionary itself is empty
                    logger.info(
                        f"Empty dictionary returned for earnings calendar {symbol}.")
                    return None
                else:
                    # If yfinance sometimes returns usable data in a dict, code to parse it would go here.
                    # For now, we'll log a warning and return None to prevent the error.
                    logger.warning(
                        f"Dict returned by yfinance calendar for {symbol}. Handling not implemented, returning None.")
                    # Or you could try converting the dict to a DataFrame if the structure is predictable:
                    # try:
                    #     cal_df = pd.DataFrame.from_dict(cal_df, orient='index') # Example conversion
                    #     # Add further processing if conversion works
                    # except Exception as dict_err:
                    #      logger.error(f"Failed to convert earnings dict to DataFrame: {dict_err}")
                    #      return None
                    return None  # Safest option for now is return None

            elif cal_df is None or cal_df.empty:  # Now it's safe to check .empty for a DataFrame
                logger.info(f"No earnings calendar data found for {symbol}.")
                return None
# If we reach here, cal_df should be a non-empty DataFrame

            cal_df.columns = [col.replace(' ', '')
                              for col in cal_df.columns]  # Remove spaces

            if 'EarningsDate' not in cal_df.columns:
                logger.warning(
                    f"'EarningsDate' column missing in calendar for {symbol}. Columns: {cal_df.columns}")
                return None  # Essential column missing

            # Convert date and numeric columns safely
            cal_df['EarningsDate'] = pd.to_datetime(
                cal_df['EarningsDate'], errors='coerce')
            cal_df.dropna(subset=['EarningsDate'], inplace=True)

            num_cols = ['EarningsAverage', 'EarningsLow', 'EarningsHigh',
                        'RevenueAverage', 'RevenueLow', 'RevenueHigh']
            for col in num_cols:
                if col in cal_df.columns:
                    cal_df[col] = pd.to_numeric(cal_df[col], errors='coerce')

            # Rename for consistency before returning
            cal_df.rename(columns={'EarningsDate': 'Earnings Date', 'EarningsAverage': 'Earnings Average', 'EarningsLow': 'Earnings Low',
                                   'EarningsHigh': 'Earnings High', 'RevenueAverage': 'Revenue Average', 'RevenueLow': 'Revenue Low',
                                   'RevenueHigh': 'Revenue High'}, inplace=True)

            # Select relevant columns and sort
            potential_cols = ['Earnings Date', 'Earnings Average', 'Earnings Low',
                              'Earnings High', 'Revenue Average', 'Revenue Low', 'Revenue High']
            existing_cols = [c for c in potential_cols if c in cal_df.columns]
            cal_df_final = cal_df[existing_cols].sort_values(
                'Earnings Date').reset_index(drop=True)

            logger.info(
                f"Successfully processed earnings calendar for {symbol} ({len(cal_df_final)} entries).")
            return cal_df_final

        except Exception as e:
            logger.error(
                f"Error retrieving/processing earnings calendar for {symbol}: {e}", exc_info=True)
            st.error(f"Error retrieving earnings calendar for {symbol}: {e}")
            return None

    @st.cache_data(show_spinner="Fetching dividend history...")
    def get_dividend_history(_self, symbol: str) -> Optional[pd.Series]:
        """Fetches dividend history from yfinance."""
        logger.info(f"Fetching dividend history for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            if dividends is None or dividends.empty:
                logger.info(f"No dividend data found for {symbol}.")
                return None

            dividends = dividends[dividends > 0]  # Filter out zero dividends
            if dividends.empty:
                logger.info(
                    f"Dividend data found for {symbol}, but all values were zero or less.")
                return None

            logger.info(
                f"Dividend history for {symbol} fetched ({len(dividends)} entries).")
            return dividends

        except Exception as e:
            logger.error(
                f"Error retrieving dividend history for {symbol}: {e}", exc_info=True)
            st.error(f"Error retrieving dividend history for {symbol}: {e}")
            return None
