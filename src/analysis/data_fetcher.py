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
from fredapi import Fred

# Use relative imports for files within the same package
from .. import config
from .. import utils

logger = utils.logger


class DataFetcher:
    """Handles fetching data from external APIs (yfinance, NewsAPI, Gemini)."""

    def __init__(self, newsapi_client: Optional[NewsApiClient], gemini_client: Optional[OpenAI]):
        self.newsapi = newsapi_client
        self.gemini = gemini_client

    # get_stock_symbol method remains the same
    @st.cache_data(show_spinner="Fetching stock symbol...")
    def get_stock_symbol(_self, company_name_or_symbol: str) -> Optional[str]:
        """Resolves a company name or validates a symbol using Gemini or yfinance."""
        if not company_name_or_symbol or not isinstance(company_name_or_symbol, str):
            logger.warning("get_stock_symbol called with invalid input.")
            return None
        cleaned_input = company_name_or_symbol.strip().upper()
        is_likely_symbol = all(c.isalnum() or c == '.' for c in cleaned_input) and len(
            cleaned_input) <= 10
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
                    if test_ticker.info:
                        logger.warning(
                            f"yfinance found info for '{cleaned_input}' but no market price. Treating as potentially invalid.")
                    else:
                        logger.warning(
                            f"yfinance did not return valid info for likely symbol '{cleaned_input}'.")
            except Exception as e:
                logger.warning(
                    f"yfinance check for '{cleaned_input}' failed: {e}. Will try Gemini.")
        if not _self.gemini:
            logger.error(
                "Gemini client not available for symbol lookup by name.")
            if not is_likely_symbol:
                st.error(
                    "AI Assistant not configured. Cannot look up symbol by name.")
            return None
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
            if not symbol or 'NOT_FOUND' in symbol or len(symbol) > 10 or not all(c.isalnum() or c == '.' for c in symbol):
                logger.warning(
                    f"Gemini returned invalid or 'NOT_FOUND' symbol: '{symbol}'.")
                return None
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

    # load_stock_data method remains the same
    @st.cache_data(show_spinner="Fetching historical stock data...")
    def load_stock_data(_self, symbol: str, start_date: datetime, end_date: datetime, retries: int = 3, delay: int = 2) -> Optional[pd.DataFrame]:
        """Fetches and normalizes historical stock data from yfinance with retries."""
        logger.info(
            f"Fetching data for '{symbol}' from {start_date.strftime(config.DATE_FORMAT)} to {end_date.strftime(config.DATE_FORMAT)}")
        start_str = start_date.strftime(config.DATE_FORMAT)
        end_str = (end_date + timedelta(days=1)).strftime(config.DATE_FORMAT)
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{retries} for '{symbol}'")
                raw_data = yf.download(
                    symbol, start=start_str, end=end_str, progress=False, auto_adjust=False)
                if raw_data.empty:
                    logger.warning(
                        f"No data returned by yfinance for '{symbol}' on attempt {attempt}.")
                    if attempt == retries:
                        break
                    time.sleep(delay)
                    continue
                normalized_df = utils.normalize_yfinance_data(raw_data, symbol)
                if normalized_df is not None:
                    logger.info(
                        f"Successfully loaded and normalized data for '{symbol}'.")
                    return normalized_df
                else:
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
            if attempt < retries:
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
        logger.error(
            f"Failed to load valid data for '{symbol}' after {retries} attempts.")
        st.error(
            f"Failed to load data for {symbol} after multiple retries. Check symbol or try again later.")
        return None

    # load_stock_info method remains the same
    @st.cache_data(show_spinner="Fetching company information...")
    def load_stock_info(_self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches company information from yfinance."""
        try:
            logger.info(f"Fetching company info for '{symbol}'.")
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info or not isinstance(info, dict) or info.get('regularMarketPrice') is None:
                logger.warning(
                    f"No valid/complete company info returned by yfinance for '{symbol}'. Info: {info}")
                return info if isinstance(info, dict) else None
            logger.info(
                f"Fetched company info for '{symbol}'. Keys: {list(info.keys())[:5]}...")
            return info
        except Exception as e:
            logger.error(
                f"Error fetching info for '{symbol}': {e}", exc_info=True)
            st.error(f"Error fetching company info for {symbol}: {e}")
            return None

    # load_news method remains the same
    @st.cache_data(show_spinner="Fetching news articles...")
    def load_news(_self, query: str, from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
        """Fetches news articles using NewsAPI."""
        if not _self.newsapi:
            logger.warning("NewsAPI client not available. Cannot fetch news.")
            st.error("NewsAPI client is not configured.")
            return []
        earliest_allowed = datetime.now() - timedelta(days=config.NEWS_DAYS_LIMIT)
        adjusted_from_date = min(max(from_date, earliest_allowed), to_date)
        adjusted_to_date = min(to_date, datetime.now())
        if adjusted_from_date >= adjusted_to_date:
            logger.warning(
                f"Invalid date range for news after adjustments: from={adjusted_from_date}, to={adjusted_to_date}. Using last 2 days.")
            adjusted_to_date = datetime.now()
            adjusted_from_date = adjusted_to_date - timedelta(days=2)
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

    # esg_scoring method remains the same
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
                return None
            esg_data.index = esg_data.index.str.lower()
            scores = {}
            score_column = 'esgScores' if 'esgScores' in esg_data.columns else None
            if score_column is None and not esg_data.empty:
                if 'Value' in esg_data.columns:
                    score_column = 'Value'
                elif 'Score' in esg_data.columns:
                    score_column = 'Score'
                else:
                    first_col_name = esg_data.columns[0]
                    if pd.api.types.is_numeric_dtype(esg_data[first_col_name]):
                        score_column = first_col_name
                        logger.warning(
                            f"Using first column '{score_column}' for ESG scores as standard names not found.")
                    else:
                        logger.error(
                            f"Could not identify a valid score column in ESG data for {symbol}. Columns: {esg_data.columns}")
                        return None
            if score_column:
                score_map = {
                    'totalesg': 'Total ESG Score', 'environmentscore': 'Environmental Score',
                    'socialscore': 'Social Score', 'governancescore': 'Governance Score',
                    'highestcontroversy': 'Highest Controversy',
                    'esgperformance': 'ESG Performance'
                }
                for key, label in score_map.items():
                    if key in esg_data.index:
                        value = esg_data.loc[key, score_column]
                        if pd.notna(value):
                            try:
                                scores[label] = float(value)
                            except (ValueError, TypeError):
                                scores[label] = value
            else:
                logger.error(
                    f"No score column identified for ESG data {symbol}")
                return None
            if scores:
                logger.info(
                    f"Extracted ESG scores/data for {symbol}: {scores}")
                return scores
            else:
                logger.warning(
                    f"Sustainability data found for {symbol}, but no standard ESG scores extracted.")
                return None
        except Exception as e:
            logger.error(
                f"Error fetching/processing ESG data for {symbol}: {e}", exc_info=True)
            st.error(f"Error retrieving ESG data for {symbol}: {e}")
            return None

    # get_earnings_calendar method remains the same
    @st.cache_data(show_spinner="Fetching earnings calendar...")
    def get_earnings_calendar(_self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches earnings calendar data from yfinance."""
        logger.info(f"Fetching earnings calendar for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            cal_data = ticker.calendar
            cal_df = None
            if isinstance(cal_data, dict):
                if not cal_data or 'Earnings Date' not in cal_data:
                    logger.info(
                        f"Empty or invalid dict returned for earnings calendar {symbol}.")
                    return None
                else:
                    try:
                        cal_df = pd.DataFrame(cal_data)
                        logger.warning(
                            f"Received earnings calendar as Dict for {symbol}, attempting conversion.")
                        if 'Earnings Date' not in cal_df.columns:
                            cal_df_T = pd.DataFrame(cal_data).T
                            if 'Earnings Date' in cal_df_T.columns:
                                cal_df = cal_df_T
                            else:
                                logger.error(
                                    f"Could not find 'Earnings Date' after Dict conversion/transposition for {symbol}.")
                                return None
                    except Exception as dict_err:
                        logger.error(
                            f"Failed to convert earnings dict to DataFrame for {symbol}: {dict_err}")
                        return None
            elif cal_data is None or cal_data.empty:
                logger.info(f"No earnings calendar data found for {symbol}.")
                return None
            else:  # Assume it's already a DataFrame
                cal_df = cal_data

            cal_df.columns = [col.replace(' ', '') for col in cal_df.columns]
            if 'EarningsDate' not in cal_df.columns:
                logger.warning(
                    f"'EarningsDate' column missing in calendar for {symbol}. Columns: {cal_df.columns}")
                return None
            cal_df['EarningsDate'] = pd.to_datetime(cal_df['EarningsDate'], errors='coerce', unit='s' if isinstance(
                cal_df['EarningsDate'].iloc[0], (int, float)) else None)
            cal_df.dropna(subset=['EarningsDate'], inplace=True)
            num_cols = ['EarningsAverage', 'EarningsLow', 'EarningsHigh',
                        'RevenueAverage', 'RevenueLow', 'RevenueHigh']
            for col in num_cols:
                if col in cal_df.columns:
                    cal_df[col] = pd.to_numeric(cal_df[col], errors='coerce')
            cal_df.rename(columns={'EarningsDate': 'Earnings Date', 'EarningsAverage': 'Earnings Average', 'EarningsLow': 'Earnings Low',
                                   'EarningsHigh': 'Earnings High', 'RevenueAverage': 'Revenue Average', 'RevenueLow': 'Revenue Low',
                                   'RevenueHigh': 'Revenue High'}, inplace=True)
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

    # get_dividend_history method remains the same
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
            dividends = dividends[dividends > 0]
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

    # get_financials method remains the same
    @st.cache_data(show_spinner="Fetching Income Statement...")
    def get_financials(_self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches income statement (financials) data from yfinance."""
        logger.info(f"Fetching income statement (financials) for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            financials = ticker.financials
            if financials is None or financials.empty:
                logger.info(f"No income statement data found for {symbol}.")
                return None
            logger.info(f"Income statement data fetched for {symbol}.")
            return financials.transpose()  # Transpose so years are rows
        except Exception as e:
            logger.error(
                f"Error fetching income statement for {symbol}: {e}", exc_info=True)
            st.warning(
                f"Could not retrieve income statement data for {symbol}.")
            return None

    # get_balance_sheet method remains the same
    @st.cache_data(show_spinner="Fetching Balance Sheet...")
    def get_balance_sheet(_self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches balance sheet data from yfinance."""
        logger.info(f"Fetching balance sheet for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            balance_sheet = ticker.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                logger.info(f"No balance sheet data found for {symbol}.")
                return None
            logger.info(f"Balance sheet data fetched for {symbol}.")
            return balance_sheet.transpose()  # Transpose so years are rows
        except Exception as e:
            logger.error(
                f"Error fetching balance sheet for {symbol}: {e}", exc_info=True)
            st.warning(f"Could not retrieve balance sheet data for {symbol}.")
            return None

    # get_cash_flow method remains the same
    @st.cache_data(show_spinner="Fetching Cash Flow Statement...")
    def get_cash_flow(_self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches cash flow statement data from yfinance."""
        logger.info(f"Fetching cash flow statement for {symbol}.")
        try:
            ticker = yf.Ticker(symbol)
            cash_flow = ticker.cashflow
            if cash_flow is None or cash_flow.empty:
                logger.info(f"No cash flow statement data found for {symbol}.")
                return None
            logger.info(f"Cash flow statement data fetched for {symbol}.")
            return cash_flow.transpose()  # Transpose so years are rows
        except Exception as e:
            logger.error(
                f"Error fetching cash flow statement for {symbol}: {e}", exc_info=True)
            st.warning(
                f"Could not retrieve cash flow statement data for {symbol}.")
            return None

    # --- NEW METHOD FOR ANALYST RECOMMENDATIONS ---
    @st.cache_data(show_spinner="Fetching Analyst Recommendations...")
    def get_analyst_recommendations(_self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches analyst recommendations and target prices from yfinance."""
        logger.info(f"Fetching analyst recommendations for {symbol}.")
        recommendations_data = {}
        try:
            ticker = yf.Ticker(symbol)

            # 1. Get Recommendation History
            recs = ticker.recommendations
            recommendations_data['history'] = None  # Default to None

            if recs is not None and not recs.empty:
                # Convert index to Date (sometimes it's datetime)
                try:  # Add try-except for date conversion
                    recs.index = pd.to_datetime(recs.index).date
                except Exception as date_err:
                    logger.warning(
                        f"Could not parse recommendations index as date for {symbol}: {date_err}")
                    # Proceed without date index if conversion fails

                # --- Robust Column Selection ---
                # Define desired columns
                desired_cols = ['Firm', 'To Grade', 'Action']
                # Check which exist
                available_cols = [
                    col for col in desired_cols if col in recs.columns]

                if available_cols:
                    # Select only available columns
                    recs_filtered = recs[available_cols].sort_index(
                        ascending=False).head(15)
                    recommendations_data['history'] = recs_filtered
                    logger.info(
                        f"Fetched {len(recs_filtered)} recommendation history entries for {symbol} with columns: {available_cols}.")
                else:
                    logger.warning(
                        f"Could not find expected columns {desired_cols} in recommendation history for {symbol}. Available: {list(recs.columns)}")
                # --- End Robust Column Selection ---
            else:
                logger.info(
                    f"No recommendation history found for {symbol}.")

            # 2. Get Target Prices and Summary from .info (if available)
            info = ticker.info
            recommendations_data['targets'] = {}  # Default to empty dict
            recommendations_data['current_recommendation'] = None
            recommendations_data['num_analysts'] = None

            if info:
                target_prices = {
                    'High': info.get('targetHighPrice'),
                    'Low': info.get('targetLowPrice'),
                    'Mean': info.get('targetMeanPrice'),
                    'Median': info.get('targetMedianPrice')
                }
                # Filter out None values
                recommendations_data['targets'] = {
                    k: v for k, v in target_prices.items() if v is not None}

                recommendations_data['current_recommendation'] = info.get(
                    'recommendationKey')
                recommendations_data['num_analysts'] = info.get(
                    'numberOfAnalystOpinions')
                logger.info(
                    f"Fetched target prices and summary recommendation for {symbol}.")
            else:
                logger.warning(
                    f"Could not fetch .info for target prices/summary for {symbol}.")

            # Return collected data only if something meaningful was found
            if recommendations_data.get('history') is not None or recommendations_data.get('targets'):
                return recommendations_data
            else:
                logger.info(
                    f"No analyst recommendation data found for {symbol} after checking history and info.")
                return None  # Return None if nothing useful was found

        except Exception as e:
            logger.error(
                f"Error fetching analyst recommendations for {symbol}: {e}", exc_info=True)
            st.warning(
                f"Could not retrieve analyst recommendation data for {symbol}.")
            return None

    @st.cache_data(show_spinner="Fetching economic indicators...")
    def get_economic_indicators(_self, series_ids: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetches economic indicator series from FRED."""
        if not config.FRED_API_KEY:
            logger.error("FRED API Key not found in config.")
            # Optionally show a warning in Streamlit only once
            # st.warning("FRED API Key not configured. Cannot fetch economic indicators.", icon="⚠️")
            return None

        logger.info(f"Fetching FRED data for series: {series_ids}")
        try:
            fred = Fred(api_key=config.FRED_API_KEY)
            all_series_data = {}
            for series_id in series_ids:
                try:
                    # Fetch data for the requested range
                    series = fred.get_series(
                        series_id, observation_start=start_date, observation_end=end_date)
                    if not series.empty:
                        # Convert index to timezone-naive for consistency
                        series.index = series.index.tz_localize(None)
                        all_series_data[series_id] = series
                    else:
                        logger.warning(
                            f"No data returned for FRED series '{series_id}' in the specified range.")
                except Exception as series_err:
                    logger.error(
                        f"Failed to fetch FRED series '{series_id}': {series_err}")
                    # Show specific error
                    st.warning(
                        f"Could not fetch economic indicator: {series_id}")

            if not all_series_data:
                logger.warning(
                    "No data fetched for any requested FRED series.")
                return None

            # Combine into a single DataFrame, forward fill to align dates
            combined_df = pd.DataFrame(all_series_data).ffill()
            # Keep only data within the requested range strictly
            combined_df = combined_df[(combined_df.index >= start_date) & (
                combined_df.index <= end_date)]

            logger.info(
                f"Successfully fetched and combined FRED data for {list(combined_df.columns)}.")
            return combined_df

        except Exception as e:
            logger.error(
                f"Error initializing Fred client or fetching data: {e}", exc_info=True)
            st.error("An error occurred while fetching economic indicators.")
            return None
