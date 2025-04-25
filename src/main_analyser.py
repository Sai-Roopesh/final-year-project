# src/main_analyzer.py

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from newsapi import NewsApiClient
from openai import OpenAI
import streamlit as st

# Use relative imports for files within the same package
from . import config
from .analysis.data_fetcher import DataFetcher
from .analysis.technical import TechnicalAnalysis
from .analysis.sentiment import SentimentAnalysis
from .analysis.forecasting import Forecasting
from .analysis.ml_predictor import MLPredictor
from .analysis.portfolio import Portfolio
from . import utils  # Import utils to call prepare_llm_context

logger = utils.logger


class MainAnalyzer:
    """
    Coordinates the different analysis components (Data Fetching, TA, Sentiment, etc.).
    Holds instances of the component classes.
    """

    def __init__(self, newsapi_client: Optional[NewsApiClient], gemini_client: Optional[OpenAI]):
        """Initializes all analysis components."""
        logger.info("Initializing MainAnalyzer and sub-components...")
        self.data_fetcher = DataFetcher(newsapi_client, gemini_client)
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalysis()
        self.forecaster = Forecasting()
        self.ml_predictor = MLPredictor()
        self.portfolio_analyzer = Portfolio(self.data_fetcher)
        self.gemini_client = gemini_client

        if not self.sentiment_analyzer.vader_available:
            st.warning(
                "VADER sentiment analysis unavailable (lexicon missing). FinBERT will be used if selected.", icon="⚠️")

    def run_full_analysis(self, symbol: str, start_date: datetime, end_date: datetime,
                          run_advanced: Dict[str, bool],
                          advanced_params: Dict[str, Any]
                          ) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline for a given stock symbol.
        """
        logger.info(f"--- Starting Full Analysis for {symbol} ---")
        # --- ADDED: Default currency and store original input ---
        results = {'symbol': symbol, 'currency': 'USD'}  # Default to USD
        original_user_input = advanced_params.get('company_input', symbol)
        # --- END ADDED ---

        # 1. Core Data Fetching
        logger.info("Fetching core data...")
        # Ensure dates are timezone-naive datetime objects
        if not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())
        start_date = start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)

        stock_data = self.data_fetcher.load_stock_data(
            symbol, start_date, end_date)
        stock_info = self.data_fetcher.load_stock_info(symbol)
        results['stock_data'] = stock_data
        results['stock_info'] = stock_info

        # --- ADDED: Extract and store currency ---
        if stock_info and isinstance(stock_info, dict) and 'currency' in stock_info:
            results['currency'] = stock_info['currency']
            logger.info(f"Detected currency: {results['currency']}")
        else:
            logger.warning(
                f"Currency information not found for {symbol}. Defaulting to {results['currency']}.")
        # --- END ADDED ---

        if stock_data is None or stock_data.empty:
            logger.error(f"Core data fetching failed for {symbol}.")
            st.error(f"Failed fetch historical stock data for {symbol}.")
            return results  # Return results dict even on failure

        # 2. Technical Analysis
        logger.info("Running technical analysis...")
        results['tech_data'] = self.technical_analyzer.calculate_technical_indicators(
            stock_data)
        results['patterns'] = self.technical_analyzer.analyze_patterns(
            results['tech_data'])

        # 3. News & Sentiment
        logger.info("Fetching news and analyzing sentiment...")
        news_days_back = min(config.NEWS_DAYS_LIMIT,
                             (end_date - start_date).days + 1)
        news_start_date = end_date - timedelta(days=news_days_back)
        news_end_date = end_date

        # --- Determine NewsAPI Query ---
        newsapi_query = symbol  # Default to symbol
        if stock_info and stock_info.get('longName'):
            newsapi_query = stock_info['longName']  # Prefer company name
            logger.info(
                f"Using company name '{newsapi_query}' for NewsAPI query.")
        elif original_user_input != symbol:  # If original input was different
            newsapi_query = original_user_input
            logger.info(
                f"Using original user input '{newsapi_query}' for NewsAPI query.")
        else:  # If symbol was input and no longName, remove suffixes
            newsapi_query = symbol.replace('.NS', '').replace('.BO', '')
            logger.info(
                f"Using base symbol '{newsapi_query}' for NewsAPI query.")
        # --- End NewsAPI Query Determination ---

        # 3a. NewsAPI News & Sentiment
        newsapi_articles = self.data_fetcher.load_news(
            newsapi_query, news_start_date, news_end_date
        )
        sentiment_method_selected = advanced_params.get(
            'sentiment_method', 'vader').lower()
        logger.info(f"Selected sentiment method: {sentiment_method_selected}")
        newsapi_analyzed, newsapi_avg_sent, newsapi_sent_df = self.sentiment_analyzer.analyze_sentiment(
            newsapi_articles, method=sentiment_method_selected
        )
        results['newsapi_analyzed_articles'] = newsapi_analyzed
        results['newsapi_avg_sentiment'] = newsapi_avg_sent
        results['newsapi_sentiment_df'] = newsapi_sent_df
        # Keep overall from NewsAPI
        results['avg_sentiment'] = newsapi_avg_sent
        results['sentiment_method_used'] = sentiment_method_selected

        # 3b. yfinance News & Sentiment
        logger.info("Fetching yfinance news and analyzing sentiment...")
        yfinance_articles = self.data_fetcher.load_yfinance_news(symbol)
        yfinance_analyzed, yfinance_avg_sent, yfinance_sent_df = self.sentiment_analyzer.analyze_sentiment(
            yfinance_articles, method=sentiment_method_selected
        )
        results['yfinance_analyzed_articles'] = yfinance_analyzed
        results['yfinance_avg_sentiment'] = yfinance_avg_sent
        results['yfinance_sentiment_df'] = yfinance_sent_df

        # 4. Forecasting (Prophet)
        logger.info("Generating Prophet forecast...")
        forecast_days = advanced_params.get('prediction_days', 30)
        results['prophet_forecast_data'] = self.forecaster.prophet_forecast(
            stock_data,
            forecast_days,
            results.get('newsapi_sentiment_df'),
            results.get('yfinance_sentiment_df')
        )

        # 5. Financial Data
        if run_advanced.get('show_dividends'):
            logger.info("Fetching dividend history...")
            results['dividend_history'] = self.data_fetcher.get_dividend_history(
                symbol)
        else:
            results['dividend_history'] = None
        logger.info("Fetching earnings calendar...")
        results['earnings_calendar'] = self.data_fetcher.get_earnings_calendar(
            symbol)
        logger.info("Fetching fundamental data...")
        results['financials'] = self.data_fetcher.get_financials(symbol)
        results['balance_sheet'] = self.data_fetcher.get_balance_sheet(symbol)
        results['cash_flow'] = self.data_fetcher.get_cash_flow(symbol)
        logger.info("Fetching analyst recommendations...")
        results['analyst_info'] = self.data_fetcher.get_analyst_recommendations(
            symbol)

        # 6. Optional Economic Indicators
        results['economic_data'] = None
        run_economic_flag = run_advanced.get('run_economic', False)
        logger.info(
            f"Checking economic indicators: run_economic flag = {run_economic_flag}")
        if run_economic_flag:
            logger.info("Fetching economic indicators (flag was True)...")
            series_str = advanced_params.get('economic_series_input', '')
            series_list = [s.strip().upper()
                           for s in series_str.split(',') if s.strip()]
            logger.info(f"Economic series IDs requested: {series_list}")
            if series_list:
                fetched_econ_data = self.data_fetcher.get_economic_indicators(
                    series_list, start_date, end_date)
                results['economic_data'] = fetched_econ_data
                logger.info(
                    f"Economic data fetch result type: {type(fetched_econ_data)}")
                if fetched_econ_data is not None:
                    logger.info(
                        f"Economic data fetched rows: {len(fetched_econ_data)}")
            else:
                logger.warning(
                    "Economic indicators run flag is True, but no series IDs provided.")
        else:
            logger.info("Skipping economic indicators fetch (flag was False).")

        # 7. Optional Advanced Analyses
        results['portfolio_result'] = None
        if run_advanced.get('run_portfolio'):
            logger.info("Running portfolio simulation...")
            portfolio_symbols = [s.strip().upper() for s in advanced_params.get(
                'portfolio_input', '').split(',') if s.strip()]
            if portfolio_symbols:
                results['portfolio_result'] = self.portfolio_analyzer.portfolio_simulation(portfolio_symbols, advanced_params.get(
                    'initial_investment', config.DEFAULT_INVESTMENT), advanced_params.get('strategy', 'equal_weight'))
            else:
                logger.warning(
                    "Portfolio simulation skipped: No symbols provided.")

        results['correlation_matrix'] = None
        if run_advanced.get('run_correlation'):
            logger.info("Running correlation analysis...")
            correlation_symbols = [s.strip().upper() for s in advanced_params.get(
                'correlation_input', '').split(',') if s.strip()]
            if len(correlation_symbols) >= 2:
                results['correlation_matrix'] = self.portfolio_analyzer.advanced_correlation_analysis(
                    correlation_symbols)
            else:
                logger.warning(
                    "Correlation analysis skipped: Less than 2 symbols provided.")

        results['ml_results'] = None
        selected_models_list = advanced_params.get('selected_models', [])
        if run_advanced.get('run_ml_section') and selected_models_list:
            logger.info(
                f"Running ML prediction for models: {selected_models_list}...")
            results['ml_results'] = self.ml_predictor.machine_learning_prediction(
                symbol, selected_models_list, stock_data)
        else:
            logger.info(
                "ML prediction skipped (ML section not enabled or no models selected).")

        results['esg_scores'] = None
        if run_advanced.get('run_esg'):
            logger.info("Fetching ESG scores...")
            results['esg_scores'] = self.data_fetcher.esg_scoring(symbol)

        logger.info(f"--- Full Analysis Complete for {symbol} ---")
        return results

    # --- generate_chat_response (MODIFIED to use updated context prep) ---
    def generate_chat_response(self, user_query: str, analysis_results_dict: Dict[str, Any]) -> str:
        """
        Generates a response from Gemini using the PREPARED context dictionary
        derived from the analysis results.
        """
        if not self.gemini_client:
            logger.error(
                "generate_chat_response called but Gemini client is unavailable.")
            return "Sorry, the AI Chat Assistant is currently unavailable."

        logger.info(
            f"Generating chat response for query: '{user_query[:50]}...'")

        # Prepare context using the utility function
        prepared_context_dict = utils.prepare_llm_context(
            analysis_results_dict)

        if not prepared_context_dict or 'symbol' not in prepared_context_dict or prepared_context_dict.get('error'):
            logger.error(
                f"Prepared context dict is missing, invalid, or contains an error: {prepared_context_dict}")
            return "Sorry, I don't have the necessary analysis context to answer that."

        # --- Build the prompt string from the prepared context dictionary ---
        symbol_for_prompt = prepared_context_dict.get(
            'symbol', 'the selected stock')
        context_lines = []
        context_sections = [
            'company_summary', 'key_ratios', 'latest_technicals', 'technical_patterns',
            # Added yfinance sentiment
            'fundamental_summary', 'average_news_sentiment', 'yfinance_average_sentiment',
            'prophet_forecast_summary', 'ml_prediction_summary', 'analyst_ratings_summary',
            'esg_scores', 'earnings_summary', 'latest_economic_data', 'derived_signals'
        ]
        # Get currency for prompt formatting if needed
        currency_code = analysis_results_dict.get('currency', 'USD')

        for key in context_sections:
            value = prepared_context_dict.get(key)
            if value is None:
                continue  # Skip empty sections

            title = key.replace('_', ' ').title()
            if key == 'yfinance_average_sentiment':
                title = "Average News Sentiment (YFinance)"
            context_lines.append(f"\n**{title}:**")

            if isinstance(value, dict):
                items_str = ""
                for k, v in value.items():
                    sub_key_title = k.replace('_', ' ').title()
                    v_str = str(v)
                    v_display = f"{v_str[:150]}{'...' if len(v_str) > 150 else ''}"
                    items_str += f"\n- {sub_key_title}: {v_display}"
                context_lines.append(items_str if items_str else "\n- N/A")
            elif isinstance(value, list):
                if not value:
                    context_lines.append("\n- N/A")
                else:
                    items_str = "".join(
                        [f"\n- {str(item)[:150]}{'...' if len(str(item)) > 150 else ''}" for item in value[:5]])
                    context_lines.append(items_str)
                    if len(value) > 5:
                        context_lines.append("\n- ... (more items exist)")
            else:
                v_str = str(value)
                context_lines.append(
                    f"\n{v_str[:300]}{'...' if len(v_str) > 300 else ''}")

        context_detail_string = "".join(context_lines)
        # --- End prompt building ---

        # Define the system prompt
        system_prompt = (
            # Added currency code
            f"You are a seasoned financial advisor specializing in stock analysis for {symbol_for_prompt} ({currency_code}).\n"
            f"Your objective is to analyze the provided context data and offer clear, actionable financial advice and recommendations. "
            f"Answer the user's query by examining key data points and providing guidance (e.g., buy, sell, or hold), along with supporting reasoning.\n\n"
            f"**Important Disclaimer:** The advice and recommendations you provide are based solely on the provided context data and are not personalized financial advice. "
            f"Users must conduct their own research or consult a professional financial advisor before making any investment decisions.\n\n"
            f"**Guidelines for Responding to Common Queries:**\n"
            f"- For questions such as 'Should I buy this stock?', structure your response as follows:\n"
            f"    1. **Fundamentals:** Summarize key trends (e.g., revenue growth or decline, profit changes) from the 'Fundamental Summary'.\n"
            # Mention currency
            f"    2. **Technicals:** Report the latest closing price (in {currency_code}) and highlight significant indicators or patterns from the 'Latest Technicals' and 'Technical Patterns' sections.\n"
            f"    3. **Sentiment:** Present the 'Average News Sentiment' from both sources (NewsAPI, YFinance) and any notable trends.\n"
            f"    4. **Forecast & ML Predictions:** Describe the forecast trend ('Prophet Forecast Summary') and the model’s predictions ('ML Prediction Summary') for the next day's return, emphasizing strong signals.\n"
            f"    5. **Analyst View:** Include the 'Analyst Ratings Summary' consensus.\n"
            f"    6. **Actionable Recommendation:** Based on the synthesis of the above data, provide a clear recommendation (e.g., buy, sell, or hold) with supporting evidence from the context.\n"
            f"    7. **Closing Disclaimer:** Remind the user that this guidance is solely based on the provided data and does not constitute direct financial advice.\n"
            f"- For specific data inquiries (e.g., 'What was the revenue last quarter?'), answer directly using the context provided.\n\n"
            f"Your responses should be assertive, structured, and professional. Use bullet points or numbered lists where appropriate.\n\n"
            # Added currency code
            f"--- START CONTEXT DATA FOR {symbol_for_prompt} ({currency_code}) ---\n"
            f"{context_detail_string}"
            f"\n--- END CONTEXT DATA ---"
        )

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}]

        try:
            response = self.gemini_client.chat.completions.create(
                model=config.GEMINI_CHAT_MODEL, messages=messages,
                max_tokens=config.MAX_CHAT_TOKENS, temperature=0.2
            )
            generated_text = response.choices[0].message.content.strip()
            logger.info(
                f"Chat response received, length: {len(generated_text)}")
            return generated_text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            st.error(f"Error communicating with the AI assistant: {e}")
            return "Sorry, I encountered an error while processing your request with the AI assistant."
