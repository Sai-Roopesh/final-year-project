# src/main_analyzer.py

from datetime import datetime, timedelta  # Ensure datetime is imported
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
from . import utils

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
        self.sentiment_analyzer = SentimentAnalysis()  # Instantiated here
        self.forecaster = Forecasting()
        self.ml_predictor = MLPredictor()
        self.portfolio_analyzer = Portfolio(self.data_fetcher)
        self.gemini_client = gemini_client

        # Optionally warn if VADER isn't available, FinBERT loads on demand
        if not self.sentiment_analyzer.vader_available:
            # Use st.toast for less intrusive warnings if preferred
            st.warning(
                "VADER sentiment analysis unavailable (lexicon missing). FinBERT will be used if selected.", icon="⚠️")

    def run_full_analysis(self, symbol: str, start_date: datetime, end_date: datetime,
                          run_advanced: Dict[str, bool],
                          advanced_params: Dict[str, Any]
                          ) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline for a given stock symbol.

        Args:
            symbol (str): The stock symbol.
            start_date (datetime): Start date for historical data.
            end_date (datetime): End date for historical data.
            run_advanced (Dict[str, bool]): Dict indicating whether to run optional analyses.
            advanced_params (Dict[str, Any]): Dict containing parameters for optional analyses.

        Returns:
            Dict[str, Any]: A dictionary containing all analysis results.
        """
        logger.info(f"--- Starting Full Analysis for {symbol} ---")
        results = {'symbol': symbol}

        # 1. Core Data Fetching
        logger.info("Fetching core data...")
        # Ensure start_date and end_date are datetime objects
        if not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())

        stock_data = self.data_fetcher.load_stock_data(
            symbol, start_date, end_date)
        stock_info = self.data_fetcher.load_stock_info(symbol)
        results['stock_data'] = stock_data
        results['stock_info'] = stock_info

        if stock_data is None or stock_data.empty:
            logger.error(
                f"Core data fetching failed for {symbol}. Aborting analysis.")
            st.error(
                f"Failed to fetch historical stock data for {symbol}. Analysis cannot proceed.")
            return results  # Return early if no stock data

        # 2. Technical Analysis
        logger.info("Running technical analysis...")
        results['tech_data'] = self.technical_analyzer.calculate_technical_indicators(
            stock_data)
        results['patterns'] = self.technical_analyzer.analyze_patterns(
            results['tech_data'])

        # 3. News & Sentiment
        logger.info("Fetching news and analyzing sentiment...")
        # Use a reasonable date range for news, e.g., last 90 days or selected range
        news_days_back = min(90, (end_date - start_date).days +
                             1) if start_date and end_date else 90
        news_start_date = end_date - \
            timedelta(days=news_days_back) if end_date else datetime.now(
            ) - timedelta(days=90)
        news_end_date = end_date if end_date else datetime.now()

        news_articles = self.data_fetcher.load_news(
            symbol, news_start_date, news_end_date)

        # --- Get selected sentiment method ---
        sentiment_method_selected = advanced_params.get(
            'sentiment_method', 'vader').lower()
        logger.info(f"Selected sentiment method: {sentiment_method_selected}")

        # --- Call analyze_sentiment with the selected method ---
        analyzed_articles, avg_sentiment, sentiment_df = self.sentiment_analyzer.analyze_sentiment(
            news_articles,
            method=sentiment_method_selected  # Pass the method here
        )
        results['analyzed_articles'] = analyzed_articles
        results['avg_sentiment'] = avg_sentiment
        results['sentiment_df'] = sentiment_df
        # Store which method was used
        results['sentiment_method_used'] = sentiment_method_selected

        # 4. Forecasting (Prophet) - Pass sentiment_df which might be based on VADER or FinBERT mapped scores
        logger.info("Generating Prophet forecast...")
        forecast_days = advanced_params.get('prediction_days', 30)
        results['prophet_forecast_data'] = self.forecaster.prophet_forecast(
            # Pass the potentially mapped sentiment_df
            stock_data, forecast_days, sentiment_df)

        # 5. Financial Data (Dividends, Earnings, Fundamentals, Analyst Recs)
        if run_advanced.get('show_dividends'):
            logger.info("Fetching dividend history...")
            results['dividend_history'] = self.data_fetcher.get_dividend_history(
                symbol)
        else:
            # Ensure key exists even if not run
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

        # --- Optional Economic Indicators ---
        results['economic_data'] = None  # Initialize key
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
                # Pass datetime objects for start and end dates
                fetched_econ_data = self.data_fetcher.get_economic_indicators(
                    series_list, start_date, end_date
                )
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

        # --- Optional Advanced Analyses ---
        # 6. Portfolio Simulation
        results['portfolio_result'] = None  # Initialize key
        if run_advanced.get('run_portfolio'):
            logger.info("Running portfolio simulation...")
            portfolio_symbols = [s.strip().upper() for s in advanced_params.get(
                'portfolio_input', '').split(',') if s.strip()]
            if portfolio_symbols:
                results['portfolio_result'] = self.portfolio_analyzer.portfolio_simulation(
                    portfolio_symbols,
                    advanced_params.get(
                        'initial_investment', config.DEFAULT_INVESTMENT),
                    advanced_params.get('strategy', 'equal_weight')
                )
            else:
                logger.warning(
                    "Portfolio simulation skipped: No symbols provided.")

        # 7. Correlation Analysis
        results['correlation_matrix'] = None  # Initialize key
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

        # 8. ML Prediction
        results['ml_results'] = None  # Initialize key
        selected_models_list = advanced_params.get('selected_models', [])
        # Check if the ML section itself was enabled via run_ml_section
        if run_advanced.get('run_ml_section') and selected_models_list:
            logger.info(
                f"Running ML prediction for models: {selected_models_list}...")
            # Pass the original stock_data (or tech_data if features rely on it)
            results['ml_results'] = self.ml_predictor.machine_learning_prediction(
                symbol, selected_models_list, stock_data  # Pass necessary data
            )
        else:
            logger.info(
                "ML prediction skipped (ML section not enabled or no models selected).")

        # 9. ESG Scoring
        results['esg_scores'] = None  # Initialize key
        if run_advanced.get('run_esg'):
            logger.info("Fetching ESG scores...")
            results['esg_scores'] = self.data_fetcher.esg_scoring(symbol)

        logger.info(f"--- Full Analysis Complete for {symbol} ---")
        return results

    def generate_chat_response(self, user_query: str, context_data: Dict[str, Any]) -> str:
        """ Generates a response from Gemini based on provided context and user query. """
        if not self.gemini_client:
            logger.error(
                "generate_chat_response called but Gemini client is unavailable.")
            return "Sorry, the AI Chat Assistant is currently unavailable."

        logger.info(
            f"Generating chat response for query: '{user_query[:50]}...'")

        # Prepare context string using the utility function
        # This function needs access to the analysis results dictionary
        context_str = utils.prepare_llm_context(context_data)

        # Add the system prompt
        # Ensure context_data['symbol'] exists or provide default
        symbol_for_prompt = context_data.get('symbol', 'the selected stock')
        system_prompt = (
            f"You are a seasoned financial advisor specializing in stock analysis for {symbol_for_prompt}.\n"
            f"Your objective is to analyze the provided context data and offer clear, actionable financial advice and recommendations. "
            f"Answer the user's query by examining key data points and providing guidance (e.g., buy, sell, or hold), along with supporting reasoning.\n\n"
            f"**Important Disclaimer:** The advice and recommendations you provide are based solely on the provided context data and are not personalized financial advice. "
            f"Users must conduct their own research or consult a professional financial advisor before making any investment decisions.\n\n"
            f"**Guidelines for Responding to Common Queries:**\n"
            f"- For questions such as 'Should I buy this stock?', structure your response as follows:\n"
            f"    1. **Fundamentals:** Summarize key trends (e.g., revenue growth or decline, profit changes) from the 'Fundamental Summary'.\n"
            f"    2. **Technicals:** Report the latest closing price and highlight significant indicators or patterns from the 'Technical Analysis' section.\n"
            f"    3. **Sentiment:** Present the 'Average News Sentiment' and any notable trends from the sentiment analysis.\n"
            f"    4. **Forecast & ML Predictions:** Describe the forecast trend and the model’s predictions for the next day's return, emphasizing strong signals.\n"
            f"    5. **Actionable Recommendation:** Based on the above data, provide a clear recommendation (e.g., buy, sell, or hold) with supporting evidence.\n"
            f"    6. **Closing Disclaimer:** Remind the user that this guidance is solely based on the provided data and does not constitute direct financial advice.\n"
            f"- For specific data inquiries (e.g., 'What was the revenue last quarter?'), answer directly using the context provided.\n\n"
            f"Your responses should be assertive, structured, and professional. Use bullet points or numbered lists where appropriate.\n\n"
            f"--- START CONTEXT DATA FOR {symbol_for_prompt} ---\n"
            f"{context_str}"  # Append the formatted context from utils
            f"\n--- END CONTEXT DATA ---"
        )

        messages = [
            # Use the combined prompt
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.gemini_client.chat.completions.create(
                model=config.GEMINI_CHAT_MODEL,
                messages=messages,
                max_tokens=config.MAX_CHAT_TOKENS,
                temperature=0.2  # Lower temperature for more factual advice
            )
            generated_text = response.choices[0].message.content.strip()
            logger.info(
                f"Chat response received, length: {len(generated_text)}")
            return generated_text

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            # Show error in UI
            st.error(f"Error communicating with the AI assistant: {e}")
            return "Sorry, I encountered an error while processing your request with the AI assistant."
