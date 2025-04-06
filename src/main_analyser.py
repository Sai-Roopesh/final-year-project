# src/main_analyzer.py

from datetime import timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from newsapi import NewsApiClient
from openai import OpenAI
import streamlit as st  # For session state access maybe
# import config

from src import config
# Import analysis components
from .analysis.data_fetcher import DataFetcher
from .analysis.technical import TechnicalAnalysis
from .analysis.sentiment import SentimentAnalysis
from .analysis.forecasting import Forecasting
from .analysis.ml_predictor import MLPredictor
from .analysis.portfolio import Portfolio

from . import utils  # For logger

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
        # These components are mostly static methods now or self-contained
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalysis()
        self.forecaster = Forecasting()
        self.ml_predictor = MLPredictor()
        # Portfolio needs DataFetcher (or specific methods)
        self.portfolio_analyzer = Portfolio(self.data_fetcher)
        self.gemini_client = gemini_client  # Keep for chat if needed directly

        # NLTK check (can be done once at init)
        if not self.sentiment_analyzer.sentiment_available:
            st.warning(
                "Sentiment Analysis disabled due to NLTK VADER issues.", icon="⚠️")

    def run_full_analysis(self, symbol: str, start_date, end_date,
                          # Flags for optional analyses
                          run_advanced: Dict[str, bool],
                          # Params like portfolio symbols, investment
                          advanced_params: Dict[str, Any]
                          ) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline for a given stock symbol.

        Args:
            symbol: The stock symbol.
            start_date: Start date for historical data.
            end_date: End date for historical data.
            run_advanced: Dict indicating whether to run optional analyses (ml, portfolio, etc.).
            advanced_params: Dict containing parameters for optional analyses.

        Returns:
            A dictionary containing all analysis results.
        """
        logger.info(f"--- Starting Full Analysis for {symbol} ---")
        results = {'symbol': symbol}  # Initialize results dict

        # 1. Core Data Fetching
        logger.info("Fetching core data...")
        stock_data = self.data_fetcher.load_stock_data(
            symbol, start_date, end_date)
        stock_info = self.data_fetcher.load_stock_info(symbol)
        results['stock_data'] = stock_data
        results['stock_info'] = stock_info

        # Crucial Check: Stop if core data failed
        if stock_data is None or stock_data.empty:
            logger.error(
                f"Core data fetching failed for {symbol}. Aborting analysis.")
            # No st.error here, let main app handle UI based on returned None
            return results  # Return partial results indicating failure

        # 2. Technical Analysis
        logger.info("Running technical analysis...")
        results['tech_data'] = self.technical_analyzer.calculate_technical_indicators(
            stock_data)
        results['patterns'] = self.technical_analyzer.analyze_patterns(
            results['tech_data'])

        # 3. News & Sentiment
        logger.info("Fetching news and analyzing sentiment...")
        # Define date range for news (e.g., use analysis period or fixed lookback)
        # Example: last 90 days or period start
        news_start = max(start_date, end_date - timedelta(days=90))
        news_articles = self.data_fetcher.load_news(
            symbol, news_start, end_date)
        analyzed_articles, avg_sentiment, sentiment_df = self.sentiment_analyzer.analyze_sentiment(
            news_articles)
        results['analyzed_articles'] = analyzed_articles
        results['avg_sentiment'] = avg_sentiment
        results['sentiment_df'] = sentiment_df

        # 4. Forecasting (Prophet)
        logger.info("Generating Prophet forecast...")
        forecast_days = advanced_params.get(
            'prediction_days', 30)  # Get days from params
        results['prophet_forecast_data'] = self.forecaster.prophet_forecast(
            stock_data, forecast_days, sentiment_df)  # Pass sentiment_df

        # 5. Financial Data (Dividends, Earnings)
        if run_advanced.get('show_dividends'):  # Check flag from UI
            logger.info("Fetching dividend history...")
            results['dividend_history'] = self.data_fetcher.get_dividend_history(
                symbol)
        # Always fetch earnings? Or add flag? Let's always fetch for now.
        logger.info("Fetching earnings calendar...")
        results['earnings_calendar'] = self.data_fetcher.get_earnings_calendar(
            symbol)

        # --- Optional Advanced Analyses ---

        # 6. Portfolio Simulation
        if run_advanced.get('run_portfolio'):
            logger.info("Running portfolio simulation...")
            portfolio_symbols = [s.strip().upper() for s in advanced_params.get(
                'portfolio_input', '').split(',') if s.strip()]
            if portfolio_symbols:
                results['portfolio_result'] = self.portfolio_analyzer.portfolio_simulation(
                    portfolio_symbols, advanced_params.get(
                        'initial_investment', config.DEFAULT_INVESTMENT),
                    advanced_params.get('strategy', 'equal_weight')
                )
            else:
                logger.warning(
                    "Portfolio simulation skipped: No symbols provided.")

        # 7. Correlation Analysis
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
        # Check if the new UI parameter exists and has models selected
        selected_models_list = advanced_params.get('selected_models', [])
        # Or check the run_ml_section flag if you used Option 1 in the UI
        # run_ml_flag = run_advanced.get('run_ml_section', False)

        # if run_ml_flag and selected_models_list: # If using Option 1
        if selected_models_list:  # If using Option 2 (simpler)
            logger.info(
                f"Running ML prediction for models: {selected_models_list}...")
            results['ml_results'] = self.ml_predictor.machine_learning_prediction(
                symbol, selected_models_list  # Pass the list of selected models
            )
        else:
            logger.info("ML prediction skipped (no models selected).")
            results['ml_results'] = None  # Ensure it's None if not run

        # 9. ESG Scoring
        if run_advanced.get('run_esg'):
            logger.info("Fetching ESG scores...")
            results['esg_scores'] = self.data_fetcher.esg_scoring(symbol)

        logger.info(f"--- Full Analysis Complete for {symbol} ---")
        return results

    # --- Chat Functionality ---
    # Keep this separate as it uses context derived from the analysis results
    def generate_chat_response(self, user_query: str, context_data: Dict[str, Any]) -> str:
        """ Generates a response from Gemini based on provided context and user query. """
        if not self.gemini_client:
            logger.error(
                "generate_chat_response called but Gemini client is unavailable.")
            return "Sorry, the AI Chat Assistant is currently unavailable."

        logger.info(
            f"Generating chat response for query: '{user_query[:50]}...'")

        # Build the prompt string using the structured context from utils.prepare_llm_context
        system_prompt = (
            f"You are a helpful financial analyst assistant for the stock: {context_data.get('symbol', 'N/A')}.\n"
            f"Answer the user's query concisely using ONLY the provided context data below.\n"
            f"Do NOT use external data or prior knowledge.\n"
            f"If the context doesn't contain the answer, state that the information is not available in the provided data.\n"
            f"Format reasonably (e.g., use bullet points for lists).\n\n"
            f"--- START CONTEXT DATA ---\n"
        )
        context_str = system_prompt
        # Iterate through the prepared context and format it nicely
        for key, value in context_data.items():
            if key == 'symbol':
                continue  # Already in system prompt
            if value is None:
                continue

            context_str += f"\n**{key.replace('_', ' ').title()}:**\n"
            if isinstance(value, dict):
                for k, v in value.items():
                    # Limit length of values if necessary
                    v_str = str(v)
                    if len(v_str) > 150:
                        v_str = v_str[:150] + "..."
                    context_str += f"- {k.replace('_', ' ').title()}: {v_str}\n"
            elif isinstance(value, list):
                if not value:
                    context_str += "- N/A\n"
                else:
                    # Show max 5 items
                    context_str += "".join(
                        [f"- {item}\n" for item in value[:5]])
            else:
                # Limit length
                v_str = str(value)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                context_str += f"{v_str}\n"

        context_str += "\n--- END CONTEXT DATA ---"

        messages = [
            {"role": "system", "content": context_str},
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.gemini_client.chat.completions.create(
                model=config.GEMINI_CHAT_MODEL,
                messages=messages,
                max_tokens=config.MAX_CHAT_TOKENS,
                temperature=0.5  # Balance creativity/factuality
            )
            generated_text = response.choices[0].message.content.strip()
            logger.info(
                f"Chat response received, length: {len(generated_text)}")
            return generated_text

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            st.error(f"Error communicating with the AI assistant: {e}")
            return "Sorry, I encountered an error processing your request."
