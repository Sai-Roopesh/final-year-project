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
        # ... (Implementation remains the same as artifact main_analyser_ml_call_fix) ...
        # This function correctly gathers all results into the 'results' dictionary.
        logger.info(f"--- Starting Full Analysis for {symbol} ---")
        results = {'symbol': symbol}

        # 1. Core Data Fetching
        logger.info("Fetching core data...")
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
            logger.error(f"Core data fetching failed for {symbol}.")
            st.error(f"Failed fetch historical stock data for {symbol}.")
            return results

        # 2. Technical Analysis
        logger.info("Running technical analysis...")
        results['tech_data'] = self.technical_analyzer.calculate_technical_indicators(
            stock_data)
        results['patterns'] = self.technical_analyzer.analyze_patterns(
            results['tech_data'])

        # 3. News & Sentiment
        logger.info("Fetching news and analyzing sentiment...")
        news_days_back = min(90, (end_date - start_date).days +
                             1) if start_date and end_date else 90
        news_start_date = end_date - \
            timedelta(days=news_days_back) if end_date else datetime.now(
            ) - timedelta(days=90)
        news_end_date = end_date if end_date else datetime.now()
        news_articles = self.data_fetcher.load_news(
            symbol, news_start_date, news_end_date)
        sentiment_method_selected = advanced_params.get(
            'sentiment_method', 'vader').lower()
        logger.info(f"Selected sentiment method: {sentiment_method_selected}")
        analyzed_articles, avg_sentiment, sentiment_df = self.sentiment_analyzer.analyze_sentiment(
            news_articles, method=sentiment_method_selected)
        results['analyzed_articles'] = analyzed_articles
        results['avg_sentiment'] = avg_sentiment
        results['sentiment_df'] = sentiment_df
        results['sentiment_method_used'] = sentiment_method_selected

        # 4. Forecasting (Prophet)
        logger.info("Generating Prophet forecast...")
        forecast_days = advanced_params.get('prediction_days', 30)
        results['prophet_forecast_data'] = self.forecaster.prophet_forecast(
            stock_data, forecast_days, sentiment_df)

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

        # --- Optional Economic Indicators ---
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

        # --- Optional Advanced Analyses ---
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

    # --- generate_chat_response (MODIFIED) ---

    def generate_chat_response(self, user_query: str, prepared_context_dict: Dict[str, Any]) -> str:
        """
        Generates a response from Gemini using the PREPARED context dictionary.

        Args:
            user_query (str): The user's question.
            prepared_context_dict (Dict[str, Any]): The dictionary returned by utils.prepare_llm_context.

        Returns:
            str: The generated response from the LLM.
        """
        if not self.gemini_client:
            logger.error(
                "generate_chat_response called but Gemini client is unavailable.")
            return "Sorry, the AI Chat Assistant is currently unavailable."

        logger.info(
            f"Generating chat response for query: '{user_query[:50]}...' using prepared context.")

        if not prepared_context_dict or 'symbol' not in prepared_context_dict:
            logger.error("Prepared context dict is missing or invalid.")
            return "Sorry, I don't have the necessary analysis context to answer that."

        # --- Build the prompt string from the prepared context dictionary ---
        symbol_for_prompt = prepared_context_dict.get(
            'symbol', 'the selected stock')
        context_lines = []
        for key, value in prepared_context_dict.items():
            # Skip symbol (already in system prompt) and None values
            if key == 'symbol' or value is None:
                continue

            # Format keys nicely
            title = key.replace('_', ' ').title()
            context_lines.append(f"\n**{title}:**")

            # Format values based on type
            if isinstance(value, dict):
                # Format dictionary items
                items_str = "".join(
                    [f"\n- {k.replace('_', ' ').title()}: {str(v)[:150]}{'...' if len(str(v)) > 150 else ''}" for k, v in value.items()])
                context_lines.append(items_str if items_str else "\n- N/A")
            elif isinstance(value, list):
                # Format list items
                if not value:
                    context_lines.append("\n- N/A")
                else:
                    context_lines.append("".join(
                        # Show first 5
                        [f"\n- {str(item)[:150]}{'...' if len(str(item)) > 150 else ''}" for item in value[:5]]))
                    if len(value) > 5:
                        context_lines.append("\n- ... (more items)")
            else:
                # Format other types as string (truncate long strings)
                v_str = str(value)
                # Increased truncation limit slightly
                context_lines.append(
                    f"\n{v_str[:300]}{'...' if len(v_str) > 300 else ''}")

        context_detail_string = "".join(context_lines)
        # --- End prompt building ---

        # Define the system prompt (using your advisor persona)
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
            f"{context_detail_string}"  # Append the formatted context details
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
