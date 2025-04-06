# src/analysis/sentiment.py

import streamlit as st  # Needed for caching
import pandas as pd
import logging as logger
from typing import List, Dict, Any, Tuple, Optional
import nltk
from datetime import datetime
import numpy as np
import torch  # Added for FinBERT
# Added for FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except LookupError:
    logger.info("NLTK VADER lexicon not found, attempting download...")
    try:
        nltk.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception as e:
        logger.error(f"Failed to download or import VADER lexicon: {e}")
        SentimentIntensityAnalyzer = None

from src import config, utils
logger = utils.logger

# --- FinBERT Model Loading (Cached) ---


@st.cache_resource(show_spinner="Loading FinBERT sentiment model...")
def load_finbert_model():
    """Loads the FinBERT model and tokenizer."""
    try:
        model_name = "ProsusAI/finbert"  # Common choice for financial sentiment
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info(f"FinBERT model '{model_name}' loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(
            f"Failed to load FinBERT model '{model_name}': {e}", exc_info=True)
        st.error(f"Could not load FinBERT sentiment model. Please check logs. Falling back to VADER if available.", icon="⚠️")
        return None, None


class SentimentAnalysis:
    """Handles sentiment analysis of news articles using VADER or FinBERT."""

    def __init__(self):
        self.sia = None
        self.vader_available = False
        if SentimentIntensityAnalyzer:
            try:
                self.sia = SentimentIntensityAnalyzer()
                self.vader_available = True
                logger.info("VADER SentimentIntensityAnalyzer initialized.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize VADER SentimentIntensityAnalyzer: {e}")
        else:
            logger.warning(
                "VADER sentiment analysis unavailable: VADER lexicon missing.")

        # FinBERT model is loaded on demand via the cached function

    def _analyze_sentiment_vader(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, Optional[pd.DataFrame]]:
        """Analyzes sentiment using VADER (internal helper)."""
        if not self.vader_available or self.sia is None:
            logger.warning("VADER analysis skipped; analyzer not available.")
            # Return neutral results but keep article structure
            analyzed_articles = []
            for article in articles:
                article_with_sentiment = article.copy()
                article_with_sentiment['sentiment'] = {
                    'compound': 0.0, 'label': 'neutral', 'score': 0.0, 'error': 'VADER unavailable'}
                analyzed_articles.append(article_with_sentiment)
            return analyzed_articles, 0.0, None

        logger.info(
            f"Analyzing sentiment for {len(articles)} articles using VADER.")
        sentiments_compound = []
        analyzed_articles = []
        daily_sentiment_scores = {}

        for article in articles:
            title = article.get('title', '') or ""
            description = article.get('description', '') or ""
            content = article.get('content', '') or ""
            text_to_analyze = f"{title}. {description}".strip()
            if len(text_to_analyze) < 20 and content:
                text_to_analyze = f"{text_to_analyze} {content[:280]}".strip()

            article_with_sentiment = article.copy()

            if text_to_analyze:
                try:
                    vader_scores = self.sia.polarity_scores(text_to_analyze)
                    compound_score = vader_scores['compound']
                    # Add VADER-specific results
                    article_with_sentiment['sentiment_vader'] = vader_scores
                    # Add standardized results
                    label = 'positive' if compound_score > 0.05 else 'negative' if compound_score < - \
                        0.05 else 'neutral'
                    article_with_sentiment['sentiment'] = {
                        # Use compound as score for VADER
                        'compound': compound_score, 'label': label, 'score': compound_score}
                    sentiments_compound.append(compound_score)

                    pub_date_str = article.get('publishedAt', '')[:10]
                    if pub_date_str:
                        try:
                            datetime.strptime(pub_date_str, config.DATE_FORMAT)
                            daily_sentiment_scores.setdefault(
                                pub_date_str, []).append(compound_score)
                        except ValueError:
                            logger.debug(
                                f"Invalid date format '{pub_date_str}' in article (VADER).")

                except Exception as e:
                    logger.warning(
                        f"VADER analysis failed for article '{title[:50]}...': {e}")
                    article_with_sentiment['sentiment'] = {
                        'compound': 0.0, 'label': 'neutral', 'score': 0.0, 'error': str(e)}
                    sentiments_compound.append(0.0)
            else:
                article_with_sentiment['sentiment'] = {
                    'compound': 0.0, 'label': 'neutral', 'score': 0.0, 'error': 'No text content'}
                sentiments_compound.append(0.0)

            analyzed_articles.append(article_with_sentiment)

        avg_sentiment = np.mean(
            sentiments_compound) if sentiments_compound else 0.0
        sentiment_df = self._create_daily_df(daily_sentiment_scores)

        logger.info(
            f"VADER analysis complete. Avg score: {avg_sentiment:.3f}.")
        return analyzed_articles, avg_sentiment, sentiment_df

    def _analyze_sentiment_finbert(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, Optional[pd.DataFrame]]:
        """Analyzes sentiment using FinBERT (internal helper)."""
        logger.info(
            f"Analyzing sentiment for {len(articles)} articles using FinBERT.")
        tokenizer, model = load_finbert_model()  # Load cached model/tokenizer

        if tokenizer is None or model is None:
            logger.error(
                "FinBERT model/tokenizer not loaded. Cannot perform FinBERT analysis.")
            # Return neutral results but keep article structure
            analyzed_articles = []
            for article in articles:
                article_with_sentiment = article.copy()
                article_with_sentiment['sentiment'] = {
                    'label': 'neutral', 'score': 0.0, 'error': 'FinBERT model unavailable'}
                analyzed_articles.append(article_with_sentiment)
            return analyzed_articles, 0.0, None

        sentiments_mapped = []  # Store mapped scores (-1, 0, 1)
        analyzed_articles = []
        daily_sentiment_scores = {}  # Store mapped scores per day

        # Map labels to scores for aggregation
        label_to_score = {'positive': 1, 'negative': -1, 'neutral': 0}

        with torch.no_grad():  # Disable gradient calculation for inference
            for article in articles:
                title = article.get('title', '') or ""
                description = article.get('description', '') or ""
                content = article.get('content', '') or ""
                text_to_analyze = f"{title}. {description}".strip()
                if len(text_to_analyze) < 20 and content:
                    text_to_analyze = f"{text_to_analyze} {content[:280]}".strip(
                    )

                article_with_sentiment = article.copy()

                if text_to_analyze:
                    try:
                        # Tokenize and predict
                        inputs = tokenizer(
                            text_to_analyze, return_tensors="pt", truncation=True, padding=True, max_length=512)
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probabilities = torch.softmax(logits, dim=-1)
                        scores, predictions = torch.max(probabilities, dim=-1)

                        predicted_label_id = predictions.item()
                        predicted_label = model.config.id2label[predicted_label_id]
                        predicted_score = scores.item()

                        # Store detailed FinBERT results if needed
                        # article_with_sentiment['sentiment_finbert_probs'] = probabilities.tolist()[0]
                        # article_with_sentiment['sentiment_finbert_label'] = predicted_label
                        # article_with_sentiment['sentiment_finbert_score'] = predicted_score

                        # Store standardized results
                        article_with_sentiment['sentiment'] = {
                            'label': predicted_label, 'score': predicted_score}

                        # Map label to score for aggregation
                        mapped_score = label_to_score.get(predicted_label, 0)
                        sentiments_mapped.append(mapped_score)

                        # Store daily mapped score
                        pub_date_str = article.get('publishedAt', '')[:10]
                        if pub_date_str:
                            try:
                                datetime.strptime(
                                    pub_date_str, config.DATE_FORMAT)
                                daily_sentiment_scores.setdefault(
                                    pub_date_str, []).append(mapped_score)
                            except ValueError:
                                logger.debug(
                                    f"Invalid date format '{pub_date_str}' in article (FinBERT).")

                    except Exception as e:
                        # Avoid long tracebacks in logs
                        logger.warning(
                            f"FinBERT analysis failed for article '{title[:50]}...': {e}", exc_info=False)
                        article_with_sentiment['sentiment'] = {
                            'label': 'neutral', 'score': 0.0, 'error': str(e)}
                        sentiments_mapped.append(0)  # Neutral score on error
                else:
                    article_with_sentiment['sentiment'] = {
                        'label': 'neutral', 'score': 0.0, 'error': 'No text content'}
                    sentiments_mapped.append(0)

                analyzed_articles.append(article_with_sentiment)

        # Calculate average based on mapped scores
        avg_sentiment_mapped = np.mean(
            sentiments_mapped) if sentiments_mapped else 0.0
        sentiment_df = self._create_daily_df(
            daily_sentiment_scores)  # Use helper for daily df

        logger.info(
            f"FinBERT analysis complete. Avg mapped score: {avg_sentiment_mapped:.3f}.")
        # Return the average *mapped* score for consistency with VADER's compound range [-1, 1]
        return analyzed_articles, avg_sentiment_mapped, sentiment_df

    def _create_daily_df(self, daily_scores: Dict[str, List[float]]) -> Optional[pd.DataFrame]:
        """Helper function to create the daily sentiment DataFrame."""
        daily_data = []
        if daily_scores:
            for date_str in sorted(daily_scores.keys()):
                scores = daily_scores[date_str]
                daily_avg = np.mean(scores) if scores else 0.0
                try:
                    daily_data.append({'Date': pd.to_datetime(
                        date_str), 'Daily_Sentiment': daily_avg})
                except ValueError:
                    logger.warning(
                        f"Could not convert date string '{date_str}' for daily sentiment DataFrame.")

        sentiment_df = pd.DataFrame(daily_data) if daily_data else None
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df.set_index('Date', inplace=True)
        return sentiment_df

    # --- Updated Main Method ---
    def analyze_sentiment(self, articles: List[Dict[str, Any]], method: str = 'vader') -> Tuple[List[Dict[str, Any]], float, Optional[pd.DataFrame]]:
        """
        Analyzes sentiment of news articles using the specified method ('vader' or 'finbert').

        Returns:
            Tuple containing:
                - List of articles with added 'sentiment' dictionary ({'label': str, 'score': float}).
                - Average sentiment score (VADER compound or FinBERT mapped score).
                - Optional DataFrame of daily average sentiment scores.
        """
        if not articles:
            logger.info("No articles provided for sentiment analysis.")
            return [], 0.0, None

        if method.lower() == 'finbert':
            logger.info("Using FinBERT method for sentiment analysis.")
            return self._analyze_sentiment_finbert(articles)
        elif method.lower() == 'vader':
            logger.info("Using VADER method for sentiment analysis.")
            if not self.vader_available:
                st.warning(
                    "VADER selected but unavailable. No sentiment analysis performed.", icon="⚠️")
                # Will return neutral results
                return self._analyze_sentiment_vader(articles)
            return self._analyze_sentiment_vader(articles)
        else:
            logger.warning(
                f"Unknown sentiment analysis method '{method}'. Defaulting to VADER.")
            if not self.vader_available:
                st.warning(
                    f"Unknown method '{method}' and VADER unavailable. No sentiment analysis performed.", icon="⚠️")
                # Will return neutral results
                return self._analyze_sentiment_vader(articles)
            return self._analyze_sentiment_vader(articles)
