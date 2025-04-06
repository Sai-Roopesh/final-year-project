# src/analysis/sentiment.py

import pandas as pd
import logging as logger
from typing import List, Dict, Any, Tuple, Optional
import nltk
from datetime import datetime
import numpy as np
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except LookupError:
    logger.info("NLTK VADER lexicon not found, attempting download...")
    try:
        nltk.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception as e:
        logger.error(f"Failed to download or import VADER lexicon: {e}")
        SentimentIntensityAnalyzer = None  # Ensure it's None if download fails

from src import config, utils
logger = utils.logger


class SentimentAnalysis:
    """Handles sentiment analysis of news articles."""

    def __init__(self):
        self.sia = None
        self.sentiment_available = False
        if SentimentIntensityAnalyzer:
            try:
                self.sia = SentimentIntensityAnalyzer()
                self.sentiment_available = True
                logger.info("SentimentIntensityAnalyzer initialized.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize SentimentIntensityAnalyzer: {e}")
        else:
            logger.warning(
                "Sentiment analysis disabled: VADER lexicon unavailable.")

    # @st.cache_data # Caching might be better handled on the raw news fetch
    def analyze_sentiment(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, Optional[pd.DataFrame]]:
        """Analyzes sentiment of news articles using VADER."""
        if not self.sentiment_available or self.sia is None:
            logger.warning(
                "Sentiment analysis skipped; analyzer not available.")
            return articles, 0.0, None  # Return neutral/empty values

        if not articles:
            logger.info("No articles provided for sentiment analysis.")
            return articles, 0.0, None

        logger.info(f"Analyzing sentiment for {len(articles)} articles.")
        sentiments = []
        analyzed_articles = []
        daily_sentiment_scores = {}  # date_str -> list of compound scores

        for article in articles:
            title = article.get('title', '') or ""
            description = article.get('description', '') or ""
            content = article.get('content', '') or ""  # Use content if needed
            text_to_analyze = f"{title}. {description}".strip()
            # Fallback to content if title/desc are short/empty
            if len(text_to_analyze) < 20 and content:
                # Use start of content
                text_to_analyze = f"{text_to_analyze} {content[:280]}".strip()

            article_with_sentiment = article.copy()  # Avoid modifying input list items

            if text_to_analyze:
                try:
                    sentiment_scores = self.sia.polarity_scores(
                        text_to_analyze)
                    compound_score = sentiment_scores['compound']
                    article_with_sentiment['sentiment'] = sentiment_scores
                    sentiments.append(compound_score)

                    # Store daily sentiment
                    pub_date_str = article.get('publishedAt', '')[
                        :10]  # YYYY-MM-DD
                    if pub_date_str:
                        try:
                            # Basic validation (YYYY-MM-DD format)
                            datetime.strptime(pub_date_str, config.DATE_FORMAT)
                            daily_sentiment_scores.setdefault(
                                pub_date_str, []).append(compound_score)
                        except ValueError:
                            logger.debug(
                                f"Invalid date format '{pub_date_str}' in article, skipping for daily.")

                except Exception as e:
                    logger.warning(
                        f"Sentiment analysis failed for article '{title[:50]}...': {e}")
                    article_with_sentiment['sentiment'] = {
                        'compound': 0.0, 'error': str(e)}
                    sentiments.append(0.0)  # Neutral score on error
            else:
                article_with_sentiment['sentiment'] = {
                    'compound': 0.0, 'error': 'No text content'}
                sentiments.append(0.0)

            analyzed_articles.append(article_with_sentiment)

        avg_sentiment = np.mean(sentiments) if sentiments else 0.0

        # Prepare daily sentiment DataFrame
        daily_data = []
        if daily_sentiment_scores:
            for date_str in sorted(daily_sentiment_scores.keys()):
                scores = daily_sentiment_scores[date_str]
                daily_avg = np.mean(scores) if scores else 0.0
                try:  # Ensure date conversion works before appending
                    daily_data.append({'Date': pd.to_datetime(
                        date_str), 'Daily_Sentiment': daily_avg})
                except ValueError:
                    logger.warning(
                        f"Could not convert date string '{date_str}' for daily sentiment DataFrame.")

        sentiment_df = pd.DataFrame(daily_data) if daily_data else None
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df.set_index('Date', inplace=True)
            # Optional: Calculate rolling average
            # sentiment_df['Rolling_Sentiment_3D'] = sentiment_df['Daily_Sentiment'].rolling(window=3, min_periods=1).mean()

        logger.info(
            f"Sentiment analysis complete. Avg score: {avg_sentiment:.3f}. Daily points: {len(daily_data)}")
        return analyzed_articles, avg_sentiment, sentiment_df
