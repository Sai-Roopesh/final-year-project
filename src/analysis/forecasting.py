# src/analysis/forecasting.py

import streamlit as st  # Keep for cache decorator
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Optional

from src import config, utils

logger = utils.logger


class Forecasting:
    """Handles time series forecasting using Prophet."""

    @staticmethod  # Can be static if it doesn't rely on instance state
    # Cache the forecast result
    @st.cache_data(show_spinner="Generating Prophet forecast...")
    def prophet_forecast(df_hist: pd.DataFrame, days_to_predict: int, sentiment_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Generates a forecast using Prophet, optionally with sentiment as regressor."""
        logger.info(f"Starting Prophet forecast for {days_to_predict} days.")

        # --- Input Checks ---
        if df_hist is None or df_hist.empty:
            logger.warning("Input history DataFrame empty for Prophet.")
            st.warning("Cannot generate forecast: Historical data missing.")
            return None
        required_cols = ['Date', 'Close']
        if not all(col in df_hist.columns for col in required_cols):
            logger.error(
                f"Prophet requires 'Date'/'Close' columns. Found: {df_hist.columns}")
            st.error("Cannot generate forecast: Missing 'Date' or 'Close' column.")
            return None
        min_data_points = 30
        if len(df_hist) < min_data_points:
            logger.warning(
                f"Insufficient data ({len(df_hist)} points) for Prophet (need {min_data_points}).")
            st.warning(
                f"Not enough historical data ({len(df_hist)} days) for reliable forecasting.")
            return None
        if not isinstance(days_to_predict, int) or days_to_predict <= 0:
            logger.warning(f"Invalid 'days_to_predict': {days_to_predict}.")
            st.error("Invalid number of days to forecast.")
            return None

        try:
            # --- Prepare Base Data ---
            prophet_df = df_hist[['Date', 'Close']].copy()
            prophet_df.rename(
                columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            prophet_df['ds'] = pd.to_datetime(
                prophet_df['ds']).dt.tz_localize(None)

            # --- Sentiment Regressor ---
            sentiment_regressor_added = False
            last_sentiment = 0.0
            if sentiment_df is not None and not sentiment_df.empty and 'Daily_Sentiment' in sentiment_df.columns:
                logger.info("Preparing sentiment data as regressor.")
                sentiment_prep = sentiment_df[['Daily_Sentiment']].copy()
                # Ensure index is datetime, timezone-naive, named 'ds'
                sentiment_prep.index = pd.to_datetime(
                    sentiment_prep.index).tz_localize(None)
                sentiment_prep.index.name = 'ds'
                sentiment_prep.rename(
                    columns={'Daily_Sentiment': 'sentiment_score'}, inplace=True)

                # Merge and handle NaNs
                prophet_df = pd.merge(
                    prophet_df, sentiment_prep, on='ds', how='left')
                prophet_df['sentiment_score'].ffill(
                    inplace=True)  # Forward fill sentiment
                prophet_df['sentiment_score'].fillna(
                    0.0, inplace=True)  # Fill initial NaNs

                if prophet_df['sentiment_score'].notna().any():
                    last_sentiment = prophet_df['sentiment_score'].iloc[-1]
                    sentiment_regressor_added = True
                    logger.info(
                        f"Sentiment regressor added. Last value: {last_sentiment:.3f}")
                else:
                    logger.warning(
                        "Sentiment data merged but resulted in all NaNs.")
                    if 'sentiment_score' in prophet_df.columns:
                        prophet_df.drop(
                            columns=['sentiment_score'], inplace=True)
            else:
                logger.info(
                    "Proceeding without sentiment regressor (data unavailable or empty).")

            # --- Log Transform (Optional) ---
            # Consider making this configurable via config.py
            use_log_transform = True
            if use_log_transform and (prophet_df['y'] <= 0).any():
                logger.warning(
                    "Log transform skipped: Non-positive 'y' values found.")
                use_log_transform = False  # Disable if data is not suitable

            if use_log_transform:
                logger.info("Applying log transform to 'y' data.")
                # Keep original for potential later use
                prophet_df['y_orig'] = prophet_df['y']
                prophet_df['y'] = np.log(prophet_df['y'])  # Assumes y > 0

            # --- Initialize & Fit Model ---
            model = Prophet(
                daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,  # Enable weekly
                interval_width=config.PROPHET_INTERVAL_WIDTH,
                changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR_SCALE
            )
            if sentiment_regressor_added:
                model.add_regressor('sentiment_score')

            model.fit(prophet_df)

            # --- Predict Future ---
            future = model.make_future_dataframe(periods=days_to_predict)
            if sentiment_regressor_added:
                # Assume last sentiment persists
                future['sentiment_score'] = last_sentiment
                logger.debug(
                    f"Added future sentiment values ({last_sentiment:.3f}) to prediction frame.")

            forecast = model.predict(future)

            # --- Inverse Transform & Clip ---
            cols_to_transform = ['yhat', 'yhat_lower', 'yhat_upper']
            if use_log_transform:
                logger.info("Applying inverse log transform to forecast.")
                for col in cols_to_transform:
                    if col in forecast.columns:
                        forecast[col] = np.exp(forecast[col])
            # Always clip predictions to be non-negative
            for col in cols_to_transform:
                if col in forecast.columns:
                    forecast[col] = forecast[col].clip(lower=0.0)

            logger.info("Prophet forecast generated successfully.")
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        except Exception as e:
            logger.exception(f"Error during Prophet forecasting: {e}")
            st.error(f"Failed to generate Prophet forecast: {e}")
            return None
