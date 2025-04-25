# src/analysis/forecasting.py

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Optional

# Assuming config and utils (with logger) are imported correctly relative to this file's location
from .. import config
from .. import utils

logger = utils.logger


class Forecasting:
    """Handles time series forecasting using Prophet."""

    @staticmethod
    @st.cache_data(show_spinner="Generating Prophet forecast...")
    # --- MODIFIED Function Signature ---
    def prophet_forecast(
        df_hist: pd.DataFrame,
        days_to_predict: int,
        newsapi_sentiment_df: Optional[pd.DataFrame] = None,
        yfinance_sentiment_df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        Generates a forecast using Prophet, optionally with sentiment regressors
        from NewsAPI and/or yfinance.
        """
        logger.info(
            f"Starting Prophet forecast for {days_to_predict} days (with dual sentiment attempt).")

        # --- Input Checks (remain the same) ---
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
            # Ensure ds is datetime and timezone-naive
            prophet_df['ds'] = pd.to_datetime(
                prophet_df['ds']).dt.tz_localize(None)

            # --- Process Regressors ---
            regressors_added = []
            last_sentiment_values = {}

            # Helper function to process a sentiment dataframe
            def process_sentiment(df, name):
                # Use logger defined at the class or module level
                global logger  # Or pass logger if not global/module level in your setup

                if df is not None and not df.empty and 'Daily_Sentiment' in df.columns:
                    logger.info(
                        f"Preparing sentiment data from {name} as regressor.")
                    sentiment_prep = df[['Daily_Sentiment']].copy()

                    # Ensure index is datetime first
                    try:
                        sentiment_prep.index = pd.to_datetime(
                            sentiment_prep.index)
                    except Exception as e:
                        logger.error(
                            f"Failed to convert index to datetime for {name} sentiment: {e}")
                        return None, None, 0.0

                    # --- CORRECTED LINE: Remove .dt ---
                    # Localize directly on the DatetimeIndex
                    try:
                        sentiment_prep.index = sentiment_prep.index.tz_localize(
                            None)
                    except TypeError:  # Handle cases where index might already be tz-naive
                        logger.debug(
                            f"Index for {name} sentiment already timezone naive.")
                        pass  # Already naive, continue
                    except Exception as e:
                        logger.error(
                            f"Failed to localize index for {name} sentiment: {e}")
                        return None, None, 0.0
                    # --- END CORRECTION ---

                    sentiment_prep.index.name = 'ds'  # Ensure index name is 'ds' for merging
                    col_name = f'sentiment_{name}'
                    sentiment_prep.rename(
                        columns={'Daily_Sentiment': col_name}, inplace=True)

                    # Calculate last value *before* potentially returning None
                    last_val = 0.0
                    # Check if column exists and has valid data before accessing iloc[-1]
                    if col_name in sentiment_prep.columns and not sentiment_prep.empty and sentiment_prep[col_name].notna().any():
                        last_val = sentiment_prep[col_name].iloc[-1]
                        logger.info(
                            f"Sentiment regressor '{col_name}' prepared. Last value: {last_val:.3f}")
                        # Return the processed DataFrame column
                        return sentiment_prep, col_name, last_val
                    else:
                        logger.warning(
                            f"Sentiment data from {name} resulted in empty or all-NaN column after processing.")
                        return None, None, 0.0
                else:
                    logger.info(
                        f"No valid sentiment data provided for {name}.")
                    return None, None, 0.0

            # Process NewsAPI sentiment
            newsapi_sentiment_col_df, newsapi_col_name, last_sentiment_newsapi = process_sentiment(
                newsapi_sentiment_df, 'newsapi')
            if newsapi_sentiment_col_df is not None:
                # Merge directly using the original index
                prophet_df = pd.merge(
                    prophet_df, newsapi_sentiment_col_df, on='ds', how='left')
                # Fill NaNs introduced by merge (sentiment might not cover all historical price dates)
                prophet_df[newsapi_col_name] = prophet_df[newsapi_col_name].ffill()
                prophet_df[newsapi_col_name] = prophet_df[newsapi_col_name].fillna(
                    0.0)
                regressors_added.append(newsapi_col_name)
                last_sentiment_values[newsapi_col_name] = last_sentiment_newsapi

            # Process yfinance sentiment
            yfinance_sentiment_col_df, yfinance_col_name, last_sentiment_yfinance = process_sentiment(
                yfinance_sentiment_df, 'yfinance')
            if yfinance_sentiment_col_df is not None:
                # Merge onto the potentially updated prophet_df
                prophet_df = pd.merge(
                    prophet_df, yfinance_sentiment_col_df, on='ds', how='left')
                # Fill NaNs introduced by merge
                prophet_df[yfinance_col_name] = prophet_df[yfinance_col_name].ffill()
                prophet_df[yfinance_col_name] = prophet_df[yfinance_col_name].fillna(
                    0.0)
                regressors_added.append(yfinance_col_name)
                last_sentiment_values[yfinance_col_name] = last_sentiment_yfinance

            # --- Log Transform (Optional) ---
            use_log_transform = True
            if use_log_transform and 'y' in prophet_df and (prophet_df['y'] <= 0).any():
                logger.warning(
                    "Log transform skipped: Non-positive 'y' values found.")
                use_log_transform = False

            if use_log_transform and 'y' in prophet_df:
                logger.info("Applying log transform to 'y' data.")
                # Store original y if needed
                prophet_df['y_orig'] = prophet_df['y']
                # Clip to avoid log(0) or log(negative)
                prophet_df['y'] = np.log(prophet_df['y'].clip(lower=1e-9))

            # --- Initialize & Fit Model ---
            model = Prophet(
                daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
                interval_width=config.PROPHET_INTERVAL_WIDTH,
                changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR_SCALE
            )

            # Add all prepared regressors
            for regressor_name in regressors_added:
                # Check if column actually exists in prophet_df before adding
                if regressor_name in prophet_df.columns:
                    model.add_regressor(regressor_name)
                    logger.info(
                        f"Added regressor '{regressor_name}' to Prophet model.")
                else:
                    logger.warning(
                        f"Attempted to add regressor '{regressor_name}' but column not found in prophet_df.")

            # Ensure prophet_df only contains columns needed for fitting (ds, y, and added regressors)
            # Only use regs present in df
            cols_to_fit = [
                'ds', 'y'] + [reg for reg in regressors_added if reg in prophet_df.columns]
            prophet_df_fit = prophet_df[cols_to_fit].copy()
            # Drop rows where y is NaN (can happen after merges/processing if dates don't align perfectly)
            prophet_df_fit.dropna(subset=['y'], inplace=True)

            if prophet_df_fit.empty:
                logger.error(
                    "DataFrame became empty after preparing for Prophet fit (likely due to NaN handling or date mismatches).")
                st.error(
                    "Failed to generate forecast: Data processing error led to empty dataset.")
                return None

            model.fit(prophet_df_fit)

            # --- Predict Future ---
            future = model.make_future_dataframe(periods=days_to_predict)

            # Add future values for all regressors used
            for regressor_name in regressors_added:
                if regressor_name in model.extra_regressors:  # Check if regressor was actually added
                    future[regressor_name] = last_sentiment_values.get(
                        regressor_name, 0.0)
                    logger.debug(
                        f"Added future values for '{regressor_name}': {last_sentiment_values.get(regressor_name, 0.0):.3f}")

            forecast = model.predict(future)

            # --- Inverse Transform & Clip ---
            cols_to_transform = ['yhat', 'yhat_lower', 'yhat_upper']
            if use_log_transform:
                logger.info("Applying inverse log transform to forecast.")
                for col in cols_to_transform:
                    if col in forecast.columns:
                        # Use np.exp correctly
                        forecast[col] = np.exp(forecast[col])
            # Always clip predictions to be non-negative
            for col in cols_to_transform:
                if col in forecast.columns:
                    forecast[col] = forecast[col].clip(lower=0.0)

            logger.info(
                "Prophet forecast generated successfully (with dual sentiment).")
            # Return only the essential forecast columns
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        except Exception as e:
            logger.exception(f"Error during Prophet forecasting: {e}")
            st.error(f"Failed to generate Prophet forecast: {e}")
            return None
