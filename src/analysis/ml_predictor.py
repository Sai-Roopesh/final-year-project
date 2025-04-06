# src/analysis/ml_predictor.py

import streamlit as st  # Keep for cache decorator
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error  # For evaluation
from typing import Optional, Dict, Any, List

from .. import config
from .. import utils  # For logger and normalize_yfinance_data

logger = utils.logger


class MLPredictor:
    """Handles machine learning based price prediction."""

    @staticmethod
    @st.cache_data(show_spinner="Training ML model and predicting...")
    def machine_learning_prediction(symbol: str) -> Optional[Dict[str, Any]]:
        """
        Trains a Random Forest model on historical data and predicts the next day's close.
        Uses configuration for parameters and training period.
        """
        logger.info(f"Starting ML prediction for {symbol}.")
        try:
            # 1. Fetch Data (Longer period for training)
            logger.info(
                f"Fetching {config.ML_TRAINING_PERIOD} data for ML training...")
            # Use yf.download directly or reuse DataFetcher if it supports periods well
            raw_data = yf.download(
                symbol, period=config.ML_TRAINING_PERIOD, auto_adjust=False, progress=False)

            if raw_data.empty:
                logger.warning(
                    f"No data downloaded for ML training for {symbol}.")
                st.warning(
                    f"Could not download data for {symbol} for ML prediction.")
                return None

            # 2. Normalize Data
            data = utils.normalize_yfinance_data(raw_data, symbol)
            if data is None:
                logger.error(f"Data normalization failed for ML for {symbol}.")
                st.error(
                    f"Data processing error for ML prediction on {symbol}.")
                return None
            data.set_index('Date', inplace=True)  # Use Date as index

            # 3. Feature Engineering (Add more features as needed)
            logger.debug("Engineering ML features...")
            data['SMA_20'] = data['Close'].rolling(
                window=config.SMA_PERIODS[0]).mean()
            data['SMA_50'] = data['Close'].rolling(
                window=config.SMA_PERIODS[1]).mean()
            data['Close_Lag1'] = data['Close'].shift(1)
            data['Volume_Lag1'] = data['Volume'].shift(1)
            data['Pct_Change'] = data['Close'].pct_change()
            # Example: Add RSI as feature
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(com=config.RSI_PERIOD - 1,
                                min_periods=config.RSI_PERIOD).mean()
            avg_loss = loss.ewm(com=config.RSI_PERIOD - 1,
                                min_periods=config.RSI_PERIOD).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            data['RSI'] = 100.0 - (100.0 / (1.0 + rs))
            data['RSI'].fillna(50, inplace=True)

            # Define features to use
            engineered_features = [
                'SMA_20', 'SMA_50', 'Close_Lag1', 'Volume_Lag1', 'Pct_Change', 'RSI']
            features_present = config.ML_FEATURE_DEFAULT + \
                [f for f in engineered_features if f in data.columns]
            # Ensure unique features
            features_present = list(set(features_present))

            # 4. Define Target
            # Predict next day's close
            data['Target'] = data['Close'].shift(-1)

            # 5. Clean Data (Drop NaNs after feature engineering/target shift)
            data_clean = data.dropna()
            logger.debug(f"Data cleaned, {len(data_clean)} rows remaining.")

            if len(data_clean) < 60:  # Need sufficient data after cleaning
                logger.warning(
                    f"Insufficient data ({len(data_clean)} rows) after cleaning for {symbol}.")
                st.warning(
                    "Not enough data for robust ML prediction after feature engineering.")
                return None

            # 6. Prepare X and y
            X = data_clean[features_present]
            y = data_clean['Target']
            original_index = data_clean.index  # Keep dates for plotting test results

            # 7. Split Data (Chronological)
            test_size = config.ML_TEST_SIZE
            split_index = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            test_dates = original_index[split_index:]

            # 8. Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 9. Train Model (Random Forest from config)
            logger.info(f"Training RandomForest model for {symbol}...")
            rf_model = RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS, random_state=42, n_jobs=-1,
                max_depth=config.RF_MAX_DEPTH, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF
            )
            rf_model.fit(X_train_scaled, y_train)

            # 10. Make Predictions & Evaluate
            predictions = rf_model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            # mae = mean_absolute_error(y_test, predictions) # Optional: MAE
            logger.info(f"Model trained. Test RMSE: {rmse:.4f}")

            # 11. Feature Importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,  # Use columns from X
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)

            # 12. Predict Next Day
            # Use the *last* row of the *original* data before dropna for prediction features
            last_row_features_unscaled = data[features_present].iloc[-1:]
            next_day_prediction = None
            if not last_row_features_unscaled.isnull().any().any():
                last_scaled = scaler.transform(
                    last_row_features_unscaled.values)
                next_day_prediction = rf_model.predict(last_scaled)[0]
                logger.info(
                    f"Predicted next day close for {symbol}: {next_day_prediction:.4f}")
            else:
                logger.warning(
                    f"NaNs in last feature row for {symbol}, cannot make next day prediction.")

            results = {
                'test_dates': test_dates,
                'y_test': y_test,
                'predictions': predictions,
                'rmse': rmse,
                # 'mae': mae,
                'feature_importance': feature_importance,
                'next_day_prediction': next_day_prediction,
                # Last available actual close
                'last_actual_close': data['Close'].iloc[-1],
                'features_used': features_present
            }
            logger.info("ML prediction completed successfully.")
            return results

        except Exception as e:
            logger.exception(f"Error during ML prediction for {symbol}: {e}")
            st.error(f"Error during Machine Learning prediction: {e}")
            return None
