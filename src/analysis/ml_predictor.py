# src/analysis/ml_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
# import yfinance as yf # No longer needed here
# Use TimeSeriesSplit later if needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Any, List

# Use relative imports within the src package
from .. import config
from .. import utils

logger = utils.logger


class MLPredictor:
    """Handles machine learning based price prediction."""

    # Removed @staticmethod - it's now an instance method (though doesn't use self yet)
    # Added 'stock_hist_data' argument
    def machine_learning_prediction(self, symbol: str, selected_models: List[str], stock_hist_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Trains selected ML models to predict the next day's PERCENTAGE RETURN.
        Uses the provided historical stock data.

        Args:
            symbol (str): The stock symbol (for logging/context).
            selected_models (List[str]): List of model names to train (e.g., ['Random Forest']).
            stock_hist_data (pd.DataFrame): DataFrame containing historical stock data
                                            (must include 'Date', 'Close', and features).

        Returns:
            Optional[Dict[str, Any]]: Dictionary with results keyed by model name,
                                      or None if prediction fails.
        """
        logger.info(
            f"Starting ML prediction for {symbol} using models: {selected_models}")
        if not selected_models:
            logger.warning("No ML models selected for prediction.")
            return None
        if stock_hist_data is None or stock_hist_data.empty:
            logger.error(
                f"ML prediction requires historical data, but none was provided for {symbol}.")
            st.error(
                f"ML Prediction Error: Missing historical data for {symbol}.")
            return None

        try:
            # 1. Use Provided Data (Make a copy to avoid modifying original)
            data = stock_hist_data.copy()
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(
                    f"Provided stock_hist_data is missing required columns for ML. Need: {required_cols}, Got: {data.columns}")
                st.error(
                    "ML Prediction Error: Input data missing required columns.")
                return None

            # 2. Feature Engineering (using the provided data)
            logger.debug("Engineering ML features...")
            # Ensure 'Date' is the index for time-based operations
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
            elif not isinstance(data.index, pd.DatetimeIndex):
                logger.error(
                    "ML Prediction Error: Data must have a 'Date' column or DatetimeIndex.")
                st.error("ML Prediction Error: Data index is not time-based.")
                return None
            else:  # Ensure index is timezone-naive if it's already DatetimeIndex
                data.index = data.index.tz_localize(None)

            # --- Feature Calculations ---
            data['Returns'] = data['Close'].pct_change()
            # Calculate SMAs using config periods
            for period in config.SMA_PERIODS:
                data[f'SMA_{period}'] = data['Close'].rolling(
                    window=period).mean()
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(com=config.RSI_PERIOD - 1,
                                min_periods=config.RSI_PERIOD).mean()
            avg_loss = loss.ewm(com=config.RSI_PERIOD - 1,
                                min_periods=config.RSI_PERIOD).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            data['RSI'] = 100.0 - (100.0 / (1.0 + rs))
            data['RSI'] = data['RSI'].fillna(50)
            # Lagged Features
            for lag in [1, 2, 3, 5]:
                data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag{lag}'] = data['Volume'].shift(
                    lag)  # Add lagged volume
                data[f'Return_Lag{lag}'] = data['Returns'].shift(
                    lag)  # Add lagged returns
            # Volatility
            data[f'Volatility_{config.VOLATILITY_WINDOW}d'] = data['Returns'].rolling(
                window=config.VOLATILITY_WINDOW).std()
            # Time Features
            data['DayOfWeek'] = data.index.dayofweek
            data['Month'] = data.index.month
            data['WeekOfYear'] = data.index.isocalendar().week.astype(int)
            data['DayOfYear'] = data.index.dayofyear
            # --- End Feature Calculations ---

            # Define features to use
            # ['Open', 'High', 'Low', 'Close', 'Volume']
            original_features = config.ML_FEATURE_DEFAULT
            engineered_features = [
                # Include all SMAs
                f'SMA_{config.SMA_PERIODS[0]}', f'SMA_{config.SMA_PERIODS[1]}', f'SMA_{config.SMA_PERIODS[2]}',
                'RSI',
                'Close_Lag1', 'Volume_Lag1', 'Return_Lag1',
                'Close_Lag2', 'Volume_Lag2', 'Return_Lag2',
                'Close_Lag3', 'Volume_Lag3', 'Return_Lag3',
                'Close_Lag5', 'Volume_Lag5', 'Return_Lag5',
                f'Volatility_{config.VOLATILITY_WINDOW}d',
                'DayOfWeek', 'Month', 'WeekOfYear', 'DayOfYear'
            ]
            # Combine and ensure only existing columns are selected
            all_feature_candidates = list(
                set(original_features + engineered_features))
            features_present = [
                f for f in all_feature_candidates if f in data.columns]
            logger.debug(f"Features used for ML: {features_present}")

            # 3. Define Target (Percentage Return for next day)
            data['Target'] = data['Returns'].shift(-1)

            # 4. Clean Data (Drop rows with NaNs in features or target)
            data_clean = data.dropna(subset=features_present + ['Target'])
            logger.debug(f"Data cleaned, {len(data_clean)} rows remaining.")

            # Need sufficient data after cleaning and for train/test split
            min_rows_needed = 20  # Adjust as needed
            if len(data_clean) < min_rows_needed:
                logger.warning(
                    f"Insufficient data ({len(data_clean)} rows) after cleaning for {symbol}. Need at least {min_rows_needed}.")
                st.warning(
                    f"Not enough data for robust ML prediction after feature engineering ({len(data_clean)} rows).")
                return None

            # 5. Prepare X and y (using the cleaned data with DateTimeIndex)
            X = data_clean[features_present]
            y_returns = data_clean['Target']  # Target returns
            # Actual close prices corresponding to X/y
            clean_closes = data_clean['Close']

            # 6. Split Data Chronologically
            test_size = config.ML_TEST_SIZE
            split_index = int(len(X) * (1 - test_size))

            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train_returns, y_test_returns = y_returns.iloc[:
                                                             split_index], y_returns.iloc[split_index:]

            # Get actual prices for the test set and the last training price
            y_test_actual_prices_split = clean_closes.iloc[split_index:]
            # Handle edge case
            last_train_actual_close = clean_closes.iloc[split_index -
                                                        1] if split_index > 0 else clean_closes.iloc[0]

            # 7. Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            all_results = {}

            # --- Loop through selected models ---
            for model_name in selected_models:
                logger.info(f"Processing model: {model_name}")
                model = None
                model_results = {}  # Dictionary for this model's results

                # 8. Instantiate Model
                try:
                    if model_name == "Random Forest":
                        model = RandomForestRegressor(
                            n_estimators=config.RF_N_ESTIMATORS, random_state=42, n_jobs=-1,
                            max_depth=config.RF_MAX_DEPTH, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF
                        )
                    elif model_name == "Linear Regression":
                        model = LinearRegression(n_jobs=-1)
                    elif model_name == "Ridge Regression":
                        model = RidgeCV(
                            alphas=np.logspace(-6, 6, 13), store_cv_values=True)
                    elif model_name == "Lasso Regression":
                        model = LassoCV(
                            alphas=np.logspace(-6, 6, 13), cv=5, n_jobs=-1)
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(
                            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
                        )
                    else:
                        logger.warning(
                            f"Model '{model_name}' not recognized. Skipping.")
                        all_results[model_name] = {
                            "error": f"Model '{model_name}' not recognized."}
                        continue

                    # Train model to predict returns
                    model.fit(X_train_scaled, y_train_returns)

                    # 9. Make Predictions (Predicted Returns) & Evaluate
                    predicted_returns = model.predict(X_test_scaled)
                    predicted_returns = np.nan_to_num(
                        predicted_returns, nan=0.0, posinf=0.0, neginf=0.0)  # Sanitize

                    # Evaluate based on returns
                    rmse = np.sqrt(mean_squared_error(
                        y_test_returns, predicted_returns))
                    model_results['rmse_returns'] = rmse
                    logger.info(
                        f"{model_name} - Test RMSE (Returns): {rmse:.6f}")

                    # Store test dates and actual prices
                    model_results['test_dates'] = y_test_actual_prices_split.index
                    model_results['y_test_actual_prices'] = y_test_actual_prices_split

                    # 10. Reconstruct Predicted Prices as Series
                    previous_actual_prices = clean_closes.shift(
                        1).loc[X_test.index].fillna(last_train_actual_close)
                    predicted_prices_values = previous_actual_prices.values * \
                        (1 + predicted_returns)
                    predictions_prices_series = pd.Series(
                        predicted_prices_values, index=X_test.index, name=f"{model_name}_Predicted")
                    model_results['predictions_prices'] = predictions_prices_series

                    # 11. Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(
                            'Importance', ascending=False).reset_index(drop=True)
                        model_results['feature_importance'] = feature_importance
                    elif hasattr(model, 'coef_'):
                        feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_}).sort_values(
                            'Coefficient', key=abs, ascending=False).reset_index(drop=True)
                        model_results['feature_importance'] = feature_importance
                    else:
                        model_results['feature_importance'] = None

                    # 12. Predict Next Day's Return and Price
                    last_row_features_unscaled = X.iloc[-1:]
                    last_actual_close_final = clean_closes.iloc[-1]
                    model_results['last_actual_close'] = last_actual_close_final

                    next_day_predicted_return = None
                    next_day_predicted_price = None

                    if not last_row_features_unscaled.isnull().any().any():
                        last_scaled = scaler.transform(
                            last_row_features_unscaled.values)
                        next_day_predicted_return = model.predict(last_scaled)[
                            0]
                        next_day_predicted_price = last_actual_close_final * \
                            (1 + next_day_predicted_return)
                        logger.info(
                            f"{model_name} - Predicted next day return: {next_day_predicted_return:.6f}")
                        logger.info(
                            f"{model_name} - Predicted next day price: {next_day_predicted_price:.4f}")
                    else:
                        logger.warning(
                            f"{model_name} - NaNs in last feature row, cannot make next day prediction.")

                    model_results['next_day_prediction_return'] = next_day_predicted_return
                    # Corrected key name
                    model_results['next_day_prediction_price'] = next_day_predicted_price
                    model_results['features_used'] = features_present

                    # Add results for this model
                    all_results[model_name] = model_results

                except Exception as model_err:
                    logger.error(
                        f"Error processing model {model_name}: {model_err}", exc_info=True)
                    all_results[model_name] = {
                        "error": str(model_err)}  # Store error

            logger.info("ML prediction completed for selected models.")
            # Return None if no models succeeded
            return all_results if all_results else None

        except Exception as e:
            logger.exception(
                f"Error during ML prediction process for {symbol}: {e}")
            st.error(f"Error during Machine Learning prediction: {e}")
            return None
