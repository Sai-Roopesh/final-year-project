# src/analysis/ml_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Any, List

# Use relative imports within the src package
from .. import config
from .. import utils

logger = utils.logger


class MLPredictor:
    """Handles machine learning based price prediction."""

    @staticmethod
    # @st.cache_data # Removed caching here as it was causing issues with 'self' before refactor,
    # and results depend heavily on input data which might change subtly.
    # Consider adding caching back carefully if performance is an issue, potentially
    # hashing the input data explicitly or using st.cache_resource if models are large.
    def machine_learning_prediction(symbol: str, selected_models: List[str]) -> Optional[Dict[str, Any]]:
        """
        Trains selected ML models to predict the next day's PERCENTAGE RETURN.
        Returns a dictionary with results keyed by model name, using keys
        expected by the plotting function.
        """
        logger.info(
            f"Starting ML prediction for {symbol} using models: {selected_models}")
        if not selected_models:
            logger.warning("No ML models selected for prediction.")
            return None

        try:
            # 1. Fetch Data
            logger.info(
                f"Fetching {config.ML_TRAINING_PERIOD} data for ML training...")
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

            # 3. Feature Engineering
            logger.debug("Engineering ML features...")
            # Set Date as index for easier time-based operations
            data = data.set_index('Date')
            data['Returns'] = data['Close'].pct_change()
            data['SMA_20'] = data['Close'].rolling(
                window=config.SMA_PERIODS[0]).mean()
            data['SMA_50'] = data['Close'].rolling(
                window=config.SMA_PERIODS[1]).mean()
            data['Close_Lag1'] = data['Close'].shift(1)
            data['Volume_Lag1'] = data['Volume'].shift(1)
            # Use 'Returns' directly
            data['Pct_Change_Daily'] = data['Returns']
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
            for lag in [2, 3, 5]:  # Example lags
                data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
            rolling_window = 10  # Example window
            data[f'Volatility_{rolling_window}d'] = data['Returns'].rolling(
                window=rolling_window).std()
            data['DayOfWeek'] = data.index.dayofweek
            data['Month'] = data.index.month

            original_features = config.ML_FEATURE_DEFAULT
            engineered_features = [
                'SMA_20', 'SMA_50', 'Close_Lag1', 'Volume_Lag1', 'Pct_Change_Daily', 'RSI',
                'Close_Lag2', 'Close_Lag3', 'Close_Lag5', f'Volatility_{rolling_window}d',
                'DayOfWeek', 'Month'
            ]
            # Ensure only existing columns are selected
            features_present = list(
                set(original_features + [f for f in engineered_features if f in data.columns]))
            logger.debug(f"Features used for ML: {features_present}")

            # 4. Define Target (Percentage Return for next day)
            data['Target'] = data['Returns'].shift(-1)

            # 5. Clean Data (Drop rows with NaNs in features or target)
            # Reset index before dropna to keep Date column temporarily
            data_reset = data.reset_index()
            data_clean = data_reset.dropna(
                # Set Date back to index
                subset=features_present + ['Target']).set_index('Date')
            logger.debug(f"Data cleaned, {len(data_clean)} rows remaining.")
            if len(data_clean) < 60:  # Check after setting index back
                logger.warning(
                    f"Insufficient data ({len(data_clean)} rows) after cleaning for {symbol}.")
                st.warning(
                    "Not enough data for robust ML prediction after feature engineering.")
                return None

            # 6. Prepare X and y (using the cleaned data with DateTimeIndex)
            X = data_clean[features_present]
            y_returns = data_clean['Target']  # Target returns
            # Actual close prices corresponding to X/y (already indexed by Date)
            clean_closes = data_clean['Close']

            # 7. Split Data Chronologically
            test_size = config.ML_TEST_SIZE
            split_index = int(len(X) * (1 - test_size))

            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train_returns, y_test_returns = y_returns.iloc[:
                                                             split_index], y_returns.iloc[split_index:]

            # Get the actual prices for the test set (will have DateTimeIndex) -> KEY: y_test_actual_prices
            y_test_actual_prices_split = clean_closes.iloc[split_index:]
            # Get the last actual close price from the training set for reconstruction start
            last_train_actual_close = clean_closes.iloc[split_index - 1]

            # 8. Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            all_results = {}

            # --- Loop through selected models ---
            for model_name in selected_models:
                logger.info(f"Processing model: {model_name}")
                model = None
                # Use a dictionary for results for this specific model
                model_results = {}

                # 9. Instantiate Model
                if model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=config.RF_N_ESTIMATORS, random_state=42, n_jobs=-1, max_depth=config.RF_MAX_DEPTH,
                                                  min_samples_split=config.RF_MIN_SAMPLES_SPLIT, min_samples_leaf=config.RF_MIN_SAMPLES_LEAF)
                elif model_name == "Linear Regression":
                    model = LinearRegression(n_jobs=-1)
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                else:
                    logger.warning(
                        f"Model '{model_name}' not recognized. Skipping.")
                    continue

                if model is not None:
                    try:
                        # Train model to predict returns
                        model.fit(X_train_scaled, y_train_returns)

                        # 10. Make Predictions (Predicted Returns) & Evaluate
                        predicted_returns = model.predict(X_test_scaled)

                        # Check for NaNs/Infs in predicted returns before using them
                        if np.isnan(predicted_returns).any() or np.isinf(predicted_returns).any():
                            logger.warning(
                                f"{model_name}: NaNs or Infs found in predicted returns. Replacing with 0.")
                            predicted_returns = np.nan_to_num(
                                predicted_returns, nan=0.0, posinf=0.0, neginf=0.0)

                        # Evaluate based on returns -> KEY: rmse_returns
                        rmse = np.sqrt(mean_squared_error(
                            y_test_returns, predicted_returns))
                        model_results['rmse_returns'] = rmse
                        logger.info(
                            f"{model_name} - Test RMSE (Returns): {rmse:.6f}")

                        # Store test dates (which is the index of y_test_actual_prices_split) -> KEY: test_dates
                        # Use the index directly
                        model_results['test_dates'] = y_test_actual_prices_split.index
                        # Store actual prices for the test set (already has index) -> KEY: y_test_actual_prices
                        model_results['y_test_actual_prices'] = y_test_actual_prices_split

                        # --- FIX: Reconstruct Predicted Prices as Series ---
                        # Get the actual prices of the *previous* day relative to the test set dates
                        # The index of X_test holds the dates for which we are predicting the return
                        # We need the actual close price from the day *before* each X_test date
                        # The `clean_closes` Series holds all actual prices with DateTimeIndex
                        # We can get the previous day's close by shifting `clean_closes` and selecting the dates matching X_test.index
                        previous_actual_prices = clean_closes.shift(
                            1).loc[X_test.index].fillna(last_train_actual_close)

                        # Calculate the predicted prices based on the previous day's actual price and the predicted return
                        predicted_prices_values = previous_actual_prices.values * \
                            (1 + predicted_returns)

                        # Create a Pandas Series for predicted prices with the correct test dates index
                        predictions_prices_series = pd.Series(
                            predicted_prices_values, index=X_test.index, name=f"{model_name}_Predicted")

                        # Store the Series -> KEY: predictions_prices
                        model_results['predictions_prices'] = predictions_prices_series
                        # --- End FIX ---

                        # 11. Feature Importance -> KEY: feature_importance
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
                        # Use the last row of the *original cleaned data* before splitting
                        # Last row of features used in training/testing
                        last_row_features_unscaled = X.iloc[-1:]
                        # Last known actual close price
                        last_actual_close_final = clean_closes.iloc[-1]
                        # Store last actual close -> KEY: last_actual_close
                        model_results['last_actual_close'] = last_actual_close_final

                        next_day_predicted_return = None
                        next_day_predicted_price = None

                        if not last_row_features_unscaled.isnull().any().any():
                            last_scaled = scaler.transform(
                                last_row_features_unscaled.values)
                            next_day_predicted_return = model.predict(last_scaled)[
                                0]
                            # Calculate next price based on the very last actual price
                            next_day_predicted_price = last_actual_close_final * \
                                (1 + next_day_predicted_return)
                        else:
                            logger.warning(
                                f"{model_name} - NaNs in last feature row, cannot make next day prediction.")

                        # Store next day predictions -> KEYS: next_day_prediction_return, next_day_prediction_price
                        model_results['next_day_prediction_return'] = next_day_predicted_return
                        # Corrected variable name
                        model_results['next_day_prediction_price'] = next_day_predicted_price

                        if next_day_predicted_price is not None:
                            logger.info(
                                f"{model_name} - Predicted next day return: {next_day_predicted_return:.6f}")
                            logger.info(
                                f"{model_name} - Predicted next day price: {next_day_predicted_price:.4f}")

                        # Store features used -> KEY: features_used
                        model_results['features_used'] = features_present
                        all_results[model_name] = model_results

                    except Exception as model_err:
                        logger.error(
                            f"Error processing model {model_name}: {model_err}", exc_info=True)
                        # Store error information for this model
                        all_results[model_name] = {"error": str(model_err)}

            logger.info("ML prediction completed for selected models.")
            return all_results

        except Exception as e:
            logger.exception(
                f"Error during ML prediction process for {symbol}: {e}")
            st.error(f"Error during Machine Learning prediction: {e}")
            return None
