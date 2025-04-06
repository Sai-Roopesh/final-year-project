# src/analysis/technical.py

import pandas as pd
import numpy as np
from typing import List, Optional

from src import config, utils

logger = utils.logger


class TechnicalAnalysis:
    """Handles calculation of technical indicators and pattern analysis."""

    @staticmethod  # Make static as it doesn't depend on instance state
    # @st.cache_data # Caching might be better handled on the raw data fetch
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates various technical indicators using config parameters."""
        logger.info("Calculating technical indicators.")
        if df is None or df.empty:
            logger.warning(
                "Input DataFrame is empty. Cannot calculate indicators.")
            return pd.DataFrame()
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(
                f"Missing required columns for indicators. Got: {df.columns}")
            return df  # Return original if columns missing

        df_tech = df.copy()

        # SMAs
        for window in config.SMA_PERIODS:
            if len(df_tech) >= window:
                df_tech[f'SMA_{window}'] = df_tech['Close'].rolling(
                    window=window).mean()
            else:
                df_tech[f'SMA_{window}'] = np.nan

        # EMAs (for MACD)
        df_tech['EMA_Fast'] = df_tech['Close'].ewm(
            span=config.MACD_FAST, adjust=False).mean()
        df_tech['EMA_Slow'] = df_tech['Close'].ewm(
            span=config.MACD_SLOW, adjust=False).mean()

        # RSI
        delta = df_tech['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=config.RSI_PERIOD - 1,
                            min_periods=config.RSI_PERIOD).mean()
        avg_loss = loss.ewm(com=config.RSI_PERIOD - 1,
                            min_periods=config.RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_tech['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        df_tech['RSI'] = df_tech['RSI'].fillna(50)  # Fill initial NaNs

        # MACD
        df_tech['MACD'] = df_tech['EMA_Fast'] - df_tech['EMA_Slow']
        df_tech['Signal_Line'] = df_tech['MACD'].ewm(
            span=config.MACD_SIGNAL, adjust=False).mean()
        df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['Signal_Line']
        df_tech.drop(columns=['EMA_Fast', 'EMA_Slow'],
                     inplace=True)  # Clean up intermediate EMAs

        # Bollinger Bands
        bb_window = config.BB_WINDOW
        if len(df_tech) >= bb_window:
            df_tech['BB_Middle'] = df_tech['Close'].rolling(
                window=bb_window).mean()
            bb_std = df_tech['Close'].rolling(window=bb_window).std()
            df_tech['BB_Upper'] = df_tech['BB_Middle'] + \
                config.BB_STD_DEV * bb_std
            df_tech['BB_Lower'] = df_tech['BB_Middle'] - \
                config.BB_STD_DEV * bb_std
        else:
            df_tech[['BB_Middle', 'BB_Upper', 'BB_Lower']] = np.nan

        # Volatility (Annualized)
        vol_window = config.VOLATILITY_WINDOW
        if len(df_tech) >= vol_window:
            # Calculate daily returns first for volatility
            df_tech['Daily_Return'] = df_tech['Close'].pct_change()
            df_tech[f'Volatility_{vol_window}'] = df_tech['Daily_Return'].rolling(
                window=vol_window).std() * np.sqrt(252)  # Annualize
            df_tech.drop(columns=['Daily_Return'], inplace=True)  # Clean up
        else:
            df_tech[f'Volatility_{vol_window}'] = np.nan

        # ATR
        atr_window = config.ATR_WINDOW
        if len(df_tech) >= atr_window:
            high_low = df_tech['High'] - df_tech['Low']
            high_close = np.abs(df_tech['High'] - df_tech['Close'].shift())
            low_close = np.abs(df_tech['Low'] - df_tech['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df_tech[f'ATR_{atr_window}'] = true_range.ewm(
                alpha=1/atr_window, adjust=False).mean()
        else:
            df_tech[f'ATR_{atr_window}'] = np.nan

        logger.info("Technical indicators calculated.")
        return df_tech

    @staticmethod  # Make static as it doesn't depend on instance state
    def analyze_patterns(df: pd.DataFrame) -> List[str]:
        """Analyzes technical patterns based on calculated indicators."""
        logger.info("Analyzing technical patterns.")
        patterns = []
        # Need enough data for longest SMA
        min_rows_for_patterns = max(config.SMA_PERIODS)
        if df is None or df.empty or len(df) < min_rows_for_patterns:
            logger.warning(
                f"Insufficient data ({len(df) if df is not None else 0} rows) for pattern analysis (need {min_rows_for_patterns}).")
            return ["Not enough data for pattern analysis."]

        required_cols = ['Close', f'SMA_{config.SMA_PERIODS[0]}', f'SMA_{config.SMA_PERIODS[1]}',
                         'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']
        latest = df.iloc[-1]
        # Need previous row for crossovers
        previous = df.iloc[-2] if len(df) > 1 else latest

        # Check latest data availability
        missing_latest = [
            col for col in required_cols if pd.isna(latest.get(col))]
        if missing_latest:
            logger.warning(
                f"Latest data missing for indicator columns: {missing_latest}. Patterns might be incomplete.")
            # patterns.append("Warning: Recent indicator data missing.") # Optional warning

        # Use first two SMAs for cross
        sma_short, sma_long = f'SMA_{config.SMA_PERIODS[0]}', f'SMA_{config.SMA_PERIODS[1]}'
        # SMA Golden/Death Cross
        if pd.notna(latest.get(sma_short)) and pd.notna(latest.get(sma_long)) and \
           pd.notna(previous.get(sma_short)) and pd.notna(previous.get(sma_long)):
            if latest[sma_short] > latest[sma_long] and previous[sma_short] <= previous[sma_long]:
                patterns.append(
                    f"Golden Cross ({sma_short} / {sma_long}) - Potential Bullish Signal")
            elif latest[sma_short] < latest[sma_long] and previous[sma_short] >= previous[sma_long]:
                patterns.append(
                    f"Death Cross ({sma_short} / {sma_long}) - Potential Bearish Signal")

        # RSI Overbought/Oversold
        if pd.notna(latest.get('RSI')):
            if latest['RSI'] > 70:
                patterns.append(f"Overbought (RSI = {latest['RSI']:.2f})")
            elif latest['RSI'] < 30:
                patterns.append(f"Oversold (RSI = {latest['RSI']:.2f})")

        # MACD Crossover
        if pd.notna(latest.get('MACD')) and pd.notna(latest.get('Signal_Line')) and \
           pd.notna(previous.get('MACD')) and pd.notna(previous.get('Signal_Line')):
            if latest['MACD'] > latest['Signal_Line'] and previous['MACD'] <= previous['Signal_Line']:
                patterns.append("MACD Bullish Crossover")
            elif latest['MACD'] < latest['Signal_Line'] and previous['MACD'] >= previous['Signal_Line']:
                patterns.append("MACD Bearish Crossover")

        # Bollinger Bands Breach
        if pd.notna(latest.get('Close')) and pd.notna(latest.get('BB_Upper')) and pd.notna(latest.get('BB_Lower')):
            if latest['Close'] > latest['BB_Upper']:
                patterns.append("Price above Upper Bollinger Band")
            elif latest['Close'] < latest['BB_Lower']:
                patterns.append("Price below Lower Bollinger Band")

        # Add more pattern checks here if needed (e.g., volume spikes, candlestick patterns)

        if not patterns:
            patterns.append(
                "No significant standard patterns detected recently.")

        logger.info(f"Patterns identified: {len(patterns)}")
        return patterns
