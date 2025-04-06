# src/analysis/portfolio.py

import streamlit as st  # Keep for cache decorator
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import yfinance as yf
from .. import config
from .. import utils
# Import DataFetcher class to use its methods
from .data_fetcher import DataFetcher

logger = utils.logger


class Portfolio:
    """Handles portfolio simulation and correlation analysis."""

    def __init__(self, data_fetcher: DataFetcher):
        """Requires a DataFetcher instance to get underlying data."""
        self.data_fetcher = data_fetcher

    # --- Helper moved from DataFetcher ---
    # This helper is specific to portfolio/correlation needs
    def _fetch_portfolio_close_data(self, symbols: List[str], period: str = '1y') -> Optional[pd.DataFrame]:
        """ Fetches adjusted close data for multiple symbols into a single DataFrame."""
        logger.info(
            f"Fetching portfolio close data for: {symbols} over period: {period}")
        if not symbols:
            return None

        try:
            # Use yfinance download directly for simplicity here
            # auto_adjust handles splits/dividends
            all_data = yf.download(
                symbols, period=period, auto_adjust=True, progress=False)
            if all_data.empty:
                logger.warning(f"No portfolio data returned for: {symbols}")
                return None

            # Extract 'Close' prices
            if 'Close' in all_data.columns:
                close_data = all_data['Close']
                # For single symbol download, result might not be MultiIndex
                if isinstance(close_data, pd.Series):
                    # Convert Series to DataFrame
                    close_data = pd.DataFrame({symbols[0]: close_data})
                # Drop columns that are all NaN (e.g., for symbols with no data)
                close_data = close_data.dropna(axis=1, how='all')
                if close_data.empty:
                    logger.warning(
                        f"All symbols had NaN close data for period {period}.")
                    return None
                return close_data
            else:
                logger.error(
                    f"Could not extract 'Close' data columns for portfolio: {symbols}.")
                return None

        except Exception as e:
            logger.error(
                f"Error fetching portfolio close data for {symbols}: {e}", exc_info=True)
            st.error(f"Error fetching data for portfolio/correlation analysis.")
            return None

    @st.cache_data(show_spinner="Simulating portfolio performance...")
    def portfolio_simulation(self, symbols: List[str], initial_investment: float, investment_strategy: str = 'equal_weight') -> Optional[Dict[str, Any]]:
        """Runs a portfolio simulation based on historical data and strategy."""
        logger.info(
            f"Starting portfolio simulation for {symbols} using strategy '{investment_strategy}'.")

        # 1. Fetch Close Data (using internal helper)
        portfolio_df = self._fetch_portfolio_close_data(
            symbols, period='1y')  # Use 1 year lookback

        if portfolio_df is None or portfolio_df.empty:
            st.error(
                "No valid historical data found for any portfolio symbols. Cannot run simulation.")
            return None

        actual_symbols_used = list(portfolio_df.columns)
        if len(actual_symbols_used) < len(symbols):
            missing = set(symbols) - set(actual_symbols_used)
            st.warning(
                f"Excluded symbols with missing data: {', '.join(missing)}. Simulating with: {', '.join(actual_symbols_used)}.")
            logger.warning(
                f"Portfolio simulation excluded: {', '.join(missing)}.")
        if not actual_symbols_used:  # Should be caught above, but double-check
            st.error("No symbols remain. Cannot simulate.")
            return None

        # 2. Calculate Returns
        # Ensure sufficient non-NaN data for pct_change
        portfolio_df = portfolio_df.dropna()  # Drop rows where any stock has NaN
        if len(portfolio_df) < 2:
            st.error(
                "Insufficient overlapping historical data for portfolio calculations.")
            return None
        # Drop first NaN row from pct_change
        portfolio_returns = portfolio_df.pct_change().dropna()
        if portfolio_returns.empty:
            st.error(
                "Could not calculate returns (maybe only 1 day of overlapping data).")
            return None

        # 3. Determine Weights
        num_stocks = len(actual_symbols_used)
        weights = np.ones(num_stocks) / num_stocks  # Default: Equal weight

        if investment_strategy == 'market_cap_weighted':
            logger.info("Calculating Market Cap weights.")
            market_caps = {}
            missing_caps = []
            with st.spinner("Fetching market caps for weighting..."):
                for stock in actual_symbols_used:
                    try:
                        # Use cached info fetcher method
                        stock_info = self.data_fetcher.load_stock_info(stock)
                        cap = stock_info.get(
                            'marketCap') if stock_info else None
                        if cap and isinstance(cap, (int, float)) and cap > 0:
                            market_caps[stock] = cap
                        else:
                            logger.warning(
                                f"Missing/invalid marketCap for {stock}. Will affect weighting.")
                            missing_caps.append(stock)
                    except Exception as e:
                        logger.warning(
                            f"Error fetching market cap for {stock}: {e}. Affects weighting.")
                        missing_caps.append(stock)

            if missing_caps:
                st.warning(
                    f"Market cap data missing/invalid for: {', '.join(missing_caps)}. Falling back to EQUAL WEIGHT strategy.")
                logger.warning(
                    f"Market cap missing for {missing_caps}, falling back to equal weight.")
                # Keep default equal weights
            elif market_caps:  # All caps found
                total_cap = sum(market_caps.values())
                weights_dict = {stock: cap / total_cap for stock,
                                cap in market_caps.items()}
                # Ensure weights align with portfolio_returns columns order
                weights = np.array([weights_dict.get(s, 0)
                                   for s in actual_symbols_used])
                weights = weights / weights.sum()  # Normalize again just in case
                logger.info("Using market cap weighted strategy.")
            else:  # No valid caps found at all
                st.warning(
                    "Could not retrieve valid market caps. Falling back to EQUAL WEIGHT strategy.")
                logger.warning(
                    "All market caps missing, falling back to equal weight.")
                # Keep default equal weights

        logger.info(
            f"Final portfolio weights: {dict(zip(actual_symbols_used, weights))}")

        # 4. Calculate Portfolio Metrics
        daily_portfolio_returns = (portfolio_returns * weights).sum(axis=1)
        cumulative_returns = (1 + daily_portfolio_returns).cumprod()
        cumulative_value = cumulative_returns * initial_investment
        # Ensure start value is exactly initial investment
        if not cumulative_value.empty:
            cumulative_value.iloc[0] = initial_investment

        # Calculate annualized metrics (handle potential insufficient data)
        if len(daily_portfolio_returns) > 1:
            trading_days = 252
            annual_volatility = daily_portfolio_returns.std() * np.sqrt(trading_days)
            # Calculate cumulative return for the period
            total_period_return = cumulative_returns.iloc[-1] - 1
            # Annualize return (approximation for periods != 1 year)
            num_years = len(daily_portfolio_returns) / trading_days
            annual_return = ((1 + total_period_return) **
                             (1/num_years)) - 1 if num_years > 0 else 0.0
            # Sharpe Ratio (assuming risk-free rate = 0)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0.0
        else:
            logger.warning(
                "Not enough return data points (<2) for annualized metrics.")
            annual_volatility, annual_return, sharpe_ratio = 0.0, 0.0, 0.0

        logger.info("Portfolio simulation calculations complete.")
        return {
            'daily_returns': daily_portfolio_returns,
            'cumulative_value': cumulative_value,
            'annualized_volatility': annual_volatility,
            'annualized_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'weights': dict(zip(actual_symbols_used, weights)),
            'symbols_used': actual_symbols_used
        }

    @st.cache_data(show_spinner="Calculating correlation matrix...")
    def advanced_correlation_analysis(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Calculates the correlation matrix of daily returns for given symbols."""
        logger.info(f"Starting correlation analysis for {symbols}.")

        # 1. Fetch Close Data
        portfolio_df = self._fetch_portfolio_close_data(
            symbols, period='1y')  # Use 1 year lookback

        if portfolio_df is None or portfolio_df.empty:
            st.error("No valid historical data found for correlation analysis.")
            return None

        actual_symbols_used = list(portfolio_df.columns)
        if len(actual_symbols_used) < 2:
            st.warning(
                f"Correlation analysis requires at least two stocks with valid data. Only found: {', '.join(actual_symbols_used)}")
            return None
        if len(actual_symbols_used) < len(symbols):
            missing = set(symbols) - set(actual_symbols_used)
            st.warning(f"Correlation excluded symbols: {', '.join(missing)}.")

        # 2. Calculate Returns and Correlation
        portfolio_df = portfolio_df.dropna()
        if len(portfolio_df) < 2:
            st.error("Insufficient overlapping data for correlation calculation.")
            return None
        returns_df = portfolio_df.pct_change().dropna()
        if len(returns_df) < 2:  # Need >1 row after pct_change for correlation
            st.error(
                "Insufficient returns data points (<2) to calculate correlations.")
            return None

        correlation_matrix = returns_df.corr()
        logger.info("Correlation matrix computed successfully.")
        return correlation_matrix
