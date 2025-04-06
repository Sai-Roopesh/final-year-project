# src/plotting.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import yfinance as yf
from datetime import datetime  # Ensure datetime is imported
from datetime import timedelta
# Use relative imports within the src package
from . import config
from . import utils
logger = utils.logger
format_value = utils.format_value


# --- Analyst Recommendations Display ---
def display_analyst_recommendations(analyst_info: Optional[Dict[str, Any]]):
    """Displays analyst recommendations and target price data."""
    logger.info("Displaying analyst recommendations.")
    st.subheader("Analyst Ratings & Price Targets")

    if not analyst_info:
        st.info("Analyst recommendation data not available for this stock.")
        return

    targets = analyst_info.get('targets', {})
    current_rec = analyst_info.get('current_recommendation', 'N/A')
    num_analysts = analyst_info.get('num_analysts')
    history = analyst_info.get('history')  # This should be a DataFrame

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Consensus**")
        rec_display = current_rec.replace('_', ' ').title(
        ) if isinstance(current_rec, str) else 'N/A'
        st.metric("Recommendation", rec_display,
                  help="The consensus recommendation key from analysts (e.g., buy, hold, sell). Source: yfinance.")
        if num_analysts is not None:
            st.caption(f"Based on {num_analysts} analyst opinions.")
        else:
            st.caption("Number of analysts not available.")

    with col2:
        st.markdown("**Price Targets**")
        if targets:
            tgt_col1, tgt_col2 = st.columns(2)
            with tgt_col1:
                if 'Low' in targets:
                    st.metric(
                        "Low Target", f"${targets['Low']:.2f}", help="Lowest analyst price target.")
                if 'Mean' in targets:
                    st.metric(
                        "Mean Target", f"${targets['Mean']:.2f}", help="Average analyst price target.")
            with tgt_col2:
                if 'High' in targets:
                    st.metric(
                        "High Target", f"${targets['High']:.2f}", help="Highest analyst price target.")
                if 'Median' in targets:
                    st.metric(
                        "Median Target", f"${targets['Median']:.2f}", help="Median analyst price target.")
        else:
            st.info("Price target data not available.")

    st.markdown("---")

    st.markdown("**Recent Recommendation Changes**")
    if history is not None and not history.empty:
        history_display = history.copy()
        if pd.api.types.is_datetime64_any_dtype(history_display.index):
            history_display.index = history_display.index.strftime('%Y-%m-%d')
        # Ensure columns exist before displaying
        display_cols = [col for col in ['Firm', 'To Grade',
                                        'Action'] if col in history_display.columns]
        if display_cols:
            st.dataframe(history_display[display_cols],
                         use_container_width=True)
            st.caption("Recent analyst rating changes. Source: yfinance.")
        else:
            st.info(
                "Recommendation history available but standard columns ('Firm', 'To Grade', 'Action') not found.")
            # Display raw data if columns differ
            st.dataframe(history_display, use_container_width=True)
    else:
        st.info("No recent recommendation history available.")


# --- Fundamental Data Display ---
def display_fundamental_data(financials: Optional[pd.DataFrame],
                             balance_sheet: Optional[pd.DataFrame],
                             cash_flow: Optional[pd.DataFrame]):
    """Displays fundamental financial statements in formatted tables."""
    logger.info("Displaying fundamental data.")
    st.subheader("Fundamental Financial Statements (Annual)")
    data_found = False

    if financials is not None and not financials.empty:
        data_found = True
        st.markdown("##### Income Statement")
        common_items_income = ['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense', 'Operating Income',
                               'Interest Expense', 'Income Before Tax', 'Income Tax Expense', 'Net Income', 'Basic EPS', 'Diluted EPS', 'EBITDA']
        # Ensure index is datetime before trying to access columns
        if not pd.api.types.is_datetime64_any_dtype(financials.index):
            try:
                financials.index = pd.to_datetime(financials.index)
            except Exception:
                logger.warning(
                    "Could not convert financials index to datetime.")
                # Proceed cautiously, column check might fail if index is wrong type

        cols_to_display = [
            col for col in common_items_income if col in financials.columns]
        if not cols_to_display:
            st.info("No standard income statement items found.")
        else:
            display_fin = financials[cols_to_display].copy()
            # Ensure index is formatted nicely (e.g., Year)
            if pd.api.types.is_datetime64_any_dtype(display_fin.index):
                display_fin.index = display_fin.index.strftime('%Y')
            elif pd.api.types.is_integer_dtype(display_fin.index):
                pass  # Assume index is already year
            for col in display_fin.columns:
                if pd.api.types.is_numeric_dtype(display_fin[col]):
                    display_fin[col] = display_fin[col].apply(format_value)
            # Transpose back for display
            st.dataframe(display_fin.T, use_container_width=True)
            st.caption(
                "Values typically in thousands or millions, depending on currency/source.")
    else:
        st.info("Income Statement data not available.")

    st.markdown("---")
    if balance_sheet is not None and not balance_sheet.empty:
        data_found = True
        st.markdown("##### Balance Sheet")
        common_items_balance = ['Total Assets', 'Current Assets', 'Cash And Cash Equivalents', 'Receivables', 'Inventory', 'Total Non Current Assets', 'Net PPE', 'Total Liabilities Net Minority Interest', 'Current Liabilities',
                                'Payables And Accrued Expenses', 'Current Debt', 'Total Non Current Liabilities Net Minority Interest', 'Long Term Debt And Capital Lease Obligation', 'Total Equity Gross Minority Interest', 'Stockholders Equity', 'Retained Earnings']
        if not pd.api.types.is_datetime64_any_dtype(balance_sheet.index):
            try:
                balance_sheet.index = pd.to_datetime(balance_sheet.index)
            except Exception:
                logger.warning(
                    "Could not convert balance sheet index to datetime.")

        cols_to_display = [
            col for col in common_items_balance if col in balance_sheet.columns]
        if not cols_to_display:
            st.info("No standard balance sheet items found.")
        else:
            display_bal = balance_sheet[cols_to_display].copy()
            if pd.api.types.is_datetime64_any_dtype(display_bal.index):
                display_bal.index = display_bal.index.strftime('%Y')
            for col in display_bal.columns:
                if pd.api.types.is_numeric_dtype(display_bal[col]):
                    display_bal[col] = display_bal[col].apply(format_value)
            st.dataframe(display_bal.T, use_container_width=True)
            st.caption(
                "Snapshot of assets, liabilities, and equity at year-end.")
    else:
        st.info("Balance Sheet data not available.")

    st.markdown("---")
    if cash_flow is not None and not cash_flow.empty:
        data_found = True
        st.markdown("##### Cash Flow Statement")
        common_items_cashflow = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'End Cash Position', 'Capital Expenditure',
                                 'Free Cash Flow', 'Issuance Of Capital Stock', 'Repurchase Of Capital Stock', 'Repayment Of Debt', 'Issuance Of Debt']
        if not pd.api.types.is_datetime64_any_dtype(cash_flow.index):
            try:
                cash_flow.index = pd.to_datetime(cash_flow.index)
            except Exception:
                logger.warning(
                    "Could not convert cash flow index to datetime.")

        cols_to_display = [
            col for col in common_items_cashflow if col in cash_flow.columns]
        if not cols_to_display:
            st.info("No standard cash flow statement items found.")
        else:
            display_cf = cash_flow[cols_to_display].copy()
            if pd.api.types.is_datetime64_any_dtype(display_cf.index):
                display_cf.index = display_cf.index.strftime('%Y')
            for col in display_cf.columns:
                if pd.api.types.is_numeric_dtype(display_cf[col]):
                    display_cf[col] = display_cf[col].apply(format_value)
            st.dataframe(display_cf.T, use_container_width=True)
            st.caption(
                "Movement of cash between operating, investing, and financing activities.")
    else:
        st.info("Cash Flow Statement data not available.")

    if not data_found:
        st.warning(
            "No fundamental financial statement data could be retrieved for this stock.")


# --- Main Price Chart (Price/Volume/SMA/BB) ---
def plot_stock_data(df: pd.DataFrame, symbol: str, indicators: Dict[str, bool], patterns: List[str]):
    """ Plots the main stock chart with OHLC, Volume, SMA, BB and lists patterns. """
    logger.info(f"Plotting main stock data for {symbol}")
    st.subheader("Price Chart & Volume")  # Changed subheader
    if df is None or df.empty:
        st.warning("No historical data available to plot.")
        return

    # Create figure with ONLY ONE ROW, but keep secondary y-axis for volume
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,  # Less relevant now
                        # Keep spec for volume
                        specs=[[{"secondary_y": True}]])

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                  name=f'{symbol} Price', increasing_line_color=config.POSITIVE_COLOR, decreasing_line_color=config.NEGATIVE_COLOR),
                  secondary_y=False)  # Assign to primary axis

    # 2. Volume Trace
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
                  marker_color=config.VOLUME_COLOR),
                  secondary_y=True)  # Assign to secondary axis

    # 3. Add SMA Overlays
    if indicators.get('show_sma'):
        for window in config.SMA_PERIODS:
            sma_col = f'SMA_{window}'
            if sma_col in df.columns and df[sma_col].notna().any():
                color = config.SMA_COLORS.get(
                    sma_col, '#cccccc')  # Use colors from config
                fig.add_trace(go.Scatter(x=df['Date'], y=df[sma_col], mode='lines', line=dict(
                    color=color, width=1.5), name=f'SMA {window}'),
                    secondary_y=False)  # Overlay on price axis

    # 4. Add Bollinger Bands Overlays
    if indicators.get('show_bollinger'):
        bb_upper_col, bb_lower_col = 'BB_Upper', 'BB_Lower'
        if bb_upper_col in df.columns and df[bb_upper_col].notna().any():
            fig.add_trace(go.Scatter(x=df['Date'], y=df[bb_upper_col], mode='lines', line=dict(
                color=config.BB_BAND_COLOR, width=1, dash='dash'), name='BB Upper'),
                secondary_y=False)  # Overlay on price axis
            # Add lower band only if upper exists
            if bb_lower_col in df.columns and df[bb_lower_col].notna().any():
                fig.add_trace(go.Scatter(x=df['Date'], y=df[bb_lower_col], mode='lines', line=dict(
                    color=config.BB_BAND_COLOR, width=1, dash='dash'), fill='tonexty', fillcolor=config.BB_FILL_COLOR, name='BB Lower'),
                    secondary_y=False)  # Overlay on price axis

    # --- Configure Layout for Price Chart ---
    fig.update_layout(
        template="plotly_dark",
        height=500,  # Adjust height as needed
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font_color='white'),
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis_rangeslider_visible=False,  # Hide rangeslider
        xaxis_title="Date",
        yaxis=dict(title="Price (USD)", showgrid=True,
                   gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title="Volume", showgrid=False, overlaying='y',
                    side='right', showticklabels=False)  # Keep volume axis config
    )

    # Display the main price chart
    st.plotly_chart(fig, use_container_width=True, key="main_stock_chart")

    # Display Technical Patterns
    st.subheader("Technical Pattern Analysis")
    if patterns:
        num_patterns = len(patterns)
        cols = st.columns(min(num_patterns, 3))  # Max 3 columns
        for i, pattern in enumerate(patterns):
            with cols[i % 3]:
                st.info(f"💡 {pattern}")
    else:
        st.info(
            "No significant standard technical patterns detected in the recent data.")
    st.caption(
        "Pattern analysis is based on standard indicator readings and is illustrative, not investment advice.")


# --- Technical Indicators Plot (RSI/MACD) ---
def plot_technical_indicators(df: pd.DataFrame, indicators: Dict[str, bool]):
    """Plots technical indicators like RSI and MACD in separate subplots."""
    logger.info("Plotting technical indicators (RSI, MACD).")

    # Check which indicators are enabled AND data exists
    show_rsi = indicators.get(
        'show_rsi', False) and 'RSI' in df.columns and df['RSI'].notna().any()
    show_macd = indicators.get(
        'show_macd', False) and 'MACD' in df.columns and df['MACD'].notna().any()

    if not show_rsi and not show_macd:
        logger.info("No RSI or MACD indicators enabled or available to plot.")
        return  # Don't plot anything if neither is selected

    num_rows = 0
    subplot_titles_list = []
    if show_rsi:
        num_rows += 1
        subplot_titles_list.append("RSI")
    if show_macd:
        num_rows += 1
        subplot_titles_list.append("MACD")

    if num_rows == 0:
        return  # Safety check

    st.subheader("Technical Indicators")  # Add subheader for this section

    # Create subplots figure
    fig_indicators = make_subplots(
        rows=num_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,  # Adjust spacing
        subplot_titles=subplot_titles_list  # Use dynamic titles
    )

    current_row = 1

    # Plot RSI
    if show_rsi:
        fig_indicators.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(
            color=config.PRIMARY_COLOR)), row=current_row, col=1)
        fig_indicators.add_hline(
            y=70, line_dash='dash', line_color='rgba(200, 0, 0, 0.5)', row=current_row, col=1)
        fig_indicators.add_hline(
            y=30, line_dash='dash', line_color='rgba(0, 150, 0, 0.5)', row=current_row, col=1)
        fig_indicators.update_yaxes(title_text="RSI", range=[
                                    0, 100], row=current_row, col=1)
        current_row += 1

    # Plot MACD
    if show_macd:
        macd_col, signal_col, hist_col = 'MACD', 'Signal_Line', 'MACD_Histogram'
        fig_indicators.add_trace(go.Scatter(x=df['Date'], y=df[macd_col], name='MACD', line=dict(
            color=config.SECONDARY_COLOR)), row=current_row, col=1)
        if signal_col in df.columns and df[signal_col].notna().any():
            fig_indicators.add_trace(go.Scatter(x=df['Date'], y=df[signal_col], name='Signal Line', line=dict(
                color=config.PRIMARY_COLOR, dash='dot')), row=current_row, col=1)
        if hist_col in df.columns and df[hist_col].notna().any():
            hist_colors = [config.POSITIVE_COLOR if v >=
                           0 else config.NEGATIVE_COLOR for v in df[hist_col]]
            fig_indicators.add_trace(go.Bar(x=df['Date'], y=df[hist_col], name='Histogram',
                                            marker_color=hist_colors), row=current_row, col=1)
        fig_indicators.update_yaxes(title_text="MACD", row=current_row, col=1)
        # current_row += 1 # No need to increment after the last plot

    # Update layout for the indicator subplots
    fig_indicators.update_layout(
        template="plotly_dark",
        height=250 * num_rows,  # Adjust height based on number of indicators shown
        hovermode='x unified',
        showlegend=False,  # Hide legend as traces are simple
        margin=dict(l=50, r=20, t=40 if subplot_titles_list else 20,
                    b=50)  # Adjust top margin for titles
    )
    # Ensure bottom x-axis label is visible and add title
    fig_indicators.update_xaxes(showticklabels=True, row=num_rows, col=1)
    fig_indicators.update_xaxes(title_text="Date", row=num_rows, col=1)

    # Display the indicators chart
    st.plotly_chart(fig_indicators, use_container_width=True,
                    key="indicator_subplots")


def display_company_info(info: Optional[Dict[str, Any]], show_tooltips: bool):
    """Displays company overview and key financial metrics."""
    logger.info(
        f"Displaying company info for {info.get('symbol', 'N/A') if info else 'N/A'}")
    if not info:
        st.warning("Company information unavailable.")
        return

    st.subheader(
        f"Company Overview: {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")
    # --- Top Row Metrics ---
    cols_metrics = st.columns(4)
    with cols_metrics[0]:
        value = info.get('currentPrice') or info.get(
            'regularMarketPrice') or info.get('previousClose')
        st.metric("Last Price", f"${value:,.2f}" if pd.notna(
            value) else "N/A", help="Most recent available closing or intraday price.")
    with cols_metrics[1]:
        mkt_cap = info.get('marketCap')
        # Use format_value for consistency
        mkt_cap_str = format_value(mkt_cap) if mkt_cap else "N/A"
        st.metric("Market Cap", mkt_cap_str,
                  help="Total market value of all outstanding shares (Current Price * Shares Outstanding). Indicates company size.")
    with cols_metrics[2]:
        st.metric("Sector", info.get('sector', 'N/A'),
                  help="The broad economic sector the company belongs to.")
    with cols_metrics[3]:
        st.metric("Industry", info.get('industry', 'N/A'),
                  help="The specific industry group the company operates within.")

    st.markdown("---")
    st.subheader("Business Summary")
    st.info(info.get('longBusinessSummary', 'No business summary available.'))
    st.markdown("---")

    # --- Expander for Detailed Metrics with Enhanced Tooltips ---
    with st.expander("Key Financial Metrics & Ratios", expanded=False):
        # --- UPDATED metric_definitions with detailed tooltips ---
        metric_definitions = {
            # Valuation Ratios
            'trailingPE': ("Trailing P/E", "Price-to-Earnings ratio based on the last 12 months of actual reported earnings (EPS). Lower values may indicate undervaluation, higher values may indicate overvaluation or high growth expectations.", ".2f"),
            'forwardPE': ("Forward P/E", "Price-to-Earnings ratio based on estimated earnings for the next 12 months. Reflects market expectations about future profitability.", ".2f"),
            'priceToBook': ("Price/Book (P/B)", "Compares the company's market capitalization to its book value (Assets - Liabilities). A value below 1 might suggest undervaluation. Varies significantly by industry.", ".2f"),
            'priceToSalesTrailing12Months': ("Price/Sales (P/S)", "Compares the company's market capitalization to its total revenue over the last 12 months. Useful for companies with negative earnings.", ".2f"),
            'enterpriseValue': ("Enterprise Value (EV)", "Theoretical takeover price: Market Cap + Total Debt + Minority Interest + Preferred Shares - Cash & Cash Equivalents. Considered a more comprehensive valuation measure than market cap.", ",.0f"),
            'enterpriseToRevenue': ("EV/Revenue", "Enterprise Value divided by total revenue (usually TTM). Compares total company value to its sales generation.", ".2f"),
            'enterpriseToEbitda': ("EV/EBITDA", "Enterprise Value divided by Earnings Before Interest, Taxes, Depreciation, and Amortization. Common valuation metric that removes effects of financing and accounting decisions.", ".2f"),

            # Dividend Metrics
            'dividendYield': ("Dividend Yield", "Annual dividend per share divided by the current share price. Expressed as a percentage. Indicates the return from dividends relative to the price.", ".2%"),
            'payoutRatio': ("Payout Ratio", "Proportion of net income paid out as dividends to shareholders. A very high ratio might indicate dividends are unsustainable if earnings fall.", ".2%"),
            # yfinance usually provides this as a raw number (e.g., 1.5 for 1.5%)
            'fiveYearAvgDividendYield': ("5Y Avg Div Yield", "The average dividend yield over the past five years. Provides historical context for the current yield.", ".2f"),

            # Profitability & Margins
            'profitMargins': ("Profit Margin (Net)", "Net Income divided by Revenue (usually TTM). Shows the percentage of revenue remaining as profit after all expenses.", ".2%"),
            'grossMargins': ("Gross Margin", "Gross Profit (Revenue - Cost of Goods Sold) divided by Revenue. Indicates efficiency in production or service delivery.", ".2%"),
            'operatingMargins': ("Operating Margin", "Operating Income (EBIT) divided by Revenue. Shows profitability from core business operations before interest and taxes.", ".2%"),
            'ebitdaMargins': ("EBITDA Margin", "EBITDA divided by Revenue. Profitability before interest, taxes, depreciation, and amortization.", ".2%"),
            'returnOnEquity': ("Return on Equity (ROE)", "Net Income divided by average Shareholder Equity. Measures how effectively the company uses equity investments to generate profit. Higher is generally better.", ".2%"),
            'returnOnAssets': ("Return on Assets (ROA)", "Net Income divided by average Total Assets. Measures how efficiently the company uses its assets to generate profit.", ".2%"),

            # Per Share Data
            'trailingEps': ("Trailing EPS", "Earnings Per Share calculated from the last 12 months of reported net income.", ",.2f"),
            'forwardEps': ("Forward EPS", "Estimated Earnings Per Share for the next fiscal year based on analyst consensus.", ",.2f"),

            # Stock Price & Volume
            'beta': ("Beta", "Measures the stock's volatility relative to the overall market (typically S&P 500 = 1.0). >1 indicates higher volatility, <1 lower volatility.", ".2f"),
            'fiftyTwoWeekHigh': ("52 Week High", "Highest trading price reached in the past 52 weeks.", ",.2f"),
            'fiftyTwoWeekLow': ("52 Week Low", "Lowest trading price reached in the past 52 weeks.", ",.2f"),
            'volume': ("Volume", "Number of shares traded during the most recent trading session.", ",.0f"),
            'averageVolume': ("Avg Volume (10 Day)", "Average daily trading volume over the past 10 sessions.", ",.0f"),

            # Share Structure
            'sharesOutstanding': ("Shares Outstanding", "Total number of shares issued by the company.", ",.0f"),
            'floatShares': ("Float Shares", "Number of shares available for trading by the public (excludes closely held shares by insiders, governments, etc.).", ",.0f"),
        }
        # --- END UPDATED metric_definitions ---

        metrics_data = {}
        # (Logic to extract and format values remains the same)
        for key, (label, tooltip, fmt) in metric_definitions.items():
            value = info.get(key)
            if value is not None and pd.notna(value):
                try:
                    if fmt == ".2%":
                        value_str = f"{value:.2%}"
                    elif fmt == ",.0f":
                        prefix = "$" if key == 'enterpriseValue' and value >= 0 else "-$" if key == 'enterpriseValue' else ""
                        value_str = f"{prefix}{abs(value):,.0f}"
                    elif fmt == ",.2f":
                        prefix = "$" if key in ['trailingEps', 'forwardEps', 'fiftyTwoWeekHigh',
                                                'fiftyTwoWeekLow'] and value >= 0 else "-$" if key in ['trailingEps', 'forwardEps'] else ""
                        value_str = f"{prefix}{abs(value):,.2f}"
                    elif fmt == ".2f":
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)
                    metrics_data[label] = (value_str, tooltip)
                except (ValueError, TypeError) as format_err:
                    logger.warning(
                        f"Formatting error for metric {key} (Value: {value}, Type: {type(value)}): {format_err}")
                    metrics_data[label] = ("Error", tooltip)

        if not metrics_data:
            st.write("No detailed financial metrics available.")
        else:
            cols_per_row = 4
            num_metrics = len(metrics_data)
            labels_ordered = list(metrics_data.keys())
            num_rows = (num_metrics + cols_per_row - 1) // cols_per_row
            for i in range(num_rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    metric_index = i * cols_per_row + j
                    if metric_index < num_metrics:
                        label = labels_ordered[metric_index]
                        value_str, tooltip = metrics_data[label]
                        with cols[j]:
                            # Use the tooltip from the dictionary
                            st.metric(label=label, value=value_str,
                                      help=tooltip if show_tooltips else None)

# --- Sentiment Analysis Display ---


def display_sentiment_analysis(articles: List[Dict[str, Any]], avg_sentiment: float, sentiment_df: Optional[pd.DataFrame]):
    logger.info("Displaying sentiment analysis.")
    st.subheader("News & Sentiment Analysis")
    if not articles and (sentiment_df is None or sentiment_df.empty):
        st.warning("No news/sentiment data found.")
        return
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            sentiment_color_class = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
            sentiment_label = "Positive 😊" if avg_sentiment > 0.05 else "Negative 😞" if avg_sentiment < - \
                0.05 else "Neutral 😐"
            st.metric(label="Avg. News Sentiment", value=f"{avg_sentiment:.2f}", delta=sentiment_label,
                      help="Average VADER Compound Score or mapped FinBERT Score (-1 to +1) from recent news.")
        with col2:
            if sentiment_df is not None and not sentiment_df.empty:
                st.markdown(
                    "<p style='font-weight: bold; text-align: center; margin-bottom: 0.5em;'>Daily Sentiment Trend</p>", unsafe_allow_html=True)
                if not pd.api.types.is_datetime64_any_dtype(sentiment_df.index):
                    try:
                        sentiment_df.index = pd.to_datetime(sentiment_df.index)
                    except Exception as e:
                        logger.error(
                            f"Failed to convert sentiment_df index to datetime: {e}")
                        st.warning(
                            "Could not plot sentiment trend (date index issue).")
                        return
                fig = px.area(sentiment_df, x=sentiment_df.index, y='Daily_Sentiment', height=200, labels={
                              'Daily_Sentiment': 'Avg. Score', 'index': 'Date'})
                line_color = config.POSITIVE_COLOR if avg_sentiment > 0.05 else config.NEGATIVE_COLOR if avg_sentiment < - \
                    0.05 else config.NEUTRAL_COLOR
                fill_color = line_color.replace(
                    ')', ', 0.1)').replace('rgb', 'rgba')
                fig.update_traces(line_color=line_color, fillcolor=fill_color)
                fig.add_hline(y=0, line_dash='dash', line_color='grey')
                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=5, b=10), showlegend=False,
                                  xaxis_showgrid=False, yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                st.plotly_chart(fig, use_container_width=True,
                                key="sentiment_trend_chart")
            else:
                st.markdown("<div class='metric-card' style='display: flex; align-items: center; justify-content: center; height: 200px;'><p>No daily sentiment trend data.</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Recent News Headlines")
    if not articles:
        st.info("No recent news articles found.")
        return
    for i, article in enumerate(articles[:15]):
        sentiment = article.get('sentiment', {})
        # Use label and score from the standardized 'sentiment' dict
        label = sentiment.get('label', 'neutral')
        score = sentiment.get('score', 0.0)
        border_color = config.POSITIVE_COLOR if label == 'positive' else config.NEGATIVE_COLOR if label == 'negative' else config.NEUTRAL_COLOR
        # Show label and score
        sentiment_label = f"{label.title()} ({score:.2f})"
        source = article.get('source', {}).get('name', 'Unknown Source')
        published_at_str = article.get('publishedAt', '')
        published_date = "Unknown Date"
        if published_at_str:
            try:
                published_date = pd.to_datetime(
                    published_at_str).strftime('%b %d, %Y %H:%M')
            except Exception as date_err:
                logger.debug(
                    f"Could not parse article date '{published_at_str}': {date_err}")
                published_date = published_at_str
        expander_label = f"{article.get('title', 'No Title Available')} ({source})"
        with st.expander(expander_label):
            st.markdown(
                f"""<p style='font-size: 0.85em; color: #AAAAAA; margin-bottom: 8px;'>Published: {published_date} | Sentiment: <span style='color:{border_color}; font-weight:bold;'>{sentiment_label}</span></p><p style='font-size: 0.95em;'>{article.get('description', 'No description available.')}</p>""", unsafe_allow_html=True)
            st.link_button("Read Full Article 🔗", article.get(
                'url', '#'), type="secondary")


def display_prophet_forecast(forecast_df: Optional[pd.DataFrame], historical_df: pd.DataFrame, symbol: str, days_predicted: int):
    logger.info(f"Displaying Prophet forecast for {symbol}")
    st.subheader(f"Prophet Forecast ({days_predicted} Days)")
    if forecast_df is None or forecast_df.empty:
        st.warning("No forecast data available.")
        return
    if historical_df is None or historical_df.empty:
        st.warning("Historical data needed for context.")
        return

    fig = go.Figure()
    # Plot historical actual data
    fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Close'],
                  mode='lines', name='Actual Price', line=dict(color=config.PRIMARY_COLOR, width=2)))
    # Plot forecast line (yhat)
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(
        color=config.SECONDARY_COLOR, width=2, dash='dash')))
    # Plot confidence interval
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                  mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Confidence Interval', line=dict(
        width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', showlegend=True))

    # --- Try adding vertical line with error handling ---
    try:
        last_hist_date = pd.to_datetime(historical_df['Date'].iloc[-1])
        forecast_dates = pd.to_datetime(forecast_df['ds'])
        first_forecast_date = forecast_dates[forecast_dates > last_hist_date].min(
        )

        vline_date = None
        if pd.notna(first_forecast_date):
            vline_date = first_forecast_date
        else:
            vline_date = forecast_dates.min()  # Fallback

        # --- Pass the Timestamp object directly ---
        if pd.notna(vline_date):
            fig.add_vline(
                x=vline_date,  # Use the Timestamp object
                line_width=1,
                line_dash="dash",
                line_color="grey",
                annotation_text="Forecast Start",
                annotation_position="top left"
            )
        # --- End Timestamp passing ---

    except TypeError as te:
        # Catch the specific error related to timestamp arithmetic
        logger.warning(
            f"Could not add forecast start vline due to TypeError: {te}. Plot will continue without the line.")
    except Exception as e:
        # Catch other potential errors during vline calculation/addition
        logger.warning(f"Could not add forecast start vline: {e}")
    # --- End error handling ---

    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price (USD)", height=500, hovermode='x unified', legend=dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font_color='white'), margin=dict(l=50, r=20, t=30, b=20))
    fig.update_xaxes(type='date')  # Ensure x-axis is treated as date
    st.plotly_chart(fig, use_container_width=True,
                    key="prophet_forecast_chart")

    # Display tail of forecast data
    st.write("Forecasted Values (Upcoming):")
    forecast_display = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'}).set_index('Date')
    forecast_display.index = pd.to_datetime(forecast_display.index)
    future_forecast = forecast_display[forecast_display.index >= pd.Timestamp.today(
    ).normalize()]
    if not future_forecast.empty:
        st.dataframe(future_forecast.head(days_predicted).style.format(
            "${:.2f}"), use_container_width=True)
    else:
        st.info("No future dates in the forecast data to display.")
    st.caption("Forecasts generated using Prophet. Not investment advice.")


# --- Portfolio Simulation Display ---
def display_portfolio_analysis(result: Optional[Dict[str, Any]], initial_investment: float):
    logger.info("Displaying portfolio simulation results.")
    st.subheader("Portfolio Simulation (1 Year Performance)")
    if not result:
        st.warning("Portfolio simulation data unavailable.")
        return
    symbols_used = result.get('symbols_used', [])
    weights = result.get('weights', {})
    st.caption(f"Simulated with: {', '.join(symbols_used)}. Weights: " +
               ", ".join([f"{s}: {w*100:.1f}%" for s, w in weights.items()]))
    final_val = result['cumulative_value'].iloc[-1]
    total_return = (final_val / initial_investment) - \
        1 if initial_investment > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Value", f"${final_val:,.2f}", f"{total_return:.2%}",
                  help="Portfolio value after the 1-year simulation period.")
    with col2:
        st.metric("Annualized Return", f"{result['annualized_return']:.2%}",
                  help="The geometric average annual rate of return.")
    with col3:
        st.metric("Annualized Volatility",
                  f"{result['annualized_volatility']:.2%}", help="Standard deviation of returns (risk measure).")
    with col4:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}",
                  help="Risk-adjusted return (higher is generally better). Assumes 0% risk-free rate.")
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.write("**Portfolio Value Over Time**")
        value_df = result['cumulative_value']
        if pd.api.types.is_datetime64_any_dtype(value_df.index):
            fig_value = px.area(value_df, title=None, labels={
                                'value': 'Portfolio Value ($)', 'index': 'Date'}, height=350)
            fig_value.update_traces(line_color=config.PRIMARY_COLOR, fillcolor=config.PRIMARY_COLOR.replace(
                ')', ', 0.1)').replace('rgb', 'rgba'))
            fig_value.update_layout(
                template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_value, use_container_width=True,
                            key="portfolio_value_chart")
        else:
            logger.warning("Portfolio value chart index is not datetime.")
            st.warning("Could not plot portfolio value (date index issue).")
    with col_chart2:
        st.write("**Portfolio Allocation**")
        if weights:
            weights_df = pd.DataFrame(list(weights.items()), columns=[
                                      'Stock', 'Weight']).sort_values('Weight', ascending=False)
            weights_df = weights_df[weights_df['Weight'] > 0.001]
            if not weights_df.empty:
                fig_weights = px.pie(
                    weights_df, values='Weight', names='Stock', hole=0.3, height=350, title=None)
                fig_weights.update_traces(
                    textposition='outside', textinfo='percent+label', pull=[0.03]*len(weights_df))
                fig_weights.update_layout(
                    template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(
                    fig_weights, use_container_width=True, key="portfolio_weights_pie")
            else:
                st.warning(
                    "Could not display weights chart (no stocks with >0.1% weight).")
        else:
            st.warning("Portfolio weights data not available.")


# --- Correlation Analysis Display ---
def display_correlation_analysis(matrix: Optional[pd.DataFrame]):
    logger.info("Displaying correlation analysis.")
    st.subheader("Stock Correlation Analysis (1 Year Daily Returns)")
    if matrix is None or matrix.empty:
        st.warning("No correlation data available.")
        return
    if len(matrix.columns) < 2:
        st.warning("Correlation requires at least two stocks.")
        return
    fig = px.imshow(matrix, labels=dict(color="Correlation Coefficient"), x=matrix.columns, y=matrix.index, template="plotly_dark",
                    color_continuous_scale='RdBu', zmin=-1, zmax=1, text_auto='.2f', aspect="auto", height=max(400, len(matrix.columns)*50))
    fig.update_traces(
        hovertemplate='Correlation(%{x}, %{y}) = %{z:.2f}<extra></extra>')
    fig.update_layout(template="plotly_dark", margin=dict(
        l=10, r=10, t=30, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key='correlation_heatmap')
    st.caption("Correlation measures how stock returns move relative to each other (+1: perfect positive, -1: perfect negative, 0: no linear relationship).")


# --- ML Prediction Display ---
def display_ml_prediction(results: Optional[Dict[str, Any]], symbol: str):
    logger.info(f"Displaying ML prediction results for {symbol}.")
    st.subheader(f"Machine Learning Price Prediction (Illustrative)")
    if not results:
        st.warning("Machine Learning prediction results are not available.")
        return
    model_names = list(results.keys())
    if not model_names:
        st.warning("No ML models were selected or processed successfully.")
        return
    model_tabs = st.tabs(model_names)
    for i, model_name in enumerate(model_names):
        with model_tabs[i]:
            model_result = results[model_name]
            st.markdown(f"#### Results for: {model_name}")
            if not isinstance(model_result, dict) or model_result.get("error"):
                error_msg = model_result.get("error", "Unknown error") if isinstance(
                    model_result, dict) else "Invalid result format"
                st.error(
                    f"Failed to generate results for {model_name}: {error_msg}")
                continue
            pred_price = model_result.get('next_day_prediction_price')
            pred_return = model_result.get('next_day_prediction_return')
            last_actual = model_result.get('last_actual_close')
            rmse_returns = model_result.get('rmse_returns')
            feature_importance = model_result.get('feature_importance')
            y_test_actual_prices = model_result.get('y_test_actual_prices')
            predictions_prices = model_result.get('predictions_prices')
            features_used = model_result.get('features_used', [])
            st.markdown("**Prediction & Model Performance**")
            col1, col2 = st.columns(2)
            with col1:
                delta_str, delta_help = "N/A", ""
                if pred_price is not None and last_actual is not None and last_actual != 0:
                    delta = (pred_price / last_actual) - 1
                    delta_str = f"{delta:.2%}"
                    delta_help = f"Change vs last actual close of ${last_actual:.2f}"
                st.metric("Predicted Next Day Close", f"${pred_price:.2f}" if pred_price is not None else "N/A",
                          delta=delta_str if delta_str != "N/A" else None, help=f"{model_name}: Model's price forecast. {delta_help}")
                if pred_return is not None:
                    st.caption(f"Predicted Return: {pred_return:.4%}")
            with col2:
                st.metric(f"{model_name} Test RMSE (Returns)", f"{rmse_returns:.6f}" if rmse_returns is not None else "N/A",
                          help=f"{model_name}: Typical prediction error for *daily returns* (lower is better).")
            st.markdown("---")
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("**Model Performance on Test Data (Prices)**")
                fig = go.Figure()
                plot_successful = False
                if isinstance(y_test_actual_prices, pd.Series) and isinstance(predictions_prices, pd.Series) and pd.api.types.is_datetime64_any_dtype(y_test_actual_prices.index) and y_test_actual_prices.index.equals(predictions_prices.index) and not y_test_actual_prices.empty:
                    fig.add_trace(go.Scatter(x=y_test_actual_prices.index, y=y_test_actual_prices.values,
                                  mode='lines', name='Actual', line=dict(color=config.PRIMARY_COLOR, width=2)))
                    fig.add_trace(go.Scatter(x=predictions_prices.index, y=predictions_prices.values, mode='lines',
                                  name=f'{model_name} Predicted', line=dict(color=config.SECONDARY_COLOR, dash='dot', width=2)))
                    plot_successful = True
                else:
                    logger.warning(
                        f"Could not plot performance for {model_name}: Data validation failed.")
                    st.warning(
                        f"Could not plot performance for {model_name}: Data validation failed (check logs).")
                if plot_successful:
                    fig.update_layout(template="plotly_dark", xaxis_title='Date', yaxis_title='Price (USD)', height=350, margin=dict(
                        l=20, r=20, t=30, b=20), hovermode='x unified', legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="left", x=0))
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"ml_test_plot_{model_name}")
            with plot_col2:
                if feature_importance is not None and not feature_importance.empty:
                    st.write("**Feature Influence**")
                    st.caption(
                        f"Top features influencing {model_name}. Used: {', '.join(features_used)}")
                    importance_col = 'Importance' if 'Importance' in feature_importance.columns else 'Coefficient' if 'Coefficient' in feature_importance.columns else None
                    if importance_col:
                        feature_importance[importance_col] = pd.to_numeric(
                            feature_importance[importance_col], errors='coerce')
                        feature_importance = feature_importance.dropna(
                            subset=[importance_col])
                        sort_key = abs if importance_col == 'Coefficient' else lambda x: x
                        feature_importance = feature_importance.sort_values(
                            importance_col, ascending=False, key=sort_key)
                        fig_feat = px.bar(feature_importance.head(10), x=importance_col, y='Feature', orientation='h', height=350, text_auto='.3f', labels={
                                          importance_col: f'Relative {importance_col}', 'Feature': ''})
                        fig_feat.update_layout(template="plotly_dark", yaxis={
                                               'categoryorder': 'total ascending'}, margin=dict(l=10, r=10, t=30, b=20))
                        fig_feat.update_traces(
                            marker_color=config.PRIMARY_COLOR)
                        st.plotly_chart(
                            fig_feat, use_container_width=True, key=f"ml_feature_importance_{model_name}")
                    else:
                        st.warning(
                            f"Could not determine importance/coefficient column for {model_name}.")
                else:
                    st.warning(
                        f"Feature importance/coefficients not available for {model_name}.")
    st.caption(
        "ML predictions predict daily returns & reconstruct price. Experimental only. Not investment advice.")


# --- ESG Scores Display ---
def display_esg_scores(scores: Optional[Dict[str, Any]], symbol: str):
    logger.info(f"Displaying ESG scores for {symbol}.")
    st.subheader(f"ESG Performance for {symbol}")
    if scores is None:
        st.info(
            f"ESG scores unavailable or could not be retrieved for {symbol}.")
        return
    if not scores:
        st.info(
            f"No specific ESG scores found for {symbol}, although sustainability data might exist.")
        return
    esg_data = []
    score_map_display = {'Total ESG Score': scores.get('Total ESG Score'), 'Environmental Score': scores.get('Environmental Score'), 'Social Score': scores.get(
        'Social Score'), 'Governance Score': scores.get('Governance Score'), 'Highest Controversy': scores.get('Highest Controversy')}
    for label, value in score_map_display.items():
        if value is not None and pd.api.types.is_number(value) and pd.notna(value):
            esg_data.append({'Category': label, 'Score': float(value)})
        elif value is not None:
            logger.debug(f"Non-numeric ESG value found for {label}: {value}")
    if not esg_data:
        st.info("Could not extract formatted numeric ESG scores for plotting.")
        if scores:
            st.json(scores)  # Display raw scores if available
        return
    esg_df = pd.DataFrame(esg_data)
    fig = px.bar(esg_df, x='Category', y='Score', text='Score', color='Category', labels={
                 'Score': 'Score (Lower is often better for Risk)'})
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(template="plotly_dark", xaxis_title="", height=400, margin=dict(
        l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="esg_chart")
    st.caption("ESG scores typically from providers like Sustainalytics via yfinance. Lower scores often indicate lower unmanaged ESG risk. Interpretation varies.")


# --- Earnings Calendar Display ---
def display_earnings_calendar(calendar_df: Optional[pd.DataFrame], symbol: str):
    logger.info(f"Displaying earnings calendar for {symbol}.")
    st.subheader(f"Earnings Calendar for {symbol}")
    if calendar_df is None or calendar_df.empty:
        st.info(f"No earnings data found or available for {symbol}.")
        return
    display_df = calendar_df.copy()
    if 'Earnings Date' in display_df.columns:
        display_df['Earnings Date'] = pd.to_datetime(
            display_df['Earnings Date'], errors='coerce')
        display_df = display_df.dropna(subset=['Earnings Date'])
    else:
        st.warning("Earnings data missing 'Earnings Date' column.")
        return
    today = pd.Timestamp.today().normalize()
    display_df = display_df[(display_df['Earnings Date'] >= today - pd.Timedelta(days=90)) & (
        display_df['Earnings Date'] <= today + pd.Timedelta(days=180))].sort_values('Earnings Date')
    if display_df.empty:
        st.info(
            f"No recent or upcoming earnings dates found in the typical range (-90d to +180d).")
        return
    display_df['Earnings Date'] = display_df['Earnings Date'].dt.strftime(
        '%b %d, %Y')
    currency_cols = ['Earnings Average', 'Earnings Low', 'Earnings High']
    revenue_cols = ['Revenue Average', 'Revenue Low', 'Revenue High']
    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
    for col in revenue_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: format_value(x) if pd.notna(x) else '-')
    potential_cols = ['Earnings Date', 'Earnings Average', 'Revenue Average',
                      'Earnings Low', 'Earnings High', 'Revenue Low', 'Revenue High']
    display_cols = [c for c in potential_cols if c in display_df.columns]
    if 'Earnings Date' in display_cols:
        display_df = display_df.set_index('Earnings Date')
        display_cols.remove('Earnings Date')
    st.dataframe(display_df[display_cols], use_container_width=True)
    st.caption("Recent/Upcoming earnings dates and estimates (Source: yfinance).")


# --- Dividend History Display ---
def display_dividend_history(dividends: Optional[pd.Series], symbol: str):
    logger.info(f"Displaying dividend history for {symbol}.")
    st.subheader(f"Dividend History for {symbol}")
    if dividends is None or dividends.empty:
        st.info(f"No dividend history found or available for {symbol}.")
        return
    if not pd.api.types.is_datetime64_any_dtype(dividends.index):
        try:
            dividends.index = pd.to_datetime(dividends.index)
        except Exception as e:
            logger.warning(f"Could not parse dividend dates: {e}")
            st.warning(
                "Could not display dividend history (date parsing issues).")
            return
    fig = px.bar(x=dividends.index, y=dividends.values, labels={
                 'x': 'Date', 'y': 'Dividend per Share ($)'}, height=400)
    fig.update_traces(marker_color=config.POSITIVE_COLOR)
    fig.update_layout(template="plotly_dark",
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True,
                    key="dividend_history_chart")
    st.caption("Historical dividend payments per share (Source: yfinance).")


# --- Sector Performance Display ---
def display_sector_performance():
    logger.info("Fetching and displaying sector performance.")
    sectors = {"XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials", "XLY": "Cons. Disc.", "XLC": "Comm. Svcs.", "XLI": "Industrials",
               "XLP": "Cons. Staples", "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials", "XLE": "Energy", "SPY": "S&P 500"}
    performance = {}
    period = "1mo"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    etf_symbols = list(sectors.keys())
    try:
        data = yf.download(etf_symbols, start=start_date.strftime(
            config.DATE_FORMAT), end=end_date.strftime(config.DATE_FORMAT), progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data:
            raise ValueError(
                f"No valid sector ETF data for period {start_date.date()} to {end_date.date()}.")
        close_prices = data['Close']
        for etf, sector_name in sectors.items():
            if etf in close_prices.columns:
                etf_series = close_prices[etf].dropna()
                if len(etf_series) >= 2:
                    start_price = etf_series.iloc[0]
                    end_price = etf_series.iloc[-1]
                    performance[sector_name] = (
                        (end_price - start_price) / start_price) * 100 if start_price != 0 else 0.0
                else:
                    logger.warning(
                        f"Insufficient data points ({len(etf_series)}) for {sector_name} ({etf}).")
                    performance[sector_name] = None
            else:
                logger.warning(f"Data for sector ETF {etf} not found.")
                performance[sector_name] = None
    except Exception as e:
        logger.error(
            f"Error fetching/processing sector data: {e}", exc_info=True)
        st.warning("Could not retrieve sector performance data.")
        return
    valid_performance = {k: v for k, v in performance.items() if v is not None}
    if not valid_performance:
        st.warning(f"No valid sector performance data calculated.")
        return
    df_perf = pd.DataFrame(list(valid_performance.items()), columns=[
                           "Sector/Index", f"{period} % Change"]).sort_values(f"{period} % Change", ascending=False)
    fig = px.bar(df_perf, x="Sector/Index", y=f"{period} % Change", title=f"Sector & S&P 500 Performance (Approx. {period})",
                 color=f"{period} % Change", color_continuous_scale='RdYlGn', labels={f"{period} % Change": "% Change"}, height=400)
    fig.update_layout(template="plotly_dark", xaxis_tickangle=-
                      45, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True, key="sector_perf_chart_v3")


# --- Economic Indicators Display ---
def display_economic_indicators(economic_data: Optional[pd.DataFrame], stock_data: Optional[pd.DataFrame] = None):
    logger.info("Displaying economic indicators.")
    st.subheader("Economic Indicators")
    if economic_data is None or economic_data.empty:
        st.info("No economic indicator data fetched or available.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Plotly
    for i, indicator in enumerate(economic_data.columns):
        fig.add_trace(go.Scatter(x=economic_data.index, y=economic_data[indicator], name=indicator, mode='lines', line=dict(
            color=colors[i % len(colors)])), secondary_y=False)
    primary_axis_title = "Indicator Value"
    secondary_axis_title = "Stock Price ($)"
    show_secondary = False
    if stock_data is not None and not stock_data.empty and 'Close' in stock_data.columns and 'Date' in stock_data.columns:
        stock_plot_data = stock_data.set_index(
            'Date') if 'Date' in stock_data.columns else stock_data
        if pd.api.types.is_datetime64_any_dtype(stock_plot_data.index):
            fig.add_trace(go.Scatter(x=stock_plot_data.index, y=stock_plot_data['Close'], name="Stock Price", mode='lines', line=dict(
                color='rgba(200, 200, 200, 0.5)', dash='dot')), secondary_y=True)
            show_secondary = True
        else:
            logger.warning(
                "Could not overlay stock price on economic chart (Date index issue).")
    fig.update_layout(template="plotly_dark", title="Selected Economic Indicators Over Time" + (" (Stock Price Overlay)" if show_secondary else ""), height=500, hovermode='x unified', legend_title_text='Series', legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), yaxis=dict(title_text=primary_axis_title), yaxis2=dict(title_text=secondary_axis_title, overlaying='y', side='right', showgrid=False) if show_secondary else None)
    fig.update_xaxes(type='date')
    st.plotly_chart(fig, use_container_width=True,
                    key="economic_indicators_chart")
    st.caption("Source: FRED (Federal Reserve Economic Data)")
