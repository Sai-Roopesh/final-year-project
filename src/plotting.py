# src/plotting.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import yfinance as yf
# Ensure datetime and timedelta are imported
from datetime import datetime, timedelta

# Use relative imports within the src package
from . import config
from . import utils
logger = utils.logger
format_value = utils.format_value

# --- ADDED: Currency Symbol Mapping ---
CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    # Add more as needed
}
# --- END ADDED ---

# --- Analyst Recommendations Display ---


# Added currency_code
def display_analyst_recommendations(analyst_info: Optional[Dict[str, Any]], currency_code: str = "USD"):
    """Displays analyst recommendations and target price data."""
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info("Displaying analyst recommendations.")
    st.subheader("Analyst Ratings & Price Targets")

    if not analyst_info:
        st.info("Analyst recommendation data not available for this stock.")
        return

    targets = analyst_info.get('targets', {})
    current_rec = analyst_info.get('current_recommendation', 'N/A')
    num_analysts = analyst_info.get('num_analysts')
    history = analyst_info.get('history')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Current Consensus**")
        rec_display = current_rec.replace('_', ' ').title(
        ) if isinstance(current_rec, str) else 'N/A'
        st.metric("Recommendation", rec_display,
                  help="Consensus recommendation key.")
        if num_analysts is not None:
            st.caption(f"Based on {num_analysts} analysts.")
        else:
            st.caption("Number of analysts not available.")

    with col2:
        st.markdown("**Price Targets**")
        if targets:
            tgt_col1, tgt_col2 = st.columns(2)
            with tgt_col1:
                if 'Low' in targets:
                    st.metric(
                        "Low Target", f"{currency_symbol}{targets['Low']:.2f}", help="Lowest analyst price target.")
                if 'Mean' in targets:
                    st.metric(
                        "Mean Target", f"{currency_symbol}{targets['Mean']:.2f}", help="Average analyst price target.")
            with tgt_col2:
                if 'High' in targets:
                    st.metric(
                        "High Target", f"{currency_symbol}{targets['High']:.2f}", help="Highest analyst price target.")
                if 'Median' in targets:
                    st.metric(
                        "Median Target", f"{currency_symbol}{targets['Median']:.2f}", help="Median analyst price target.")
        else:
            st.info("Price target data not available.")

    st.markdown("---")
    st.markdown("**Recent Recommendation Changes**")
    # ... (rest of history display remains the same) ...
    if history is not None and not history.empty:
        history_display = history.copy()
        if pd.api.types.is_datetime64_any_dtype(history_display.index):
            try:
                history_display.index = history_display.index.strftime(
                    '%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Could not format history index as date: {e}")
        display_cols = [col for col in ['Firm', 'To Grade',
                                        'Action'] if col in history_display.columns]
        if display_cols:
            st.dataframe(history_display[display_cols],
                         use_container_width=True)
            st.caption("Recent analyst rating changes. Source: yfinance.")
        else:
            st.info(
                "Standard columns ('Firm', 'To Grade', 'Action') not found in history.")
            # Display raw if needed
            st.dataframe(history_display, use_container_width=True)
    else:
        st.info("No recent recommendation history available.")


# --- Fundamental Data Display ---
def display_fundamental_data(financials: Optional[pd.DataFrame],
                             balance_sheet: Optional[pd.DataFrame],
                             cash_flow: Optional[pd.DataFrame]):
    """Displays fundamental financial statements in formatted tables."""
    # Note: This function uses format_value which doesn't add currency symbols.
    # Captions indicate units. Could be enhanced later if needed.
    logger.info("Displaying fundamental data.")
    # ... (existing code for this function remains the same) ...
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

        cols_to_display = [
            col for col in common_items_income if col in financials.columns]
        if not cols_to_display:
            st.info("No standard income statement items found.")
        else:
            display_fin = financials[cols_to_display].copy()
            if pd.api.types.is_datetime64_any_dtype(display_fin.index):
                display_fin.index = display_fin.index.strftime('%Y')
            elif pd.api.types.is_integer_dtype(display_fin.index):
                pass  # Assume index is already year
            for col in display_fin.columns:
                if pd.api.types.is_numeric_dtype(display_fin[col]):
                    display_fin[col] = display_fin[col].apply(format_value)
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
# --- MODIFIED Signature ---
# Add currency_code
def plot_stock_data(df: pd.DataFrame, symbol: str, indicators: Dict[str, bool], patterns: List[str], currency_code: str = "USD"):
    """ Plots the main stock chart with OHLC, Volume, SMA, BB and lists patterns. """
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info(f"Plotting main stock data for {symbol}")
    st.subheader("Price Chart & Volume")
    if df is None or df.empty:
        st.warning("No historical data available to plot.")
        return

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, specs=[[{"secondary_y": True}]])

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                  name=f'{symbol} Price', increasing_line_color=config.POSITIVE_COLOR, decreasing_line_color=config.NEGATIVE_COLOR),
                  secondary_y=False)

    # 2. Volume Trace
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=config.VOLUME_COLOR),
                  secondary_y=True)

    # 3. Add SMA Overlays
    if indicators.get('show_sma'):
        for window in config.SMA_PERIODS:
            sma_col = f'SMA_{window}'
            if sma_col in df.columns and df[sma_col].notna().any():
                color = config.SMA_COLORS.get(sma_col, '#cccccc')
                fig.add_trace(go.Scatter(x=df['Date'], y=df[sma_col], mode='lines', line=dict(color=color, width=1.5), name=f'SMA {window}'),
                              secondary_y=False)

    # 4. Add Bollinger Bands Overlays
    if indicators.get('show_bollinger'):
        bb_upper_col, bb_lower_col = 'BB_Upper', 'BB_Lower'
        if bb_upper_col in df.columns and df[bb_upper_col].notna().any():
            fig.add_trace(go.Scatter(x=df['Date'], y=df[bb_upper_col], mode='lines', line=dict(color=config.BB_BAND_COLOR, width=1, dash='dash'), name='BB Upper', showlegend=False),
                          secondary_y=False)
            if bb_lower_col in df.columns and df[bb_lower_col].notna().any():
                fig.add_trace(go.Scatter(x=df['Date'], y=df[bb_lower_col], mode='lines', line=dict(color=config.BB_BAND_COLOR, width=1, dash='dash'),
                                         fill='tonexty', fillcolor=config.BB_FILL_COLOR, name='Bollinger Bands'),
                              secondary_y=False)

    # --- Configure Layout for Price Chart ---
    fig.update_layout(
        template="plotly_dark", height=500, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font_color='white'),
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis_rangeslider_visible=False, xaxis_title="Date",
        # --- MODIFIED: Use dynamic currency ---
        yaxis=dict(title=f"Price ({currency_code})",
                   showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        # --- END MODIFIED ---
        yaxis2=dict(title="Volume", showgrid=False, overlaying='y',
                    side='right', showticklabels=False)
    )
    fig.update_xaxes(type='date')

    # Display the main price chart
    st.plotly_chart(fig, use_container_width=True, key="main_stock_chart")

    # Display Technical Patterns (remains the same)
    # ...
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
    # ... (existing code for this function remains the same - no currency involved) ...
    logger.info("Plotting technical indicators (RSI, MACD).")
    if df is None or df.empty:
        st.warning("No data available to plot technical indicators.")
        return

    show_rsi = indicators.get(
        'show_rsi', False) and 'RSI' in df.columns and df['RSI'].notna().any()
    show_macd = indicators.get(
        'show_macd', False) and 'MACD' in df.columns and df['MACD'].notna().any()

    if not show_rsi and not show_macd:
        logger.info("No RSI or MACD indicators enabled or available to plot.")
        return

    num_rows = 0
    subplot_titles_list = []
    if show_rsi:
        num_rows += 1
        subplot_titles_list.append("Relative Strength Index (RSI)")
    if show_macd:
        num_rows += 1
        subplot_titles_list.append("MACD")

    if num_rows == 0:
        return

    st.subheader("Technical Indicators")
    fig_indicators = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.08, subplot_titles=subplot_titles_list)

    current_row = 1
    if show_rsi:
        fig_indicators.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(
            color=config.PRIMARY_COLOR)), row=current_row, col=1)
        fig_indicators.add_hline(y=70, line_dash='dash', line_color='rgba(200, 0, 0, 0.5)',
                                 annotation_text="Overbought (70)", annotation_position="top left", row=current_row, col=1)
        fig_indicators.add_hline(y=30, line_dash='dash', line_color='rgba(0, 150, 0, 0.5)',
                                 annotation_text="Oversold (30)", annotation_position="bottom left", row=current_row, col=1)
        fig_indicators.update_yaxes(title_text="RSI", range=[
                                    0, 100], row=current_row, col=1)
        current_row += 1
    if show_macd:
        macd_col, signal_col, hist_col = 'MACD', 'Signal_Line', 'MACD_Histogram'
        if macd_col in df.columns and df[macd_col].notna().any():
            fig_indicators.add_trace(go.Scatter(x=df['Date'], y=df[macd_col], name='MACD', line=dict(
                color=config.SECONDARY_COLOR)), row=current_row, col=1)
        if signal_col in df.columns and df[signal_col].notna().any():
            fig_indicators.add_trace(go.Scatter(x=df['Date'], y=df[signal_col], name='Signal Line', line=dict(
                color=config.PRIMARY_COLOR, dash='dot')), row=current_row, col=1)
        if hist_col in df.columns and df[hist_col].notna().any():
            hist_colors = [config.POSITIVE_COLOR if v >=
                           0 else config.NEGATIVE_COLOR for v in df[hist_col]]
            fig_indicators.add_trace(go.Bar(
                x=df['Date'], y=df[hist_col], name='Histogram', marker_color=hist_colors), row=current_row, col=1)
        fig_indicators.update_yaxes(title_text="MACD", row=current_row, col=1)

    fig_indicators.update_layout(template="plotly_dark", height=250 * num_rows, hovermode='x unified',
                                 showlegend=False, margin=dict(l=50, r=20, t=40 if subplot_titles_list else 20, b=50))
    if num_rows > 0:
        fig_indicators.update_xaxes(showticklabels=True, row=num_rows, col=1)
        fig_indicators.update_xaxes(
            title_text="Date", row=num_rows, col=1, type='date')

    st.plotly_chart(fig_indicators, use_container_width=True,
                    key="indicator_subplots")


# --- Company Info Display ---
# --- MODIFIED Signature ---
# Add currency_code
def display_company_info(info: Optional[Dict[str, Any]], show_tooltips: bool, currency_code: str = "USD"):
    """Displays company overview and key financial metrics."""
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info(
        f"Displaying company info for {info.get('symbol', 'N/A') if info else 'N/A'}")
    if not info:
        st.warning("Company information unavailable.")
        return

    st.subheader(
        f"Company Overview: {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")
    cols_metrics = st.columns(4)
    with cols_metrics[0]:
        value = info.get('currentPrice') or info.get(
            'regularMarketPrice') or info.get('previousClose')
        # --- MODIFIED: Use currency_symbol ---
        st.metric("Last Price", f"{currency_symbol}{value:,.2f}" if pd.notna(
            value) else "N/A", help="Most recent price.")
    with cols_metrics[1]:
        mkt_cap = info.get('marketCap')
        mkt_cap_str = format_value(
            mkt_cap) if mkt_cap is not None and pd.notna(mkt_cap) else "N/A"
        st.metric("Market Cap", f"{currency_symbol}{mkt_cap_str}",
                  help="Total market value.")  # Added symbol here too
    with cols_metrics[2]:
        st.metric("Sector", info.get('sector', 'N/A'),
                  help="Company's sector.")
    with cols_metrics[3]:
        st.metric("Industry", info.get('industry', 'N/A'),
                  help="Company's industry.")

    st.markdown("---")
    st.subheader("Business Summary")
    st.info(info.get('longBusinessSummary', 'No business summary available.'))
    st.markdown("---")

    with st.expander("Key Financial Metrics & Ratios", expanded=False):
        metric_definitions = {
            # Valuation Ratios (Flag=False as they are ratios, not currency)
            'trailingPE': ("Trailing P/E", "...", ".2f", False),
            'forwardPE': ("Forward P/E", "...", ".2f", False),
            'priceToBook': ("Price/Book (P/B)", "...", ".2f", False),
            'priceToSalesTrailing12Months': ("Price/Sales (P/S)", "...", ".2f", False),
            # Currency=True
            'enterpriseValue': ("Enterprise Value (EV)", "...", ",.0f", True),
            'enterpriseToRevenue': ("EV/Revenue", "...", ".2f", False),
            'enterpriseToEbitda': ("EV/EBITDA", "...", ".2f", False),
            # Dividend Metrics (Yield/Ratio are %, 5Y Avg is often just number)
            'dividendYield': ("Dividend Yield", "...", ".2%", False),
            'payoutRatio': ("Payout Ratio", "...", ".2%", False),
            'fiveYearAvgDividendYield': ("5Y Avg Div Yield", "...", ".2f", False),
            # Profitability & Margins (Usually %)
            'profitMargins': ("Profit Margin (Net)", "...", ".2%", False),
            'grossMargins': ("Gross Margin", "...", ".2%", False),
            'operatingMargins': ("Operating Margin", "...", ".2%", False),
            'ebitdaMargins': ("EBITDA Margin", "...", ".2%", False),
            'returnOnEquity': ("Return on Equity (ROE)", "...", ".2%", False),
            'returnOnAssets': ("Return on Assets (ROA)", "...", ".2%", False),
            # Per Share Data (Currency=True)
            'trailingEps': ("Trailing EPS", "...", ",.2f", True),
            'forwardEps': ("Forward EPS", "...", ",.2f", True),
            # Stock Price & Volume (Currency=True for prices)
            'beta': ("Beta", "...", ".2f", False),
            'fiftyTwoWeekHigh': ("52 Week High", "...", ",.2f", True),
            'fiftyTwoWeekLow': ("52 Week Low", "...", ",.2f", True),
            'volume': ("Volume", "...", ",.0f", False),  # Volume is count
            # Volume is count
            'averageVolume': ("Avg Volume (10 Day)", "...", ",.0f", False),
            # Share Structure (Counts, no currency)
            'sharesOutstanding': ("Shares Outstanding", "...", ",.0f", False),
            'floatShares': ("Float Shares", "...", ",.0f", False),
        }

        metrics_data = {}
        for key, (label, tooltip, fmt, is_currency) in metric_definitions.items():
            value = info.get(key)
            if value is not None and pd.notna(value):
                try:
                    value_str = ""
                    # Apply currency symbol only if needed
                    prefix = currency_symbol if is_currency else ""
                    # Handle potential negative currency values correctly
                    if is_currency and value < 0:
                        prefix = f"-{currency_symbol}"
                        value = abs(value)  # Use absolute value for formatting

                    # Apply formatting based on fmt string
                    if fmt == ".2%":
                        value_str = f"{value:.2%}"  # Percentage is special
                    elif fmt == ",.0f":
                        value_str = f"{prefix}{value:,.0f}"
                    elif fmt == ",.2f":
                        value_str = f"{prefix}{value:,.2f}"
                    elif fmt == ".2f":
                        value_str = f"{prefix}{value:.2f}"  # Non-comma float
                    else:
                        value_str = f"{prefix}{value}"  # Default

                    metrics_data[label] = (value_str, tooltip)
                except (ValueError, TypeError) as format_err:
                    logger.warning(
                        f"Formatting error for metric {key}: {format_err}")
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
                            st.metric(label=label, value=value_str,
                                      help=tooltip if show_tooltips else None)


# --- Sentiment Analysis Display (COMPLETE CORRECTED FUNCTION) ---
def display_sentiment_analysis(
    newsapi_articles: List[Dict[str, Any]],
    newsapi_avg_sentiment: float,
    newsapi_sentiment_df: Optional[pd.DataFrame],
    yfinance_articles: List[Dict[str, Any]],
    yfinance_avg_sentiment: float,
    yfinance_sentiment_df: Optional[pd.DataFrame],
    sentiment_method_used: str
):
    """Displays sentiment analysis results from both NewsAPI and yfinance."""
    logger.info("Displaying sentiment analysis from NewsAPI and yfinance.")
    st.subheader(
        f"News & Sentiment Analysis (using {sentiment_method_used.upper()})")

    # --- Helper function to display articles ---
    def display_articles(articles: List[Dict[str, Any]], source_name: str):
        if not articles:
            st.info(f"No recent news articles found from {source_name}.")
            return
        # Use NEWS_API_PAGE_SIZE from config if available, else use a default limit
        limit = getattr(config, 'NEWS_API_PAGE_SIZE',
                        50) // 5  # Show fewer headlines
        limit = max(5, limit)  # Ensure at least 5 are shown if available

        for i, article in enumerate(articles[:limit]):
            sentiment = article.get('sentiment', {})
            label = sentiment.get('label', 'neutral')  # FinBERT or VADER label
            score = sentiment.get(
                'compound') if 'compound' in sentiment else sentiment.get('score', 0.0)
            border_color = config.POSITIVE_COLOR if label == 'positive' else config.NEGATIVE_COLOR if label == 'negative' else config.NEUTRAL_COLOR
            # Display label and score
            sentiment_label_display = f"{label.title()} ({score:.2f})"
            source = article.get('source', {}).get(
                'name', source_name)  # Use source_name as fallback
            published_at_str = article.get('publishedAt', '')
            published_date = "Unknown Date"
            if published_at_str:
                try:
                    published_date = pd.to_datetime(
                        published_at_str).strftime('%b %d, %Y %H:%M')
                except Exception as date_err:
                    logger.debug(
                        f"Could not parse {source_name} article date '{published_at_str}': {date_err}")
                    published_date = published_at_str  # Fallback to original string
            expander_label = f"{article.get('title', 'No Title Available')} ({source})"
            with st.expander(expander_label):
                st.markdown(
                    f"""<p style='font-size: 0.85em; color: #AAAAAA; margin-bottom: 8px;'>Published: {published_date} | Sentiment: <span style='color:{border_color}; font-weight:bold;'>{sentiment_label_display}</span></p><p style='font-size: 0.95em;'>{article.get('description', 'No description available.')}</p>""", unsafe_allow_html=True)
                st.link_button("Read Full Article 🔗", article.get(
                    'url', '#'), type="secondary")

    # --- Helper function to display sentiment trend plot ---
    def display_trend_plot(df: Optional[pd.DataFrame], avg_sent: float, key_suffix: str, source_name: str):
        if df is not None and not df.empty:
            st.markdown(
                f"<p style='font-weight: bold; text-align: center; margin-bottom: 0.5em;'>{source_name} Daily Sentiment Trend</p>", unsafe_allow_html=True)
            plot_df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(plot_df.index):
                try:
                    plot_df.index = pd.to_datetime(plot_df.index)
                except Exception as e:
                    logger.error(
                        f"Failed to convert {source_name} sentiment index to datetime: {e}")
                    st.warning(
                        f"Could not plot {source_name} sentiment trend (date index issue).")
                    return
            if not plot_df.empty:
                fig = px.area(plot_df, x=plot_df.index, y='Daily_Sentiment', height=200, labels={
                              'Daily_Sentiment': 'Avg. Score', 'Date': 'Date'})
                line_color = config.POSITIVE_COLOR if avg_sent > 0.05 else config.NEGATIVE_COLOR if avg_sent < - \
                    0.05 else config.NEUTRAL_COLOR
                fill_color = line_color.replace(
                    ')', ', 0.1)').replace('rgb', 'rgba')
                fig.update_traces(line_color=line_color, fillcolor=fill_color)
                fig.add_hline(y=0, line_dash='dash', line_color='grey')
                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=5, b=10), showlegend=False,
                                  xaxis_showgrid=False, yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                st.plotly_chart(fig, use_container_width=True,
                                key=f"{key_suffix}_sentiment_trend_chart")
            else:
                st.markdown("<div class='metric-card' style='display: flex; align-items: center; justify-content: center; height: 200px;'><p>No daily sentiment data to plot.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='metric-card' style='display: flex; align-items: center; justify-content: center; height: 200px;'><p>No {source_name} daily sentiment data.</p></div>", unsafe_allow_html=True)

    # --- Section for NewsAPI ---
    st.markdown("#### NewsAPI Source")
    if not newsapi_articles and (newsapi_sentiment_df is None or newsapi_sentiment_df.empty):
        st.info("No news/sentiment data found from NewsAPI.")
    else:
        with st.container():
            col1_na, col2_na = st.columns([1, 3])
            with col1_na:
                sentiment_label = "Positive 😊" if newsapi_avg_sentiment > 0.05 else "Negative 😞" if newsapi_avg_sentiment < -0.05 else "Neutral 😐"
                st.metric(label="Avg. NewsAPI Sentiment", value=f"{newsapi_avg_sentiment:.2f}", delta=sentiment_label,
                          help=f"Average {sentiment_method_used.upper()} Score (-1 to +1) from recent NewsAPI news.")
            with col2_na:
                display_trend_plot(newsapi_sentiment_df,
                                   newsapi_avg_sentiment, "newsapi", "NewsAPI")

        st.markdown("##### Recent NewsAPI Headlines")
        display_articles(newsapi_articles, "NewsAPI")

    st.markdown("---")  # Separator

    # --- Section for yfinance ---
    st.markdown("#### Yahoo Finance Source")
    if not yfinance_articles and (yfinance_sentiment_df is None or yfinance_sentiment_df.empty):
        st.info("No news/sentiment data found from yfinance.")
    else:
        with st.container():
            col1_yf, col2_yf = st.columns([1, 3])
            with col1_yf:
                sentiment_label_yf = "Positive 😊" if yfinance_avg_sentiment > 0.05 else "Negative 😞" if yfinance_avg_sentiment < -0.05 else "Neutral 😐"
                st.metric(label="Avg. yfinance Sentiment", value=f"{yfinance_avg_sentiment:.2f}", delta=sentiment_label_yf,
                          help=f"Average {sentiment_method_used.upper()} Score (-1 to +1) from recent yfinance news.")
            with col2_yf:
                display_trend_plot(
                    yfinance_sentiment_df, yfinance_avg_sentiment, "yfinance", "yfinance")

        st.markdown("##### Recent Yahoo Finance Headlines")
        display_articles(yfinance_articles, "yfinance")


# --- Prophet Forecast Display ---
# --- MODIFIED Signature ---
# Add currency_code
def display_prophet_forecast(forecast_df: Optional[pd.DataFrame], historical_df: pd.DataFrame, symbol: str, days_predicted: int, currency_code: str = "USD"):
    """ Displays the Prophet forecast plot. """
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info(f"Displaying Prophet forecast for {symbol}")
    st.subheader(f"Prophet Forecast ({days_predicted} Days)")
    if forecast_df is None or forecast_df.empty:
        st.warning("No forecast data available.")
        return
    if historical_df is None or historical_df.empty:
        st.warning("Historical data needed for context.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Close'],
                  mode='lines', name='Actual Price', line=dict(color=config.PRIMARY_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(
        color=config.SECONDARY_COLOR, width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines',
                  # Hide from legend
                             name='Upper Bound', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Confidence Interval', line=dict(
        # Show only lower bound name
        width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', showlegend=True))

    try:
        last_hist_date = pd.to_datetime(historical_df['Date'].iloc[-1])
        forecast_dates = pd.to_datetime(forecast_df['ds'])
        first_forecast_date = forecast_dates[forecast_dates > last_hist_date].min(
        )
        vline_date = first_forecast_date if pd.notna(
            first_forecast_date) else forecast_dates.min()
        if pd.notna(vline_date):
            fig.add_vline(x=vline_date, line_width=1, line_dash="dash", line_color="grey",
                          annotation_text="Forecast Start", annotation_position="top left")
    except Exception as e:
        logger.warning(f"Could not add forecast start vline: {e}")

    fig.update_layout(
        template="plotly_dark", xaxis_title="Date",
        # --- MODIFIED: Use dynamic currency ---
        yaxis_title=f"Price ({currency_code})",
        # --- END MODIFIED ---
        height=500, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font_color='white'), margin=dict(l=50, r=20, t=30, b=20)
    )
    fig.update_xaxes(type='date')
    st.plotly_chart(fig, use_container_width=True,
                    key="prophet_forecast_chart")

    st.write("Forecasted Values (Upcoming):")
    forecast_display = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'}).set_index('Date')
    forecast_display.index = pd.to_datetime(forecast_display.index)
    future_forecast = forecast_display[forecast_display.index >= pd.Timestamp.today(
    ).normalize()]
    if not future_forecast.empty:
        # --- MODIFIED: Use currency_symbol ---
        st.dataframe(future_forecast.head(days_predicted).style.format(
            f"{currency_symbol}{{:.2f}}"), use_container_width=True)
        # --- END MODIFIED ---
    else:
        st.info("No future dates in the forecast data to display.")
    st.caption("Forecasts generated using Prophet. Not investment advice.")


# --- Portfolio Simulation Display ---
# (This function assumes USD for portfolio sim, no currency change needed here unless enhancing portfolio logic)
def display_portfolio_analysis(result: Optional[Dict[str, Any]], initial_investment: float):
    # ... (existing code remains the same) ...
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
        st.metric("Final Value", f"${final_val:,.2f}",
                  f"{total_return:.2%}", help="Portfolio value after 1 year.")
    with col2:
        st.metric("Annualized Return",
                  f"{result['annualized_return']:.2%}", help="Geometric average annual return.")
    with col3:
        st.metric("Annualized Volatility",
                  f"{result['annualized_volatility']:.2%}", help="Standard deviation of returns (risk).")
    with col4:
        st.metric(
            "Sharpe Ratio", f"{result['sharpe_ratio']:.2f}", help="Risk-adjusted return (Rf=0%).")
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
            st.warning("Could not plot portfolio value (date issue).")
    with col_chart2:
        st.write("**Portfolio Allocation**")
        if weights:
            weights_df = pd.DataFrame(list(weights.items()), columns=[
                                      'Stock', 'Weight']).sort_values('Weight', ascending=False)
            # Filter small weights
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
                st.warning("No stocks with >0.1% weight.")
        else:
            st.warning("Portfolio weights data unavailable.")


# --- Correlation Analysis Display ---
# (No currency involved)
def display_correlation_analysis(matrix: Optional[pd.DataFrame]):
    # ... (existing code remains the same) ...
    logger.info("Displaying correlation analysis.")
    st.subheader("Stock Correlation Analysis (1 Year Daily Returns)")
    if matrix is None or matrix.empty:
        st.warning("No correlation data available.")
        return
    if len(matrix.columns) < 2:
        st.warning("Correlation requires at least two stocks.")
        return
    fig = px.imshow(matrix, labels=dict(color="Correlation"), x=matrix.columns, y=matrix.index, template="plotly_dark",
                    color_continuous_scale='RdBu', zmin=-1, zmax=1, text_auto='.2f', aspect="auto", height=max(400, len(matrix.columns)*50))
    fig.update_traces(
        hovertemplate='Correlation(%{x}, %{y}) = %{z:.2f}<extra></extra>')
    fig.update_layout(template="plotly_dark", margin=dict(
        l=10, r=10, t=30, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key='correlation_heatmap')
    st.caption("Correlation measures how stock returns move relative to each other (+1: perfect positive, -1: perfect negative, 0: no linear relationship).")


# --- ML Prediction Display ---
# --- MODIFIED Signature ---
# Add currency_code
def display_ml_prediction(results: Optional[Dict[str, Any]], symbol: str, currency_code: str = "USD"):
    """Displays machine learning prediction results."""
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info(f"Displaying ML prediction results for {symbol}.")
    st.subheader(f"Machine Learning Price Prediction (Illustrative)")
    # ... (check for results, model_names remains same) ...
    if not results:
        st.warning("ML prediction results unavailable.")
        return
    model_names = list(results.keys())
    if not model_names:
        st.warning("No ML models processed.")
        return

    model_tabs = st.tabs(model_names)
    for i, model_name in enumerate(model_names):
        with model_tabs[i]:
            model_result = results[model_name]
            st.markdown(f"#### Results for: {model_name}")
            # ... (check for errors remains same) ...
            if not isinstance(model_result, dict) or model_result.get("error"):
                error_msg = model_result.get("error", "Unknown error") if isinstance(
                    model_result, dict) else "Invalid result format"
                st.error(f"Failed for {model_name}: {error_msg}")
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
                # --- MODIFIED: Use currency_symbol ---
                if pred_price is not None and last_actual is not None and last_actual != 0:
                    delta = (pred_price / last_actual) - 1
                    delta_str = f"{delta:.2%}"
                    delta_help = f"Change vs last actual close of {currency_symbol}{last_actual:.2f}"
                st.metric("Predicted Next Day Close", f"{currency_symbol}{pred_price:.2f}" if pred_price is not None else "N/A",
                          delta=delta_str if delta_str != "N/A" else None, help=f"{model_name}: Price forecast. {delta_help}")
                # --- END MODIFIED ---
                if pred_return is not None:
                    st.caption(f"Predicted Return: {pred_return:.4%}")
            with col2:
                # RMSE for returns doesn't need currency symbol
                st.metric(f"Test RMSE (Returns)", f"{rmse_returns:.6f}" if rmse_returns is not None else "N/A",
                          help=f"{model_name}: Typical prediction error for daily returns.")

            st.markdown("---")
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("**Model Performance on Test Data (Prices)**")
                fig = go.Figure()
                # ... (plotting logic remains same, check data validation) ...
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
                        f"Could not plot performance for {model_name}: Data validation failed.")

                if plot_successful:
                    fig.update_layout(template="plotly_dark", xaxis_title='Date',
                                      # --- MODIFIED: Use dynamic currency ---
                                      yaxis_title=f'Price ({currency_code})',
                                      # --- END MODIFIED ---
                                      height=350, margin=dict(l=20, r=20, t=30, b=20), hovermode='x unified',
                                      legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="left", x=0))
                    fig.update_xaxes(type='date')
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"ml_test_plot_{model_name}")

            with plot_col2:
                # (feature importance display remains the same - no currency)
                # ...
                if feature_importance is not None and not feature_importance.empty:
                    st.write("**Feature Influence**")
                    # Removed used features list for brevity
                    st.caption(f"Top features influencing {model_name}.")
                    importance_col = 'Importance' if 'Importance' in feature_importance.columns else 'Coefficient' if 'Coefficient' in feature_importance.columns else None
                    if importance_col:
                        try:  # Add error handling for numeric conversion
                            feature_importance[importance_col] = pd.to_numeric(
                                feature_importance[importance_col], errors='coerce')
                            feature_importance = feature_importance.dropna(
                                subset=[importance_col])
                            sort_key = abs if importance_col == 'Coefficient' else lambda x: x
                            feature_importance = feature_importance.sort_values(
                                # Limit here
                                importance_col, ascending=False, key=sort_key).head(10)
                            fig_feat = px.bar(feature_importance, x=importance_col, y='Feature', orientation='h', height=350, text_auto='.3f', labels={
                                              importance_col: f'Relative {importance_col}', 'Feature': ''})
                            fig_feat.update_layout(template="plotly_dark", yaxis={
                                                   'categoryorder': 'total ascending'}, margin=dict(l=10, r=10, t=30, b=20))
                            fig_feat.update_traces(
                                marker_color=config.PRIMARY_COLOR)
                            st.plotly_chart(
                                fig_feat, use_container_width=True, key=f"ml_feature_importance_{model_name}")
                        except Exception as feat_err:
                            logger.warning(
                                f"Could not plot feature importance for {model_name}: {feat_err}")
                            st.warning(
                                f"Could not plot feature importance for {model_name}.")
                    else:
                        st.warning(
                            f"Could not determine importance column for {model_name}.")
                else:
                    st.warning(
                        f"Feature influence data not available for {model_name}.")
    st.caption("ML predictions are experimental. Not investment advice.")


# --- ESG Scores Display ---
# (No currency involved)
def display_esg_scores(scores: Optional[Dict[str, Any]], symbol: str):
    # ... (existing code remains the same) ...
    logger.info(f"Displaying ESG scores for {symbol}.")
    st.subheader(f"ESG Performance for {symbol}")
    if scores is None:
        st.info(f"ESG scores unavailable for {symbol}.")
        return
    if not scores:
        st.info(f"No specific ESG scores found for {symbol}.")
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
        st.info("Could not extract numeric ESG scores for plotting.")
        return

    esg_df = pd.DataFrame(esg_data)
    fig = px.bar(esg_df, x='Category', y='Score', text='Score', color='Category', labels={
                 'Score': 'Score (Lower is often better for Risk)'})
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(template="plotly_dark", xaxis_title="", height=400, margin=dict(
        l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="esg_chart")
    st.caption(
        "ESG scores source: yfinance/Sustainalytics. Lower scores often indicate lower unmanaged ESG risk.")


# --- Earnings Calendar Display ---
# --- MODIFIED Signature ---
# Add currency_code
def display_earnings_calendar(calendar_df: Optional[pd.DataFrame], symbol: str, currency_code: str = "USD"):
    """Displays earnings calendar with currency formatting."""
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info(f"Displaying earnings calendar for {symbol}.")
    st.subheader(f"Earnings Calendar for {symbol}")
    if calendar_df is None or calendar_df.empty:
        st.info(f"No earnings data found for {symbol}.")
        return

    display_df = calendar_df.copy()
    if 'Earnings Date' in display_df.columns:
        display_df['Earnings Date'] = pd.to_datetime(
            display_df['Earnings Date'], errors='coerce')
        display_df = display_df.dropna(subset=['Earnings Date'])
    else:
        st.warning("Earnings data missing 'Earnings Date'.")
        return

    today = pd.Timestamp.today().normalize()
    # Filter for relevant dates
    display_df = display_df[(display_df['Earnings Date'] >= today - pd.Timedelta(days=90)) & (
        display_df['Earnings Date'] <= today + pd.Timedelta(days=180))].sort_values('Earnings Date')

    if display_df.empty:
        st.info(f"No recent/upcoming earnings dates found.")
        return

    display_df['Earnings Date'] = display_df['Earnings Date'].dt.strftime(
        '%b %d, %Y')

    # --- MODIFIED: Use currency_symbol in formatting ---
    currency_cols = ['Earnings Average', 'Earnings Low', 'Earnings High']
    revenue_cols = ['Revenue Average', 'Revenue Low', 'Revenue High']
    for col in currency_cols:  # Format EPS-like values
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{currency_symbol}{x:,.2f}" if pd.notna(x) else '-')
    for col in revenue_cols:  # Format Revenue values
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{currency_symbol}{format_value(x)}" if pd.notna(
                x) else '-')  # Use format_value AND add symbol
    # --- END MODIFIED ---

    potential_cols = ['Earnings Date', 'Earnings Average', 'Revenue Average',
                      'Earnings Low', 'Earnings High', 'Revenue Low', 'Revenue High']
    display_cols = [c for c in potential_cols if c in display_df.columns]
    if 'Earnings Date' in display_cols:
        display_df = display_df.set_index('Earnings Date')
        display_cols.remove('Earnings Date')
    st.dataframe(display_df[display_cols], use_container_width=True)
    st.caption("Recent/Upcoming earnings dates and estimates (Source: yfinance).")


# --- Dividend History Display ---
# --- MODIFIED Signature ---
# Add currency_code
def display_dividend_history(dividends: Optional[pd.Series], symbol: str, currency_code: str = "USD"):
    """Displays dividend history with currency formatting."""
    currency_symbol = CURRENCY_SYMBOLS.get(
        currency_code, currency_code)  # Get symbol
    logger.info(f"Displaying dividend history for {symbol}.")
    st.subheader(f"Dividend History for {symbol}")
    if dividends is None or dividends.empty:
        st.info(f"No dividend history found for {symbol}.")
        return

    if not pd.api.types.is_datetime64_any_dtype(dividends.index):
        try:
            dividends.index = pd.to_datetime(dividends.index)
        except Exception as e:
            logger.warning(f"Could not parse dividend dates: {e}")
            st.warning("Could not display dividend history (date issue).")
            return

    # --- MODIFIED: Use currency_code in label ---
    fig = px.bar(x=dividends.index, y=dividends.values, labels={
                 'x': 'Date', 'y': f'Dividend per Share ({currency_code})'}, height=400)
    # --- END MODIFIED ---
    fig.update_traces(marker_color=config.POSITIVE_COLOR)
    fig.update_layout(template="plotly_dark",
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True,
                    key="dividend_history_chart")
    st.caption("Historical dividend payments per share (Source: yfinance).")


# --- Sector Performance Display ---
# (No currency involved)
def display_sector_performance():
    # ... (existing code remains the same) ...
    logger.info("Fetching and displaying sector performance.")
    sectors = {"XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials", "XLY": "Cons. Disc.", "XLC": "Comm. Svcs.", "XLI": "Industrials",
               "XLP": "Cons. Staples", "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials", "XLE": "Energy", "SPY": "S&P 500"}
    performance = {}
    period = "1mo"  # Keep period short for relevance
    end_date = datetime.now()
    # Go back enough to capture ~1 month trading days
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
                    start_price, end_price = etf_series.iloc[0], etf_series.iloc[-1]
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
# (No currency involved)
def display_economic_indicators(economic_data: Optional[pd.DataFrame], stock_data: Optional[pd.DataFrame] = None):
    # ... (existing code remains the same) ...
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
    # Keep as $ for overlay, might need dynamic if comparing non-USD stocks
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

    fig.update_layout(template="plotly_dark", title="Selected Economic Indicators Over Time"+(" (Stock Price Overlay)" if show_secondary else ""), height=500, hovermode='x unified', legend_title_text='Series', legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), yaxis=dict(title_text=primary_axis_title), yaxis2=dict(title_text=secondary_axis_title, overlaying='y', side='right', showgrid=False) if show_secondary else None)
    fig.update_xaxes(type='date')
    st.plotly_chart(fig, use_container_width=True,
                    key="economic_indicators_chart")
    st.caption("Source: FRED (Federal Reserve Economic Data)")
