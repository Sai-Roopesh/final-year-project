# src/plotting.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import yfinance as yf

# Use relative imports within the src package
from . import config
from . import utils

logger = utils.logger  # Use the logger from utils

# --- Main Price Chart & Technicals ---
# (plot_stock_data function remains the same)


def plot_stock_data(df: pd.DataFrame, symbol: str, indicators: Dict[str, bool], patterns: List[str]):
    """ Plots the main stock chart with OHLC, Volume, selected indicators and lists patterns. """
    logger.info(f"Plotting stock data for {symbol}")
    st.subheader("Price Chart & Technical Analysis")
    if df is None or df.empty:
        st.warning("No historical data available to plot.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],
                        specs=[[{"secondary_y": True}],
                               [{"secondary_y": False}]])

    # --- Row 1: Price and Volume ---
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name=f'{symbol} Price',
        increasing_line_color=config.POSITIVE_COLOR, decreasing_line_color=config.NEGATIVE_COLOR
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'], name='Volume',
        marker_color=config.VOLUME_COLOR,
    ), row=1, col=1, secondary_y=True)

    # SMA Lines
    if indicators.get('show_sma'):
        for window in config.SMA_PERIODS:
            sma_col = f'SMA_{window}'
            if sma_col in df.columns and df[sma_col].notna().any():
                color = config.SMA_COLORS.get(sma_col, '#cccccc')
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df[sma_col], mode='lines',
                    line=dict(color=color, width=1.5), name=f'SMA {window}'
                ), row=1, col=1, secondary_y=False)

    # Bollinger Bands
    if indicators.get('show_bollinger'):
        bb_upper_col, bb_lower_col = 'BB_Upper', 'BB_Lower'
        if bb_upper_col in df.columns and df[bb_upper_col].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[bb_upper_col], mode='lines',
                line=dict(color=config.BB_BAND_COLOR, width=1, dash='dash'),
                name='BB Upper'
            ), row=1, col=1, secondary_y=False)
            if bb_lower_col in df.columns and df[bb_lower_col].notna().any():
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df[bb_lower_col], mode='lines',
                    line=dict(color=config.BB_BAND_COLOR,
                              width=1, dash='dash'),
                    fill='tonexty', fillcolor=config.BB_FILL_COLOR,
                    name='BB Lower'
                ), row=1, col=1, secondary_y=False)

    # --- Row 2: RSI and MACD ---
    added_row2 = False
    if indicators.get('show_rsi') and 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(
            color=config.PRIMARY_COLOR)), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash',
                      line_color='rgba(200, 0, 0, 0.5)', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash',
                      line_color='rgba(0, 150, 0, 0.5)', row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        added_row2 = True

    if indicators.get('show_macd') and 'MACD' in df.columns and df['MACD'].notna().any():
        macd_col, signal_col, hist_col = 'MACD', 'Signal_Line', 'MACD_Histogram'
        macd_row = 2
        fig.add_trace(go.Scatter(x=df['Date'], y=df[macd_col], name='MACD', line=dict(
            color=config.SECONDARY_COLOR)), row=macd_row, col=1)
        if signal_col in df.columns and df[signal_col].notna().any():
            fig.add_trace(go.Scatter(x=df['Date'], y=df[signal_col], name='Signal Line', line=dict(
                color=config.PRIMARY_COLOR, dash='dot')), row=macd_row, col=1)
        if hist_col in df.columns and df[hist_col].notna().any():
            hist_colors = [config.POSITIVE_COLOR if v >=
                           0 else config.NEGATIVE_COLOR for v in df[hist_col]]
            fig.add_trace(go.Bar(x=df['Date'], y=df[hist_col], name='Histogram',
                          marker_color=hist_colors), row=macd_row, col=1)
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
        added_row2 = True

    # --- Layout Updates ---
    fig.update_layout(
        template="plotly_dark", height=650 if added_row2 else 500, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font_color='white'),
        margin=dict(l=50, r=20, t=30, b=50), xaxis_rangeslider_visible=False, xaxis_showticklabels=True,
        xaxis2_showticklabels=True, yaxis=dict(title="Price (USD)", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title="Volume", showgrid=False, overlaying='y',
                    side='right', showticklabels=False),
        yaxis3=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    if not added_row2:
        fig.update_layout(height=500, xaxis2_visible=False,
                          yaxis3_visible=False)

    st.plotly_chart(fig, use_container_width=True,
                    key="main_stock_chart_indicators")

    # --- Technical Patterns Display ---
    st.subheader("Technical Pattern Analysis")
    if patterns:
        num_patterns = len(patterns)
        cols = st.columns(min(num_patterns, 3))
        for i, pattern in enumerate(patterns):
            with cols[i % 3]:
                # Using an icon for better visual cue
                st.info(f"💡 {pattern}")
    else:
        st.info("No significant standard technical patterns detected.")
    st.caption("Pattern analysis is illustrative, not investment advice.")


# --- Company Info Display ---
# (display_company_info function remains the same)
def display_company_info(info: Optional[Dict[str, Any]], show_tooltips: bool):
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
        st.metric("Last Price", f"${value:,.2f}" if pd.notna(
            value) else "N/A", help="Most recent price.")
    with cols_metrics[1]:
        mkt_cap = info.get('marketCap')
        # Format market cap nicely (Billions/Millions)
        if mkt_cap and isinstance(mkt_cap, (int, float)):
            if mkt_cap >= 1e9:
                mkt_cap_str = f"${mkt_cap / 1e9:.2f}B"
            elif mkt_cap >= 1e6:
                mkt_cap_str = f"${mkt_cap / 1e6:.2f}M"
            else:
                mkt_cap_str = f"${mkt_cap:,.0f}"
        else:
            mkt_cap_str = "N/A"
        st.metric("Market Cap", mkt_cap_str, help="Total market value.")
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
        # More comprehensive metric definitions
        metric_definitions = {
            # Valuation
            'trailingPE': ("Trailing P/E", "Price/Earnings (past 12m).", ".2f"),
            'forwardPE': ("Forward P/E", "Price/Earnings (est. next year).", ".2f"),
            'priceToBook': ("Price/Book (P/B)", "Market value vs. book value.", ".2f"),
            # Use TTM P/S
            'priceToSalesTrailing12Months': ("Price/Sales (P/S)", "Market value vs. revenue (past 12m).", ".2f"),
            'enterpriseValue': ("Enterprise Value", "Total company value (market cap + debt - cash).", ",.0f"),
            'enterpriseToRevenue': ("EV/Revenue", "EV vs. total revenue.", ".2f"),
            'enterpriseToEbitda': ("EV/EBITDA", "EV vs. EBITDA.", ".2f"),
            # Dividends & Payout
            'dividendYield': ("Dividend Yield", "Annual dividend / share price.", ".2%"),
            'payoutRatio': ("Payout Ratio", "% of earnings paid as dividends.", ".2%"),
            # Value is often float, not %
            'fiveYearAvgDividendYield': ("5Y Avg Div Yield", "Avg. dividend yield (5y).", ".2f"),
            # Profitability & Margins
            'profitMargins': ("Profit Margin", "Net income / revenue.", ".2%"),
            'grossMargins': ("Gross Margin", "Gross profit / revenue.", ".2%"),
            'operatingMargins': ("Operating Margin", "Operating income / revenue.", ".2%"),
            # Added EBITDA margin
            'ebitdaMargins': ("EBITDA Margin", "EBITDA / revenue.", ".2%"),
            'returnOnEquity': ("Return on Equity (ROE)", "Net income / shareholder equity.", ".2%"),
            'returnOnAssets': ("Return on Assets (ROA)", "Net income / total assets.", ".2%"),
            # Per Share
            # Add $ sign format
            'trailingEps': ("Trailing EPS", "Earnings per share (past 12m).", ",.2f"),
            # Add $ sign format
            'forwardEps': ("Forward EPS", "Est. earnings per share (next year).", ",.2f"),
            # Volatility & Other
            'beta': ("Beta", "Volatility vs. market (S&P 500).", ".2f"),
            # Add $ sign format
            'fiftyTwoWeekHigh': ("52 Week High", "Highest price (past 52w).", ",.2f"),
            # Add $ sign format
            'fiftyTwoWeekLow': ("52 Week Low", "Lowest price (past 52w).", ",.2f"),
            'volume': ("Volume", "Shares traded (latest session).", ",.0f"),
            'averageVolume': ("Avg Volume (10 Day)", "Avg daily volume (10d).", ",.0f"),
            # Added shares outstanding
            'sharesOutstanding': ("Shares Outstanding", "Total number of shares.", ",.0f"),
        }
        metrics_data = {}
        # Extract and format metrics safely
        for key, (label, tooltip, fmt) in metric_definitions.items():
            value = info.get(key)
            if value is not None and pd.notna(value):  # Check for None and NaN
                try:
                    if fmt == ".2%":  # Format as percentage
                        value_str = f"{value:.2%}"
                    elif fmt == ",.0f":  # Format as integer with commas
                        value_str = f"${value:,.0f}" if key == 'enterpriseValue' else f"{value:,.0f}"
                    elif fmt == ",.2f":  # Format as float with 2 decimals
                        # Add $ sign for currency-like values
                        if key in ['trailingEps', 'forwardEps', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow']:
                            value_str = f"${value:,.2f}"
                        else:  # Ratios like P/E, P/B, Beta
                            value_str = f"{value:.2f}"
                    else:  # Default string conversion
                        value_str = str(value)
                    metrics_data[label] = (value_str, tooltip)
                except (ValueError, TypeError):  # Handle potential formatting errors
                    metrics_data[label] = ("Error", tooltip)
                    logger.warning(
                        f"Formatting error for metric {key} with value {value}")
            # else: # Optionally include N/A for missing values
                # metrics_data[label] = ("N/A", tooltip)

        if not metrics_data:
            st.write("No detailed financial metrics available.")
        else:
            # Display metrics in columns
            cols_per_row = 4  # Adjust number of columns as needed
            num_metrics = len(metrics_data)
            # Keep order consistent if needed
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
                            # Use st.metric for better visual separation
                            if show_tooltips:
                                st.metric(
                                    label=label, value=value_str, help=tooltip)
                            else:
                                st.metric(label=label, value=value_str)

# --- Sentiment Analysis Display ---
# (display_sentiment_analysis function remains the same)


def display_sentiment_analysis(articles: List[Dict[str, Any]], avg_sentiment: float, sentiment_df: Optional[pd.DataFrame]):
    logger.info("Displaying sentiment analysis.")
    st.subheader("News & Sentiment Analysis")
    if not articles and (sentiment_df is None or sentiment_df.empty):
        st.warning("No news/sentiment data found.")
        return
    with st.container():
        col1, col2 = st.columns([1, 3])  # Score vs Trend chart ratio
        with col1:
            # Determine sentiment label and color
            sentiment_color_class = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
            sentiment_label = "Positive 😊" if avg_sentiment > 0.05 else "Negative 😞" if avg_sentiment < - \
                0.05 else "Neutral 😐"

            # Display metric using custom HTML for better styling control if needed, or st.metric
            st.metric("Avg. News Sentiment", f"{avg_sentiment:.2f}", delta=sentiment_label,
                      help="Average VADER Compound Score (-1 to +1) from recent news.")
            # Optional custom card styling:
            # st.markdown(f"""
            #      <div class='metric-card' style='text-align: center;'>
            #          <p style='font-size: 0.9em; color: #AAAAAA; margin-bottom: 0.2em;'>Avg. Sentiment Score</p>
            #          <p class='{sentiment_color_class}' style='font-size: 2em; font-weight: bold; margin-bottom: 0.2em;'>{avg_sentiment:.2f}</p>
            #          <p class='{sentiment_color_class}' style='font-size: 1em;'>({sentiment_label})</p>
            #      </div>
            #  """, unsafe_allow_html=True)

        with col2:
            # Plot daily sentiment trend if available
            if sentiment_df is not None and not sentiment_df.empty:
                st.markdown(
                    "<p style='font-weight: bold; text-align: center; margin-bottom: 0.5em;'>Daily Sentiment Trend</p>", unsafe_allow_html=True)
                fig = px.area(sentiment_df, x=sentiment_df.index, y='Daily_Sentiment', height=200, labels={
                              'Daily_Sentiment': 'Avg. Score', 'index': 'Date'})
                # Match line/fill color to overall sentiment
                line_color = config.POSITIVE_COLOR if avg_sentiment > 0.05 else config.NEGATIVE_COLOR if avg_sentiment < - \
                    0.05 else config.NEUTRAL_COLOR
                fill_color = line_color.replace(')', ', 0.1)').replace(
                    'rgb', 'rgba')  # Make fill semi-transparent
                fig.update_traces(line_color=line_color, fillcolor=fill_color)
                fig.add_hline(y=0, line_dash='dash',
                              line_color='grey')  # Add neutral line
                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=5, b=10), showlegend=False,
                                  xaxis_showgrid=False, yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                st.plotly_chart(fig, use_container_width=True,
                                key="sentiment_trend_chart")
            else:
                # Placeholder if no daily data
                st.markdown(
                    "<div class='metric-card'><p style='text-align: center; padding: 3rem 0;'>No daily sentiment trend data.</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Recent News Headlines")
    if not articles:
        st.info("No recent news articles found.")
        return

    # Display news articles with sentiment scores
    for i, article in enumerate(articles[:10]):  # Show top 10 articles
        sentiment = article.get('sentiment', {})
        compound_score = sentiment.get('compound', 0)
        # Determine border color based on sentiment
        border_color = config.POSITIVE_COLOR if compound_score > 0.05 else config.NEGATIVE_COLOR if compound_score < - \
            0.05 else config.NEUTRAL_COLOR
        sentiment_label = f"{compound_score:.2f}"

        source = article.get('source', {}).get('name', 'Unknown Source')
        published_at_str = article.get('publishedAt', '')
        published_date = "Unknown Date"
        if published_at_str:
            try:  # Safely parse date
                published_date = pd.to_datetime(
                    published_at_str).strftime('%b %d, %Y %H:%M')
            except Exception:
                published_date = published_at_str  # Fallback to original string

        # Use an expander for each article
        with st.expander(f"{article.get('title', 'No Title Available')} ({source})"):
            st.caption(f"Published: {published_date} | Sentiment: <span style='color:{border_color}; font-weight:bold;'>{sentiment_label}</span>",
                       unsafe_allow_html=True)
            st.markdown(f"_{article.get('description', 'No description.')}_")
            st.link_button("Read Article", article.get(
                'url', '#'), type="secondary")


# --- Prophet Forecast Display ---
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
    fig.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['Close'],
        mode='lines', name='Actual Price',
        line=dict(color=config.PRIMARY_COLOR, width=2)  # Use color from config
    ))

    # Plot forecast line (yhat)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines', name='Forecast',
        line=dict(color=config.SECONDARY_COLOR, width=2,
                  dash='dash')  # Use color from config
    ))

    # Plot confidence interval (yhat_lower, yhat_upper) - transparent fill
    # Plot upper bound line (no fill, optional legend entry)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines', name='Upper Bound',
        line=dict(width=0),  # Make line invisible
        showlegend=False  # Hide from legend
    ))
    # Plot lower bound line and fill to upper bound
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines', name='Confidence Interval',
        line=dict(width=0),  # Make line invisible
        fill='tonexty',  # Fill area to previous trace (yhat_upper)
        # Use secondary color with transparency
        fillcolor='rgba(255, 165, 0, 0.2)',
        showlegend=True  # Show interval in legend
    ))

    # Add vertical line indicating start of forecast
    # Ensure forecast_df['ds'] is sorted if not already
    start_forecast_date = forecast_df['ds'].min()
    # --- FIX: Remove annotation arguments from add_vline ---
    fig.add_vline(x=start_forecast_date, line_width=1,
                  line_dash="dash", line_color="grey")
    # --- End FIX ---

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font_color='white'),
        margin=dict(l=50, r=20, t=30, b=20)  # Adjust margins
    )

    st.plotly_chart(fig, use_container_width=True,
                    key="prophet_forecast_chart")

    # Display table of upcoming forecast values
    st.write("Forecasted Values (Upcoming):")
    forecast_display = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Forecast',
                 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'}
    ).set_index('Date')
    # Filter for future dates only
    future_forecast = forecast_display[forecast_display.index >= pd.Timestamp.today(
    ).normalize()]
    # Show only the requested number of days or available future days
    st.dataframe(future_forecast.head(days_predicted).style.format(
        "${:.2f}"), use_container_width=True)
    st.caption("Forecasts generated using Prophet. Not investment advice.")


# --- Portfolio Simulation Display ---
# (display_portfolio_analysis function remains the same)
def display_portfolio_analysis(result: Optional[Dict[str, Any]], initial_investment: float):
    logger.info("Displaying portfolio simulation results.")
    st.subheader("Portfolio Simulation (1 Year Performance)")
    if not result:
        st.warning("Portfolio simulation data unavailable.")
        return

    symbols_used = result.get('symbols_used', [])
    weights = result.get('weights', {})
    # Display symbols and weights clearly
    st.caption(f"Simulated with: {', '.join(symbols_used)}. Weights: " +
               ", ".join([f"{s}: {w*100:.1f}%" for s, w in weights.items()]))

    # Display performance metrics
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
        st.metric("Annualized Volatility", f"{result['annualized_volatility']:.2%}",
                  help="Standard deviation of returns (risk measure).")
    with col4:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}",
                  help="Risk-adjusted return (higher is generally better). Assumes 0% risk-free rate.")

    st.markdown("---")

    # Display charts in columns
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.write("**Portfolio Value Over Time**")
        value_df = result['cumulative_value']
        # Ensure index is datetime before plotting
        if pd.api.types.is_datetime64_any_dtype(value_df.index):
            fig_value = px.area(value_df, title=None, labels={
                                'value': 'Portfolio Value ($)', 'index': 'Date'}, height=350)
            # Use primary color from config
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
            # Create DataFrame and sort for consistent pie chart slices
            weights_df = pd.DataFrame(list(weights.items()), columns=[
                                      'Stock', 'Weight']).sort_values('Weight', ascending=False)
            # Filter out tiny weights for clarity
            # Threshold for display
            weights_df = weights_df[weights_df['Weight'] > 0.001]

            if not weights_df.empty:
                fig_weights = px.pie(
                    weights_df, values='Weight', names='Stock', hole=0.3, height=350, title=None)
                # Improve text labels
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
# (display_correlation_analysis function remains the same)
def display_correlation_analysis(matrix: Optional[pd.DataFrame]):
    logger.info("Displaying correlation analysis.")
    st.subheader("Stock Correlation Analysis (1 Year Daily Returns)")
    if matrix is None or matrix.empty:
        st.warning("No correlation data available.")
        return
    if len(matrix.columns) < 2:
        st.warning("Correlation requires at least two stocks.")
        return

    # Create heatmap
    fig = px.imshow(
        matrix,
        labels=dict(color="Correlation Coefficient"),
        x=matrix.columns,
        y=matrix.index,
        template="plotly_dark",
        color_continuous_scale='RdBu',  # Red-Blue scale for correlation
        zmin=-1, zmax=1,  # Ensure full correlation range
        text_auto='.2f',  # Show values on heatmap
        aspect="auto",  # Adjust aspect ratio
        # Dynamic height based on number of stocks
        height=max(400, len(matrix.columns)*50)
    )
    # Customize hover text
    fig.update_traces(
        hovertemplate='Correlation(%{x}, %{y}) = %{z:.2f}<extra></extra>')
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_tickangle=-45  # Angle labels if many stocks
    )
    st.plotly_chart(fig, use_container_width=True, key='correlation_heatmap')
    st.caption("Correlation measures how stock returns move relative to each other (+1: perfect positive, -1: perfect negative, 0: no linear relationship).")


# --- ML Prediction Display ---
# (display_ml_prediction function remains the same as previous fix)
def display_ml_prediction(results: Optional[Dict[str, Any]], symbol: str):
    """Displays results for selected ML models, handling predicted returns."""
    logger.info(f"Displaying ML prediction results for {symbol}.")
    st.subheader(f"Machine Learning Price Prediction (Illustrative)")

    # Check if results exist and are not empty
    if not results:
        st.warning(
            "Machine Learning prediction results are not available (no models run or failed).")
        return

    model_names = list(results.keys())
    if not model_names:
        st.warning("No ML models were selected or processed successfully.")
        return

    # Create tabs for each model
    model_tabs = st.tabs(model_names)

    for i, model_name in enumerate(model_names):
        with model_tabs[i]:
            model_result = results[model_name]
            st.markdown(f"#### Results for: {model_name}")

            # Check if this specific model encountered an error
            if model_result.get("error"):
                st.error(
                    f"Failed to generate results for {model_name}: {model_result['error']}")
                continue  # Skip to the next model tab

            # Extract results for this model
            pred_price = model_result.get('next_day_prediction_price')
            pred_return = model_result.get('next_day_prediction_return')
            last_actual = model_result.get('last_actual_close')
            rmse_returns = model_result.get('rmse_returns')
            feature_importance = model_result.get('feature_importance')
            # These should be Pandas Series with DateTimeIndex now
            y_test_actual_prices = model_result.get('y_test_actual_prices')
            predictions_prices = model_result.get('predictions_prices')
            features_used = model_result.get('features_used', [])

            # Display Prediction Metrics
            st.markdown("**Prediction & Model Performance**")
            col1, col2 = st.columns(2)
            with col1:
                delta_str, delta_help = "N/A", ""
                # Calculate delta vs last actual close if possible
                if pred_price is not None and last_actual is not None and last_actual != 0:
                    delta = (pred_price / last_actual) - 1
                    delta_str = f"{delta:.2%}"
                    delta_help = f"Change vs last actual close of ${last_actual:.2f}"

                st.metric("Predicted Next Day Close", f"${pred_price:.2f}" if pred_price is not None else "N/A",
                          delta=delta_str if delta_str != "N/A" else None,
                          help=f"{model_name}: Model's price forecast. {delta_help}")
                if pred_return is not None:
                    # Show predicted return
                    st.caption(f"Predicted Return: {pred_return:.4%}")
            with col2:
                # Display RMSE based on returns
                st.metric(f"{model_name} Test RMSE (Returns)", f"{rmse_returns:.6f}" if rmse_returns is not None else "N/A",
                          help=f"{model_name}: Typical prediction error for *daily returns* (lower is better).")

            st.markdown("---")

            # Display Plots
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.write("**Model Performance on Test Data (Prices)**")
                # --- Simplified Plotting Logic ---
                fig = go.Figure()
                plot_successful = False
                # Check if both actual and predicted prices are Pandas Series with index
                if isinstance(y_test_actual_prices, pd.Series) and isinstance(predictions_prices, pd.Series):
                    # Ensure the indices match (important!)
                    if y_test_actual_prices.index.equals(predictions_prices.index) and not y_test_actual_prices.empty:
                        fig.add_trace(go.Scatter(
                            x=y_test_actual_prices.index,  # Use index directly
                            y=y_test_actual_prices.values,
                            mode='lines', name='Actual',
                            line=dict(color=config.PRIMARY_COLOR, width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=predictions_prices.index,  # Use index directly
                            y=predictions_prices.values,
                            mode='lines', name=f'{model_name} Predicted',
                            line=dict(color=config.SECONDARY_COLOR,
                                      dash='dot', width=2)
                        ))
                        plot_successful = True
                    else:
                        # Log and warn if indices don't match or data is empty
                        if y_test_actual_prices.empty:
                            logger.warning(
                                f"Plotting skipped for {model_name}: Actual prices Series is empty.")
                        elif not y_test_actual_prices.index.equals(predictions_prices.index):
                            logger.warning(
                                f"Plotting skipped for {model_name}: Index mismatch between actual ({len(y_test_actual_prices.index)}) and predicted ({len(predictions_prices.index)}) prices.")
                            logger.debug(
                                f"Actual Index Head: {y_test_actual_prices.index[:5]}")
                            logger.debug(
                                f"Predicted Index Head: {predictions_prices.index[:5]}")
                        st.warning(
                            f"Index mismatch or empty data for {model_name}. Cannot plot performance.")
                else:
                    # Log and warn if data is not in the expected Series format
                    logger.warning(
                        f"Plotting skipped for {model_name}: Data not in expected Pandas Series format.")
                    st.warning(
                        f"ML plot data for {model_name} is not in the expected format. Cannot plot.")

                if plot_successful:
                    # Update layout only if plotting was successful
                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="top",
                                    y=1.15, xanchor="left", x=0)
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"ml_test_plot_{model_name}")
                # --- End Simplified Plotting Logic ---

            with plot_col2:
                # Display Feature Importance/Coefficients
                if feature_importance is not None and not feature_importance.empty:
                    st.write("**Feature Influence**")
                    st.caption(
                        f"Top features influencing {model_name}. Used: {', '.join(features_used)}")
                    # Determine column name ('Importance' for tree models, 'Coefficient' for linear)
                    importance_col = 'Importance' if 'Importance' in feature_importance.columns else 'Coefficient'
                    # Create bar chart
                    fig_feat = px.bar(feature_importance.head(10),  # Show top 10
                                      x=importance_col, y='Feature', orientation='h',
                                      height=350, text_auto='.3f',
                                      labels={importance_col: f'Relative {importance_col}', 'Feature': ''})
                    # Order bars appropriately
                    # Use ascending for coefficients too based on absolute value sorting in predictor
                    yaxis_order = 'total ascending' if importance_col == 'Importance' else 'total ascending'
                    fig_feat.update_layout(template="plotly_dark", yaxis={'categoryorder': yaxis_order},
                                           margin=dict(l=10, r=10, t=30, b=20))
                    # Use consistent color
                    fig_feat.update_traces(marker_color=config.PRIMARY_COLOR)
                    st.plotly_chart(fig_feat, use_container_width=True,
                                    key=f"ml_feature_importance_{model_name}")
                else:
                    st.warning(
                        f"Feature importance/coefficients not available for {model_name}.")

    st.caption(
        "ML predictions predict daily returns & reconstruct price. Experimental only. Not investment advice.")


# --- ESG Scores Display ---
# (display_esg_scores function remains the same)
# Changed type hint for scores
def display_esg_scores(scores: Optional[Dict[str, Any]], symbol: str):
    logger.info(f"Displaying ESG scores for {symbol}.")
    st.subheader(f"ESG Performance for {symbol}")
    if scores is None:  # Check for None explicitly
        st.info(
            f"ESG scores unavailable or could not be retrieved for {symbol}.")
        return
    if not scores:  # Check for empty dictionary
        st.info(
            f"No specific ESG scores found for {symbol}, although sustainability data might exist.")
        return

    # Prepare data for plotting, handling potential non-numeric values if any slipped through
    esg_data = []
    # Map standard keys to display labels and extract scores
    score_map_display = {
        'Total ESG Score': scores.get('Total ESG Score'),
        'Environmental Score': scores.get('Environmental Score'),
        'Social Score': scores.get('Social Score'),
        'Governance Score': scores.get('Governance Score'),
        # Often a level, not score
        'Highest Controversy': scores.get('Highest Controversy')
    }

    for label, value in score_map_display.items():
        # Ensure value is numeric before adding
        if value is not None and pd.api.types.is_number(value) and pd.notna(value):
            esg_data.append({'Category': label, 'Score': float(value)})
        # Optionally handle non-numeric controversy level differently if needed
        # elif label == 'Highest Controversy' and value is not None:
        #     st.caption(f"Highest Controversy Level: {value}") # Display separately?

    if not esg_data:
        st.info("Could not extract formatted ESG scores for plotting.")
        return

    # Create Bar Chart
    esg_df = pd.DataFrame(esg_data)
    fig = px.bar(esg_df, x='Category', y='Score', text='Score', color='Category',
                 # Clarify axis label
                 labels={'Score': 'Score (Lower is often better for Risk)'})
    # Format text on bars
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    # Update layout
    fig.update_layout(template="plotly_dark", xaxis_title="", height=400,
                      margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="esg_chart")
    st.caption("ESG scores typically from providers like Sustainalytics via yfinance. Lower scores often indicate lower unmanaged ESG risk.")


# --- Earnings Calendar Display ---
# (display_earnings_calendar function remains the same)
def display_earnings_calendar(calendar_df: Optional[pd.DataFrame], symbol: str):
    logger.info(f"Displaying earnings calendar for {symbol}.")
    st.subheader(f"Earnings Calendar for {symbol}")
    if calendar_df is None or calendar_df.empty:
        st.info(f"No earnings data found or available for {symbol}.")
        return

    display_df = calendar_df.copy()

    # Ensure 'Earnings Date' is datetime and handle errors
    display_df['Earnings Date'] = pd.to_datetime(
        display_df['Earnings Date'], errors='coerce')
    # Drop rows where date conversion failed
    display_df = display_df.dropna(subset=['Earnings Date'])

    # Filter for recent past and near future dates for relevance
    today = pd.Timestamp.today().normalize()
    display_df = display_df[
        # Look back 90 days
        (display_df['Earnings Date'] >= today - pd.Timedelta(days=90)) &
        (display_df['Earnings Date'] <= today +
         pd.Timedelta(days=180))  # Look forward 180 days
    ].sort_values('Earnings Date')

    if display_df.empty:
        st.info(f"No recent or upcoming earnings dates found in the typical range.")
        return

    # Format date for display
    display_df['Earnings Date'] = display_df['Earnings Date'].dt.strftime(
        '%b %d, %Y')

    # Format numeric columns safely, showing 'N/A' or '-' for missing values
    currency_cols = ['Earnings Average', 'Earnings Low', 'Earnings High']
    revenue_cols = ['Revenue Average', 'Revenue Low', 'Revenue High']

    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
    for col in revenue_cols:
        if col in display_df.columns:
            # Format large revenue numbers in millions/billions for readability
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e9:,.1f}B" if pd.notna(x) and abs(x) >= 1e9 else (
                f"${x/1e6:,.1f}M" if pd.notna(x) and abs(x) >= 1e6 else (f"${x:,.0f}" if pd.notna(x) else '-')))

    # Select and order columns for display
    potential_cols = ['Earnings Date', 'Earnings Average', 'Revenue Average',
                      'Earnings Low', 'Earnings High', 'Revenue Low', 'Revenue High']
    display_cols = [c for c in potential_cols if c in display_df.columns]

    # Set Date as index for better table display
    if 'Earnings Date' in display_cols:
        display_df = display_df.set_index('Earnings Date')
        # Remove from list of columns to display
        display_cols.remove('Earnings Date')

    # Display the DataFrame
    # Display only remaining columns
    st.dataframe(display_df[display_cols], use_container_width=True)
    st.caption("Recent/Upcoming earnings dates and estimates (Source: yfinance).")


# --- Dividend History Display ---
# (display_dividend_history function remains the same)
def display_dividend_history(dividends: Optional[pd.Series], symbol: str):
    logger.info(f"Displaying dividend history for {symbol}.")
    st.subheader(f"Dividend History for {symbol}")
    if dividends is None or dividends.empty:
        st.info(f"No dividend history found or available for {symbol}.")
        return

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(dividends.index):
        try:
            dividends.index = pd.to_datetime(dividends.index)
        except Exception as e:
            logger.warning(f"Could not parse dividend dates: {e}")
            st.warning(
                "Could not display dividend history due to date parsing issues.")
            return

    # Create plot
    fig = px.bar(
        x=dividends.index,
        y=dividends.values,
        labels={'x': 'Date', 'y': 'Dividend per Share ($)'},  # Clearer labels
        height=400
    )
    # Use positive color from config
    fig.update_traces(marker_color=config.POSITIVE_COLOR)
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key="dividend_history_chart")
    st.caption("Historical dividend payments per share (Source: yfinance).")


# --- Sector Performance Display ---
# (display_sector_performance function remains the same)
def display_sector_performance():
    """ Fetches and displays recent sector performance using ETFs. """
    logger.info("Fetching and displaying sector performance.")
    # SPDR Sector ETFs + S&P 500
    sectors = {
        "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials",
        "XLY": "Cons. Disc.", "XLC": "Comm. Svcs.", "XLI": "Industrials",
        "XLP": "Cons. Staples", "XLU": "Utilities", "XLRE": "Real Estate",
        "XLB": "Materials", "XLE": "Energy", "SPY": "S&P 500"
    }
    performance = {}
    # Use a standard period like '1mo' or '3mo'
    period = "1mo"  # Lookback period for performance calculation
    etf_symbols = list(sectors.keys())

    try:
        # Download data for the specified period
        data = yf.download(etf_symbols, period=period,
                           progress=False, auto_adjust=True)

        if data.empty or 'Close' not in data:
            raise ValueError(
                f"No valid sector ETF data for period '{period}'.")

        close_prices = data['Close']
        # Ensure enough data points for calculation
        if len(close_prices) < 2:
            raise ValueError(
                f"Insufficient data points ({len(close_prices)}) in period '{period}'.")

        # Calculate percentage change from the first to the last day in the period
        start_prices = close_prices.iloc[0]
        end_prices = close_prices.iloc[-1]
        pct_change = ((end_prices - start_prices) / start_prices) * 100

        # Map performance back to sector names
        for etf, sector_name in sectors.items():
            if etf in pct_change.index and pd.notna(pct_change[etf]):
                performance[sector_name] = pct_change[etf]
            else:
                logger.warning(
                    f"Data missing or NaN for {etf} ({sector_name}) in period '{period}'.")
                # Mark as None if data is missing/NaN
                performance[sector_name] = None

    except Exception as e:
        logger.error(
            f"Error fetching/processing sector data: {e}", exc_info=True)
        st.warning("Could not retrieve sector performance data at this time.")
        return

    # Filter out sectors where calculation failed
    valid_performance = {k: v for k, v in performance.items() if v is not None}
    if not valid_performance:
        st.warning(
            f"No valid sector performance data could be calculated for period '{period}'.")
        return

    # Create DataFrame and plot
    df_perf = pd.DataFrame(list(valid_performance.items()), columns=[
                           "Sector/Index", f"{period} % Change"])
    df_perf = df_perf.sort_values(
        f"{period} % Change", ascending=False)  # Sort by performance

    fig = px.bar(
        df_perf,
        x="Sector/Index",
        y=f"{period} % Change",
        title=f"Sector & S&P 500 Performance ({period})",
        color=f"{period} % Change",
        color_continuous_scale='RdYlGn',  # Red -> Yellow -> Green scale
        labels={f"{period} % Change": "% Change"},
        height=400
    )
    # Update layout for better readability
    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45,  # Angle labels
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True,
                    key="sector_perf_chart_v2")  # Use unique key
