# src/plotting.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import yfinance as yf
from src import config, utils

# --- Main Price Chart & Technicals ---


def plot_stock_data(df: pd.DataFrame, symbol: str, indicators: Dict[str, bool], patterns: List[str]):
    """ Plots the main stock chart with OHLC, Volume, selected indicators and lists patterns. """
    utils.logger.info(f"Plotting stock data for {symbol}")
    st.subheader("Price Chart & Technical Analysis")
    if df is None or df.empty:
        st.warning("No historical data available to plot.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,  # Slightly more spacing
                        row_heights=[0.7, 0.3],  # Main chart larger
                        specs=[[{"secondary_y": True}],  # Row 1: Price + Volume
                               # Row 2: Indicators (RSI/MACD)
                               [{"secondary_y": False}]])

    # --- Row 1: Price and Volume ---
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name=f'{symbol} Price',
        increasing_line_color=config.POSITIVE_COLOR, decreasing_line_color=config.NEGATIVE_COLOR
    ), row=1, col=1, secondary_y=False)

    # Volume
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'], name='Volume',
        marker_color=config.VOLUME_COLOR,
    ), row=1, col=1, secondary_y=True)

    # SMA Lines (on Price chart)
    if indicators.get('show_sma'):
        for window in config.SMA_PERIODS:
            sma_col = f'SMA_{window}'
            if sma_col in df.columns and df[sma_col].notna().any():
                color = config.SMA_COLORS.get(
                    sma_col, '#cccccc')  # Use config colors
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df[sma_col], mode='lines',
                    line=dict(color=color, width=1.5), name=f'SMA {window}'
                ), row=1, col=1, secondary_y=False)

    # Bollinger Bands (on Price chart)
    if indicators.get('show_bollinger'):
        bb_upper_col, bb_lower_col, bb_mid_col = 'BB_Upper', 'BB_Lower', 'BB_Middle'
        if bb_upper_col in df.columns and df[bb_upper_col].notna().any():
            # Mid Band (optional, usually same as SMA_20)
            # if bb_mid_col in df.columns and df[bb_mid_col].notna().any():
            #    fig.add_trace(go.Scatter(x=df['Date'], y=df[bb_mid_col], line=dict(color=config.BB_BAND_COLOR, width=1, dash='dot'), name='BB Middle'), row=1, col=1, secondary_y=False)
            # Upper Band
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[bb_upper_col], mode='lines',
                line=dict(color=config.BB_BAND_COLOR, width=1, dash='dash'),
                name='BB Upper'
            ), row=1, col=1, secondary_y=False)
            # Lower Band (with fill)
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
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'], name='RSI',
            line=dict(color=config.PRIMARY_COLOR)
        ), row=2, col=1)
        # Add Overbought/Oversold lines
        fig.add_hline(y=70, line_dash='dash',
                      line_color='rgba(200, 0, 0, 0.5)', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash',
                      line_color='rgba(0, 150, 0, 0.5)', row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        added_row2 = True

    if indicators.get('show_macd') and 'MACD' in df.columns and df['MACD'].notna().any():
        macd_col, signal_col, hist_col = 'MACD', 'Signal_Line', 'MACD_Histogram'
        # Determine which row to plot MACD on (row 2 if RSI is off, else might need row 3 or overlap)
        # For simplicity, let's assume it shares row 2 if RSI is also shown, or uses row 2 if RSI is off.
        # A more robust solution might create rows dynamically.
        macd_row = 2
        # MACD Line
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[macd_col], name='MACD', line=dict(color=config.SECONDARY_COLOR)
        ), row=macd_row, col=1)
        # Signal Line
        if signal_col in df.columns and df[signal_col].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[signal_col], name='Signal Line', line=dict(color=config.PRIMARY_COLOR, dash='dot')
            ), row=macd_row, col=1)
        # Histogram
        if hist_col in df.columns and df[hist_col].notna().any():
            hist_colors = [config.POSITIVE_COLOR if v >=
                           0 else config.NEGATIVE_COLOR for v in df[hist_col]]
            fig.add_trace(go.Bar(
                x=df['Date'], y=df[hist_col], name='Histogram', marker_color=hist_colors
            ), row=macd_row, col=1)
        # Update title for the correct row
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
        added_row2 = True  # Mark row 2 as used

    # --- Layout Updates ---
    fig.update_layout(
        template="plotly_dark",
        height=650 if added_row2 else 500,  # Adjust height based on whether row 2 is used
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font_color='white'),
        margin=dict(l=50, r=20, t=30, b=50),  # Adjust margins
        xaxis_rangeslider_visible=False,
        xaxis_showticklabels=True,  # Show ticks on top x-axis (price)
        xaxis2_showticklabels=True,  # Show ticks on bottom x-axis (indicators)
        yaxis=dict(title="Price (USD)", showgrid=True,
                   gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title="Volume", showgrid=False, overlaying='y',
                    side='right', showticklabels=False),
        # Style for indicator axis (row 2)
        yaxis3=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        # If MACD and RSI were split into rows 2 and 3, you'd need yaxis4 etc.
    )
    # Hide indicator row if not used
    if not added_row2:
        fig.update_layout(height=500, xaxis2_visible=False,
                          yaxis3_visible=False)
        fig.layout.rows = 1  # Adjust layout definition
        fig.layout.row_heights = [1.0]

    st.plotly_chart(fig, use_container_width=True,
                    key="main_stock_chart_indicators")

    # --- Technical Patterns Display ---
    st.subheader("Technical Pattern Analysis")
    if patterns:
        num_patterns = len(patterns)
        # Use columns for better layout
        cols = st.columns(min(num_patterns, 3))  # Max 3 columns
        for i, pattern in enumerate(patterns):
            with cols[i % 3]:
                # Use info/warning/error based on pattern type if desired
                st.info(f"📌 {pattern}")
    else:
        st.info(
            "No significant standard technical patterns detected in the recent data.")
    st.caption(
        "Pattern analysis is based on standard indicator readings and is illustrative, not investment advice.")

# --- Company Info Display ---


def display_company_info(info: Optional[Dict[str, Any]], show_tooltips: bool):
    utils.logger.info(
        f"Displaying company info for {info.get('symbol', 'N/A') if info else 'N/A'}")
    if not info:
        st.warning("Company information is unavailable.")
        return

    st.subheader(
        f"Company Overview: {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")

    # Key Metrics Row
    cols_metrics = st.columns(4)
    with cols_metrics[0]:
        value = info.get('currentPrice') or info.get(
            'regularMarketPrice') or info.get('previousClose')
        st.metric("Last Price", f"${value:,.2f}" if pd.notna(
            value) else "N/A", help="Most recent available price.")
    with cols_metrics[1]:
        mkt_cap = info.get('marketCap')
        st.metric("Market Cap", f"${mkt_cap / 1e9:.2f}B" if mkt_cap and mkt_cap > 1e9 else (
            f"${mkt_cap / 1e6:.2f}M" if mkt_cap and mkt_cap > 1e6 else (f"${mkt_cap:,}" if mkt_cap else "N/A")), help="Total market value.")
    with cols_metrics[2]:
        st.metric("Sector", info.get('sector', 'N/A'),
                  help="Company's sector.")
    with cols_metrics[3]:
        st.metric("Industry", info.get('industry', 'N/A'),
                  help="Company's industry.")

    st.markdown("---")

    # Business Summary
    st.subheader("Business Summary")
    summary = info.get('longBusinessSummary', 'No business summary available.')
    st.info(summary)

    st.markdown("---")

    # Financial Metrics Expander
    with st.expander("Key Financial Metrics & Ratios", expanded=False):
        metric_definitions = {
            'trailingPE': ("Trailing P/E", "Price/Earnings (past 12m).", ".2f"), 'forwardPE': ("Forward P/E", "Price/Earnings (est. next year).", ".2f"),
            'priceToBook': ("Price/Book (P/B)", "Market value vs. book value.", ".2f"), 'priceToSales': ("Price/Sales (P/S)", "Market value vs. revenue.", ".2f"),
            'enterpriseValue': ("Enterprise Value", "Total company value (market cap + debt - cash).", ",.0f"), 'enterpriseToRevenue': ("EV/Revenue", "EV vs. total revenue.", ".2f"), 'enterpriseToEbitda': ("EV/EBITDA", "EV vs. EBITDA.", ".2f"),
            'dividendYield': ("Dividend Yield", "Annual dividend / share price.", ".2%"), 'payoutRatio': ("Payout Ratio", "% of earnings paid as dividends.", ".2%"), 'fiveYearAvgDividendYield': ("5Y Avg Div Yield", "Avg. dividend yield (5y).", ".2f"),
            'profitMargins': ("Profit Margin", "Net income / revenue.", ".2%"), 'grossMargins': ("Gross Margin", "Gross profit / revenue.", ".2%"), 'operatingMargins': ("Operating Margin", "Operating income / revenue.", ".2%"),
            'returnOnEquity': ("Return on Equity (ROE)", "Net income / shareholder equity.", ".2%"), 'returnOnAssets': ("Return on Assets (ROA)", "Net income / total assets.", ".2%"),
            'trailingEps': ("Trailing EPS", "Earnings per share (past 12m).", ".2f"), 'forwardEps': ("Forward EPS", "Est. earnings per share (next year).", ".2f"),
            'beta': ("Beta", "Volatility vs. market (S&P 500).", ".2f"),
            'fiftyTwoWeekHigh': ("52 Week High", "Highest price (past 52w).", ",.2f"), 'fiftyTwoWeekLow': ("52 Week Low", "Lowest price (past 52w).", ",.2f"),
            'volume': ("Volume", "Shares traded (latest session).", ",.0f"), 'averageVolume': ("Avg Volume (10 Day)", "Avg daily volume (10d).", ",.0f"),
        }

        metrics_data = {}
        for key, (label, tooltip, fmt) in metric_definitions.items():
            value = info.get(key)
            # Include 0 for some ratios like payout
            if value is not None and pd.notna(value):
                try:
                    if fmt == ".2%":
                        value_str = f"{value:.2%}"
                    elif fmt == ",.0f":
                        value_str = f"${value:,.0f}" if key == 'enterpriseValue' else f"{value:,.0f}"
                    elif fmt == ",.2f":
                        value_str = f"${value:,.2f}" if key in [
                            'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'trailingEps', 'forwardEps'] else f"{value:.2f}"
                    else:
                        value_str = str(value)
                    metrics_data[label] = (value_str, tooltip)
                except (ValueError, TypeError):
                    metrics_data[label] = ("Error", tooltip)

        if not metrics_data:
            st.write("No detailed financial metrics available.")
        else:
            cols_per_row = 4  # More compact layout
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
                            if show_tooltips:
                                st.metric(
                                    label=label, value=value_str, help=tooltip)
                            else:
                                st.metric(label=label, value=value_str)

# --- Sentiment Analysis Display ---


def display_sentiment_analysis(articles: List[Dict[str, Any]], avg_sentiment: float, sentiment_df: Optional[pd.DataFrame]):
    utils.logger.info("Displaying sentiment analysis.")
    st.subheader("News & Sentiment Analysis")  # Changed header level

    if not articles and (sentiment_df is None or sentiment_df.empty):
        st.warning("No news articles or sentiment data found for analysis.")
        return

    # Sentiment Score Card & Trend
    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            sentiment_color_class = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
            sentiment_label = "Positive 😊" if avg_sentiment > 0.05 else "Negative 😞" if avg_sentiment < - \
                0.05 else "Neutral 😐"
            st.metric("Avg. News Sentiment", f"{avg_sentiment:.2f}", delta=sentiment_label,
                      help="VADER Compound Score (-1 Negative to +1 Positive)")
            # Simple HTML card version (can be used if metric delta isn't desired)
            # st.markdown(f"""<div class='metric-card' style='text-align: center;'>...</div>""", unsafe_allow_html=True)

        with col2:
            if sentiment_df is not None and not sentiment_df.empty:
                st.markdown(
                    "<p style='font-weight: bold; text-align: center; margin-bottom: 0.5em;'>Daily Sentiment Trend (VADER)</p>", unsafe_allow_html=True)
                fig = px.area(sentiment_df, x=sentiment_df.index, y='Daily_Sentiment', height=200,
                              # Use index name
                              labels={'Daily_Sentiment': 'Avg. Score', 'index': 'Date'})

                line_color = config.POSITIVE_COLOR if avg_sentiment > 0.05 else config.NEGATIVE_COLOR if avg_sentiment < - \
                    0.05 else config.NEUTRAL_COLOR
                fill_color = line_color.replace(')', ', 0.1)').replace(
                    'rgb', 'rgba')  # Make fill transparent
                fig.update_traces(line_color=line_color, fillcolor=fill_color)
                fig.add_hline(y=0, line_dash='dash', line_color='grey')
                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=5, b=10), showlegend=False,
                                  xaxis_showgrid=False, yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                st.plotly_chart(fig, use_container_width=True,
                                key="sentiment_trend_chart")
            else:
                st.markdown(
                    "<div class='metric-card'><p style='text-align: center; padding: 3rem 0;'>No daily sentiment trend data.</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # News Articles Display
    st.subheader("Recent News Headlines")
    if not articles:
        st.info("No recent news articles found in the selected period.")
        return

    for i, article in enumerate(articles[:10]):  # Limit displayed articles
        sentiment = article.get('sentiment', {})
        compound_score = sentiment.get('compound', 0)
        sentiment_color = config.POSITIVE_COLOR if compound_score > 0.05 else config.NEGATIVE_COLOR if compound_score < - \
            0.05 else config.NEUTRAL_COLOR
        sentiment_label = f"{compound_score:.2f}"

        source = article.get('source', {}).get('name', 'Unknown Source')
        published_at_str = article.get('publishedAt', '')
        published_date = "Unknown Date"
        if published_at_str:
            try:
                published_date = pd.to_datetime(
                    published_at_str).strftime('%b %d, %Y %H:%M')
            except Exception:
                published_date = published_at_str  # Fallback

        # Use st.expander for a cleaner look
        with st.expander(f"{article.get('title', 'No Title Available')} ({source})"):
            st.caption(
                f"Published: {published_date} | Sentiment: {sentiment_label}")
            st.markdown(f"_{article.get('description', 'No description.')}_")
            st.link_button("Read Article", article.get(
                'url', '#'), type="secondary")


# --- Prophet Forecast Display ---
def display_prophet_forecast(forecast_df: Optional[pd.DataFrame], historical_df: pd.DataFrame, symbol: str, days_predicted: int):
    utils.logger.info(f"Displaying Prophet forecast for {symbol}")
    st.subheader(f"Prophet Forecast ({days_predicted} Days)")

    if forecast_df is None or forecast_df.empty:
        st.warning("No forecast data available to display.")
        return
    if historical_df is None or historical_df.empty:
        st.warning("Historical data needed for context is missing.")
        return

    fig = go.Figure()
    # Historical Actual
    fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Close'],
                  mode='lines', name='Actual Price', line=dict(color=config.PRIMARY_COLOR, width=2)))
    # Forecast Line
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(
        color=config.SECONDARY_COLOR, width=2, dash='dash')))
    # Confidence Interval
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(
        width=0), showlegend=False, name='Upper Bound'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(
        # Orange fill
        width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', showlegend=False, name='Lower Bound'))

    # Highlight forecast period start
    start_forecast_date = forecast_df['ds'].min()
    fig.add_vline(x=start_forecast_date, line_width=1,
                  line_dash="dash", line_color="grey")

    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price (USD)", height=500, hovermode='x unified',
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.01, xanchor="left", x=0, font_color='white'),
                      margin=dict(l=50, r=20, t=30, b=20))

    st.plotly_chart(fig, use_container_width=True,
                    key="prophet_forecast_chart")

    # Display tail of forecast data
    st.write("Forecasted Values (Upcoming):")
    forecast_display = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'
    }).set_index('Date')
    # Show only future dates
    future_forecast = forecast_display[forecast_display.index >= pd.Timestamp.today(
    ).normalize()]
    st.dataframe(future_forecast.head(days_predicted).style.format(
        "${:.2f}"), use_container_width=True)
    st.caption("Forecasts are based on historical trends and seasonality (and optional sentiment regressors) via Prophet. Not investment advice.")


# --- Portfolio Simulation Display ---
def display_portfolio_analysis(result: Optional[Dict[str, Any]], initial_investment: float):
    utils.logger.info("Displaying portfolio simulation results.")
    st.subheader("Portfolio Simulation (1 Year Performance)")

    if not result:
        st.warning("Portfolio simulation data is not available or failed.")
        return

    symbols_used = result.get('symbols_used', [])
    weights = result.get('weights', {})
    st.caption(f"Simulated with: {', '.join(symbols_used)}. Weights: " +
               ", ".join([f"{s}: {w*100:.1f}%" for s, w in weights.items()]))

    # Performance Metrics
    final_val = result['cumulative_value'].iloc[-1]
    total_return = (final_val / initial_investment) - 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Value", f"${final_val:,.2f}",
                  f"{total_return:.2%}", help="Portfolio value after 1 year.")
    with col2:
        st.metric("Annualized Return",
                  f"{result['annualized_return']:.2%}", help="Avg. annual growth rate.")
    with col3:
        st.metric("Annualized Volatility",
                  f"{result['annualized_volatility']:.2%}", help="Risk measure (std. dev. of returns).")
    with col4:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}",
                  help="Risk-adjusted return (vs. 0% risk-free rate). Higher is better.")

    st.markdown("---")

    # Charts side-by-side
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
            st.warning("Could not plot portfolio value (index issue).")

    with col_chart2:
        st.write("**Portfolio Allocation**")
        if weights:
            weights_df = pd.DataFrame(list(weights.items()), columns=[
                                      'Stock', 'Weight']).sort_values('Weight', ascending=False)
            # Filter tiny weights for clarity
            weights_df = weights_df[weights_df['Weight'] > 0.001]
            fig_weights = px.pie(weights_df, values='Weight',
                                 names='Stock', hole=0.3, height=350, title=None)
            fig_weights.update_traces(
                textposition='inside', textinfo='percent+label', pull=[0.03]*len(weights_df))
            fig_weights.update_layout(
                template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_weights, use_container_width=True,
                            key="portfolio_weights_pie")
        else:
            st.warning("Weights data unavailable.")

# --- Correlation Analysis Display ---


def display_correlation_analysis(matrix: Optional[pd.DataFrame]):
    utils.logger.info("Displaying correlation analysis.")
    st.subheader("Stock Correlation Analysis (1 Year Daily Returns)")

    if matrix is None or matrix.empty:
        st.warning("No correlation data available or calculation failed.")
        return
    if len(matrix.columns) < 2:
        st.warning("Correlation matrix requires at least two stocks.")
        return

    fig = px.imshow(matrix, labels=dict(color="Correlation"), x=matrix.columns, y=matrix.index,
                    template="plotly_dark", color_continuous_scale='RdBu', zmin=-1, zmax=1, text_auto='.2f',
                    aspect="auto", height=max(400, len(matrix.columns)*50))
    fig.update_traces(
        hovertemplate='Correlation(%{x}, %{y}) = %{z:.2f}<extra></extra>')
    fig.update_layout(template="plotly_dark", margin=dict(
        l=10, r=10, t=30, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key='correlation_heatmap')
    st.caption(
        "Correlation: +1 = move together, -1 = move opposite, 0 = no linear relationship.")

# --- ML Prediction Display ---


def display_ml_prediction(results: Optional[Dict[str, Any]], symbol: str):
    utils.logger.info(f"Displaying ML prediction for {symbol}.")
    st.subheader(f"Random Forest Price Prediction (Illustrative)")

    if not results:
        st.warning(
            "Machine Learning prediction results are not available or failed.")
        return

    pred_value = results.get('next_day_prediction')
    last_actual = results.get('last_actual_close')
    rmse = results.get('rmse')
    feature_importance = results.get('feature_importance')
    test_dates = results.get('test_dates')
    y_test = results.get('y_test')
    predictions = results.get('predictions')
    features_used = results.get('features_used', [])

    st.markdown("**Prediction & Model Performance**")
    col1, col2 = st.columns(2)
    with col1:
        delta_str, delta_help = "N/A", ""
        if pred_value is not None and last_actual is not None and last_actual != 0:
            delta = (pred_value / last_actual) - 1
            delta_str = f"{delta:.2%}"
            delta_help = f"Change vs last actual close of ${last_actual:.2f}"
        st.metric("Predicted Next Day Close", f"${pred_value:.2f}" if pred_value is not None else "N/A",
                  delta=delta_str if delta_str != "N/A" else None,
                  help=f"Model's forecast for next close. {delta_help} (Illustrative only).")
    with col2:
        st.metric("Model Test RMSE", f"${rmse:.2f}" if rmse is not None else "N/A",
                  help="Typical prediction error ($) on test data (lower is better).")

    st.markdown("---")

    # Charts side-by-side
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        if test_dates is not None and y_test is not None and predictions is not None:
            st.write("**Model Performance on Test Data**")
            plot_df = pd.DataFrame(
                {'Actual': y_test, 'Predicted': predictions}, index=test_dates)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual', line=dict(
                color=config.PRIMARY_COLOR, width=2)))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted', line=dict(
                color=config.SECONDARY_COLOR, dash='dot', width=2)))
            fig.update_layout(template="plotly_dark", xaxis_title='Date', yaxis_title='Price (USD)', height=350, margin=dict(
                l=20, r=20, t=30, b=20), hovermode='x unified', legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True, key="ml_test_plot")
        else:
            st.warning("Could not plot model performance.")

    with col_chart2:
        if feature_importance is not None and not feature_importance.empty:
            st.write("**Feature Importance**")
            st.caption(
                f"Top features influencing the model. Used: {', '.join(features_used)}")
            fig_feat = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h',
                              height=350, text_auto='.3f', labels={'Importance': 'Relative Importance', 'Feature': ''})
            fig_feat.update_layout(template="plotly_dark", yaxis={
                                   'categoryorder': 'total ascending'}, margin=dict(l=10, r=10, t=30, b=20))
            fig_feat.update_traces(marker_color=config.PRIMARY_COLOR)
            st.plotly_chart(fig_feat, use_container_width=True,
                            key="ml_feature_importance")
        else:
            st.warning("Feature importance data not available.")
    st.caption("Machine learning predictions are experimental and depend heavily on features and model tuning. Not investment advice.")

# --- ESG Scores Display ---


def display_esg_scores(scores: Optional[Dict[str, float]], symbol: str):
    utils.logger.info(f"Displaying ESG scores for {symbol}.")
    st.subheader(f"ESG Performance for {symbol}")

    if scores is None:
        st.info(f"ESG scores are not available for {symbol} via yfinance.")
        return
    if not scores:
        st.info(
            f"No specific ESG scores found, although sustainability data might exist.")
        return

    # Prepare data for plotting (ensure standard keys if possible)
    esg_data = []
    score_map_display = {  # Map internal keys to display names if needed
        'Total ESG Score': scores.get('Total ESG Score'),
        'E Score': scores.get('Environmental Score'),
        'S Score': scores.get('Social Score'),
        'G Score': scores.get('Governance Score'),
        # Example if available
        'Controversy': scores.get('Highest Controversy')
    }
    for label, value in score_map_display.items():
        if value is not None and pd.notna(value):
            esg_data.append({'Category': label, 'Score': float(value)})

    if not esg_data:
        st.info("Could not extract formatted ESG scores.")
        return

    esg_df = pd.DataFrame(esg_data)

    # Use bar chart
    fig = px.bar(esg_df, x='Category', y='Score', text='Score',
                 color='Category',  # Color by category
                 labels={'Score': 'Score (Lower is often better)'})
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(template="plotly_dark", xaxis_title="", height=400, margin=dict(
        l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="esg_chart")
    st.caption("ESG scores from yfinance/Sustainalytics. Interpretation varies; lower scores often indicate lower unmanaged risk in that category.")

# --- Earnings Calendar Display ---


def display_earnings_calendar(calendar_df: Optional[pd.DataFrame], symbol: str):
    utils.logger.info(f"Displaying earnings calendar for {symbol}.")
    st.subheader(f"Earnings Calendar for {symbol}")

    if calendar_df is None or calendar_df.empty:
        st.info(f"No earnings data found for {symbol}.")
        return

    display_df = calendar_df.copy()
    # Ensure 'Earnings Date' is datetime before filtering/formatting
    display_df['Earnings Date'] = pd.to_datetime(
        display_df['Earnings Date'], errors='coerce')
    display_df.dropna(subset=['Earnings Date'], inplace=True)

    # Filter for recent past and upcoming dates (e.g., last 90 days to next 180 days)
    today = pd.Timestamp.today().normalize()
    display_df = display_df[
        (display_df['Earnings Date'] >= today - pd.Timedelta(days=90)) &
        (display_df['Earnings Date'] <= today + pd.Timedelta(days=180))
    ].sort_values('Earnings Date')

    if display_df.empty:
        st.info(
            f"No recent or near-term upcoming earnings dates found for {symbol}.")
        return

    # Formatting for display
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
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:,.1f}M" if pd.notna(
                # Format revenue
                x) and x > 1e6 else (f"${x:,.0f}" if pd.notna(x) else '-'))

    display_cols = [c for c in ['Earnings Date', 'Earnings Average',
                                'Revenue Average'] if c in display_df.columns]
    display_df.set_index('Earnings Date', inplace=True)

    # Show table without date index repeating
    st.dataframe(display_df[display_cols[1:]], use_container_width=True)
    st.caption("Recent/Upcoming earnings dates and estimates (Source: yfinance).")

# --- Dividend History Display ---


def display_dividend_history(dividends: Optional[pd.Series], symbol: str):
    utils.logger.info(f"Displaying dividend history for {symbol}.")
    st.subheader(f"Dividend History for {symbol}")

    if dividends is None or dividends.empty:
        st.info(f"No dividend history found for {symbol}.")
        return

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(dividends.index):
        try:
            dividends.index = pd.to_datetime(dividends.index)
        except Exception:
            st.warning("Could not parse dividend dates.")
            return

    # Create plot
    fig = px.bar(x=dividends.index, y=dividends.values, labels={
                 'x': 'Date', 'y': 'Dividend/Share ($)'}, height=400)
    fig.update_traces(marker_color=config.POSITIVE_COLOR)
    fig.update_layout(template="plotly_dark",
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True,
                    key="dividend_history_chart")
    st.caption("Historical dividend payments per share (Source: yfinance).")

# --- Sector Performance Display ---
# (Moved from static method in original class)


def display_sector_performance():
    utils.logger.info("Fetching and displaying sector performance.")
    sectors = {
        "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials",
        "XLY": "Cons. Disc.", "XLC": "Comm. Svcs.", "XLI": "Industrials",
        "XLP": "Cons. Staples", "XLU": "Utilities", "XLRE": "Real Estate",
        "XLB": "Materials", "XLE": "Energy", "SPY": "S&P 500"  # Add SPY for context
    }
    performance = {}
    # Define lookback period (adjust as needed)
    period = "1mo"  # Use yfinance period strings

    etf_symbols = list(sectors.keys())

    try:
        # Use history for more reliable start/end alignment
        data = yf.download(etf_symbols, period=period,
                           progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data:
            raise ValueError("No valid sector ETF data downloaded.")

        close_prices = data['Close']
        if len(close_prices) < 2:
            raise ValueError(
                "Insufficient data points for performance calculation.")

        # Calculate performance using first and last available close prices
        start_prices = close_prices.iloc[0]
        end_prices = close_prices.iloc[-1]
        pct_change = ((end_prices - start_prices) / start_prices) * 100

        # Map results back to sector names
        for etf, sector_name in sectors.items():
            if etf in pct_change.index and pd.notna(pct_change[etf]):
                performance[sector_name] = pct_change[etf]
            else:
                utils.logger.warning(
                    f"Data missing or NaN for sector ETF {etf} in period {period}.")
                performance[sector_name] = None

    except Exception as e:
        utils.logger.error(
            f"Error fetching/processing sector data: {e}", exc_info=True)
        st.warning("Could not retrieve sector performance data.")
        return

    valid_performance = {k: v for k, v in performance.items() if v is not None}
    if not valid_performance:
        st.warning(
            f"No valid sector performance data calculated for period '{period}'.")
        return

    df_perf = pd.DataFrame(list(valid_performance.items()), columns=[
                           "Sector/Index", f"{period} % Change"])
    df_perf = df_perf.sort_values(f"{period} % Change", ascending=False)

    fig = px.bar(df_perf, x="Sector/Index", y=f"{period} % Change",
                 title=f"Sector & S&P 500 Performance ({period})",
                 color=f"{period} % Change", color_continuous_scale='RdYlGn',
                 labels={f"{period} % Change": "% Chg"}, height=400)
    fig.update_layout(template="plotly_dark", xaxis_tickangle=-
                      45, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True, key="sector_perf_chart_v2")
