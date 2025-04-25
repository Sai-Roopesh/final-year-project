import yfinance as yf
import pandas as pd
# Optional: Configure pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(f"Testing yfinance version: {yf.__version__}")  # Check version
ticker = yf.Ticker("AAPL")
try:
    news = ticker.get_news()
    print("\n--- AAPL News ---")
    print(f"Type: {type(news)}")
    print("Content:")
    print(news)
except Exception as e:
    print(f"Error fetching news for AAPL: {e}")

ticker_googl = yf.Ticker("GOOGL")
try:
    news_googl = ticker_googl.get_news()
    print("\n--- GOOGL News ---")
    print(f"Type: {type(news_googl)}")
    print("Content:")
    print(news_googl)
except Exception as e:
    print(f"Error fetching news for GOOGL: {e}")
