# src/config.py

import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Load from .env file in the parent directory if it exists
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- API Keys ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
GEMINI_CHAT_MODEL = "models/gemini-1.5-flash-latest"
FRED_API_KEY = os.getenv("FRED_API_KEY")

# --- Application Constants ---
LOG_DIRECTORY = 'logs'
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_STOCK_SYMBOL = "AAPL"
DEFAULT_COMPANY_NAME = "Apple Inc."
DEFAULT_ECONOMIC_SERIES = ['GDP', 'CPIAUCNS', 'FEDFUNDS', 'UNRATE']
MIN_INVESTMENT = 1000
DEFAULT_INVESTMENT = 10000
DEFAULT_PORTFOLIO_STOCKS = "AAPL, GOOGL, MSFT"
DEFAULT_CORRELATION_STOCKS = "AAPL, GOOGL, MSFT"
MAX_CHAT_TOKENS = 2048
APP_TITLE = "Advanced Stock Insights Dashboard"
APP_ICON = "🌃"

# --- Technical Indicator Defaults ---
RSI_PERIOD = 14
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]  # Used internally for MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD_DEV = 2
VOLATILITY_WINDOW = 20
ATR_WINDOW = 14

# --- Machine Learning Defaults ---
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5
ML_FEATURE_DEFAULT = ['Open', 'High', 'Low', 'Close', 'Volume']
ML_TEST_SIZE = 0.2
ML_TRAINING_PERIOD = '5y'  # Default period to fetch data for ML

# --- News Defaults ---
NEWS_API_PAGE_SIZE = 50
NEWS_DAYS_LIMIT = 30  # Max days back for NewsAPI free tier
# Example: Filter news slightly based on sentiment relevance
NEWS_RELEVANCY_THRESHOLD = 0.1

# --- Forecasting Defaults ---
PROPHET_INTERVAL_WIDTH = 0.95
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.005  # Default value

# --- Plotting ---
POSITIVE_COLOR = '#4CAF50'  # Brighter Green
NEGATIVE_COLOR = '#F44336'  # Brighter Red
NEUTRAL_COLOR = '#90A4AE'  # Blue Grey
PRIMARY_COLOR = '#1E90FF'  # Dodger Blue
SECONDARY_COLOR = '#FFA500'  # Orange
VOLUME_COLOR = 'rgba(100, 100, 180, 0.3)'
BB_BAND_COLOR = '#1565c0'  # Blue
BB_FILL_COLOR = 'rgba(21, 101, 192, 0.1)'
SMA_COLORS = {'SMA_20': '#ffa726', 'SMA_50': '#fb8c00',
              'SMA_200': '#ef6c00'}  # Orange shades


# --- Streamlit Theme CSS (Dark) ---
# (Keep the CSS string here or load from a separate file if it gets too long)
DARK_THEME_CSS = """
<style>
    /* --- Main App Dark Theme --- */
    html, body, [class*="st-"] { color: #E0E0E0; }
    .stApp { background-color: #1C1C1E; }
    .main .block-container { padding: 2rem 1.5rem; }
    /* --- Headers --- */
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF; }
    .gradient-header {
        background: linear-gradient(45deg, #3a418a, #2d67a1); color: white;
        padding: 1.2rem 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .gradient-header h1, .gradient-header h2, .gradient-header h3 { margin: 0; color: white !important; }
    /* --- Cards & Containers --- */
    .metric-card {
        background: #2E2E2E; padding: 1rem 1.2rem; border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3); margin-bottom: 1rem;
        border: 1px solid #444444; height: 100%; color: #E0E0E0;
    }
    .stExpander {
        border: 1px solid #444444; border-radius: 8px; background: #2E2E2E;
        margin-bottom: 1rem; color: #E0E0E0;
    }
    .stExpander header {
        font-weight: bold; background-color: #3a3a3c; border-radius: 8px 8px 0 0;
        color: #FFFFFF !important;
    }
    .stExpander header:hover { background-color: #4a4a4c; }
    /* --- Metrics --- */
    .stMetric { border-radius: 6px; }
    .stMetric > label { font-weight: bold; color: #AAAAAA; }
    .stMetric > div[data-testid="stMetricValue"] { font-size: 1.6em; font-weight: 600; color: #FFFFFF; }
    .stMetric > div[data-testid="stMetricDelta"] { font-size: 0.9em; }
    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] { gap: 18px; border-bottom: 1px solid #444444; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; white-space: pre-wrap; background-color: transparent;
        border-radius: 8px 8px 0 0; gap: 8px; color: #AAAAAA;
        border-bottom: 2px solid transparent; margin-bottom: -1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E2E2E; font-weight: bold; color: #FFFFFF;
        border-bottom: 2px solid #3a418a;
    }
    /* --- Sentiment & Delta Colors --- */
    .positive { color: #4CAF50; }
    .negative { color: #F44336; }
    .neutral { color: #90A4AE; }
    .stMetric > div[data-testid="stMetricDelta"] .positive { color: #4CAF50 !important; }
    .stMetric > div[data-testid="stMetricDelta"] .negative { color: #F44336 !important; }
    /* --- Plotly Chart Styling --- */
    .plotly-chart { border-radius: 8px; background-color: #1E1E1E; }
    /* --- Sidebar --- */
    .stSidebar { background-color: #242426; border-right: 1px solid #444444; }
    .stSidebar .stButton button {
        width: 100%; background-color: #3a418a; color: white; border: none;
    }
    .stSidebar .stButton button:hover { background-color: #4a519a; color: white; }
    .stSidebar .stButton button:active { background-color: #2d317a; color: white; }
    .stSidebar .stTextInput input, .stSidebar .stSelectbox select,
    .stSidebar .stDateInput input, .stSidebar .stNumberInput input {
        background-color: #3a3a3c; color: #E0E0E0; border: 1px solid #555555;
    }
    .stSidebar label { color: #E0E0E0; }
    /* --- Dataframes --- */
    .stDataFrame { background-color: #2E2E2E; color: #E0E0E0; }
    .stDataFrame thead th { background-color: #3a3a3c; color: #FFFFFF; }
    .stDataFrame tbody tr:nth-child(even) { background-color: #3a3a3c; }
    /* --- Chat --- */
    .stChatInput { background-color: #2E2E2E; }
    .stChatMessage { background-color: #3a3a3c; border-radius: 8px; padding: 10px 15px; margin: 5px 0; }
</style>
"""
