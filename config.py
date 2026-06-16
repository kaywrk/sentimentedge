# config.py — Central configuration for SentimentEdge model

TICKERS = ["AAPL", "TSLA", "NVDA"]

START_DATE = "2024-01-01"
END_DATE   = "2026-04-01"

# ── Sentiment model
FINBERT_MODEL = "ProsusAI/finbert"    # HuggingFace model ID
FINBERT_MAX_LEN = 512

# ── LSTM hyperparameters
LOOKBACK_WINDOW = 10    # days of history fed into LSTM
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.3
LSTM_UNITS      = [64, 32]            # two stacked LSTM layers

# ── Feature columns used in model input
PRICE_FEATURES = [
    "close", "open", "high", "low", "volume",
    "daily_return", "log_return",
    "sma_5", "sma_10", "sma_20",
    "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "atr_14",
    "vol_10d",       # rolling 10-day realised volatility
]

SENTIMENT_FEATURES = [
    "sent_positive", "sent_neutral", "sent_negative",
    "sent_compound",
    "sent_positive_3d",   # 3-day rolling mean
    "sent_negative_3d",
    "sent_momentum",      # today's compound minus yesterday's
    "news_count",         # number of articles that day
]

ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES

# ── Train / val / test split (chronological — no shuffle)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15

# ── Output paths
DATA_DIR   = "data/"
MODEL_DIR  = "models/saved/"
SCALER_PATH = "models/saved/scaler.joblib"
MODEL_PATH  = "models/saved/lstm_model.keras"
RESULTS_PATH = "models/saved/results.json"

# ── API keys (put these in a .env file — never hardcode)
import os
from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")