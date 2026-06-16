# features/engineer_features.py
# Step 3 — Build technical indicators and fuse with sentiment scores

import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, TICKERS, LOOKBACK_WINDOW, ALL_FEATURES


# ─────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all price-based technical features to a daily OHLCV DataFrame.
    Input df must have columns: open, high, low, close, volume
    """
    df = df.copy().sort_index()

    # ── Returns
    df["daily_return"] = df["close"].pct_change()
    df["log_return"]   = np.log(df["close"] / df["close"].shift(1))

    # ── Moving Averages
    df["sma_5"]  = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ── MACD
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── RSI (14-period)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands (20-day, 2 std)
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20

    # ── Average True Range (14-period volatility)
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # ── Realised 10-day volatility
    df["vol_10d"] = df["log_return"].rolling(10).std() * np.sqrt(252)

    return df


# ─────────────────────────────────────────
# TARGET LABEL
# ─────────────────────────────────────────

def create_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Binary classification target:
      1 = next-day close is HIGHER than today's close  (UP)
      0 = next-day close is LOWER or equal             (DOWN)

    threshold: minimum % move to label as UP (e.g. 0.002 = +0.2%)
    Setting threshold > 0 avoids labeling tiny random movements.
    """
    df = df.copy()
    next_return = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (next_return > threshold).astype(int)
    return df


# ─────────────────────────────────────────
# MERGE SENTIMENT
# ─────────────────────────────────────────

def merge_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Left-join price features with daily sentiment scores.
    Important: sentiment from day T predicts price on day T+1 (next open).
    We align by shifting sentiment forward by 1 trading day.
    This solves the temporal misalignment problem.
    """
    sent = sentiment_df[sentiment_df["ticker"] == ticker].copy()
    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.set_index("date").sort_index()

    # Shift sentiment FORWARD by 1 day → news published today affects tomorrow's price
    sent_shifted = sent.shift(1)

    # Rolling sentiment features
    sent_shifted["sent_positive_3d"] = sent["sent_positive"].rolling(3).mean().shift(1)
    sent_shifted["sent_negative_3d"] = sent["sent_negative"].rolling(3).mean().shift(1)
    sent_shifted["sent_momentum"]    = sent["sent_compound"].diff().shift(1)

    price_df.index = pd.to_datetime(price_df.index)
    merged = price_df.join(sent_shifted[[
        "sent_positive", "sent_neutral", "sent_negative",
        "sent_compound", "news_count",
        "sent_positive_3d", "sent_negative_3d", "sent_momentum"
    ]], how="left")

    # Fill gaps with 0 (neutral) on days with no news
    sent_cols = [c for c in merged.columns if c.startswith("sent_") or c == "news_count"]
    merged[sent_cols] = merged[sent_cols].fillna(0)

    return merged


# ─────────────────────────────────────────
# BUILD FINAL FEATURE MATRIX
# ─────────────────────────────────────────

def build_feature_matrix(ticker: str, sentiment_path: str = None) -> pd.DataFrame:
    """
    Full pipeline for one ticker:
    1. Load prices
    2. Add technical indicators
    3. Create target label
    4. Merge sentiment
    5. Drop NaN rows (from rolling windows)
    Returns a clean DataFrame ready for model training.
    """
    # Load prices
    price_path = f"{DATA_DIR}prices/{ticker}_prices.csv"
    df = pd.read_csv(price_path, index_col="date", parse_dates=True)

    # Technical indicators
    df = add_technical_indicators(df)

    # Target (predict next-day direction)
    df = create_target(df, threshold=0.001)  # 0.1% minimum move

    # Merge sentiment if available
    if sentiment_path and os.path.exists(sentiment_path):
        sent_df = pd.read_csv(sentiment_path, parse_dates=["date"])
        df = merge_sentiment(df, sent_df, ticker)
    else:
        # Fill sentiment columns with zeros (price-only baseline)
        for col in [
            "sent_positive", "sent_neutral", "sent_negative",
            "sent_compound", "news_count",
            "sent_positive_3d", "sent_negative_3d", "sent_momentum"
        ]:
            df[col] = 0.0

    # Remove rows with NaN (rolling window warmup)
    df = df.dropna(subset=ALL_FEATURES + ["target"])

    # Keep only feature columns + target
    df = df[ALL_FEATURES + ["target", "close"]].copy()

    print(f"[{ticker}] Feature matrix: {df.shape} — "
          f"UP: {df['target'].sum()} | DOWN: {(df['target']==0).sum()}")

    return df


def build_all_tickers(sentiment_path: str = None) -> dict:
    """Build feature matrices for all tickers and save."""
    os.makedirs(f"{DATA_DIR}features", exist_ok=True)
    all_data = {}

    for ticker in TICKERS:
        print(f"\nBuilding features for {ticker}...")
        df = build_feature_matrix(ticker, sentiment_path)

        save_path = f"{DATA_DIR}features/{ticker}_features.csv"
        df.to_csv(save_path)
        print(f"  Saved → {save_path}")
        all_data[ticker] = df

    return all_data


# ─────────────────────────────────────────
# INSPECT FEATURE CORRELATIONS
# ─────────────────────────────────────────

def print_feature_importance(df: pd.DataFrame, top_n: int = 10):
    """
    Quick sanity check: Pearson correlation between each feature and the target.
    Not a substitute for SHAP values, but useful for fast validation.
    """
    corr = df[ALL_FEATURES].corrwith(df["target"]).abs().sort_values(ascending=False)
    print(f"\nTop {top_n} features correlated with target:")
    print(corr.head(top_n).to_string())
    return corr


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 3 — FEATURE ENGINEERING")
    print("=" * 55)

    sentiment_path = f"{DATA_DIR}sentiment/daily_sentiment.csv"
    if not os.path.exists(sentiment_path):
        print("No sentiment file found — using price-only features.")
        sentiment_path = None

    all_data = build_all_tickers(sentiment_path)

    # Print correlation check for first ticker
    print("\n── Correlation Check (AAPL) ──")
    print_feature_importance(all_data["AAPL"])

    print("\nRun next: python models/train.py")