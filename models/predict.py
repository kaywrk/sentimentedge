# models/predict.py
# Step 5 — Load saved models and run live predictions

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, ALL_FEATURES, LOOKBACK_WINDOW, TICKERS
from features.engineer_features import add_technical_indicators


class SentimentEdgePredictor:
    """
    Production-ready predictor.
    Loads trained LSTM models + scalers for each ticker.
    Exposes a simple predict() method for the Flask API.
    """

    def __init__(self):
        self.models  = {}
        self.scalers = {}
        self._load_all()

    def _load_all(self):
        for ticker in TICKERS:
            model_path  = f"{MODEL_DIR}{ticker}_lstm.keras"
            scaler_path = f"{MODEL_DIR}{ticker}_scaler.joblib"

            try:
                self.models[ticker]  = tf.keras.models.load_model(model_path)
                self.scalers[ticker] = joblib.load(scaler_path)
                print(f"[predictor] Loaded {ticker} model")
            except FileNotFoundError:
                print(f"[predictor] No saved model for {ticker} — run train.py first")

    def predict(self, ticker: str, feature_df: pd.DataFrame, sentiment_scores: dict = None) -> dict:
        """
        Make a next-day prediction for a ticker.

        Args:
            ticker: 'AAPL', 'TSLA', or 'NVDA'
            feature_df: DataFrame with ALL_FEATURES columns, sorted ascending by date.
                        Must have at least LOOKBACK_WINDOW rows.
            sentiment_scores: Optional dict with keys:
                              sent_positive, sent_negative, sent_neutral,
                              sent_compound, news_count

        Returns:
            dict with: direction, confidence, probability,
                       sentiment_used, model_version
        """
        if ticker not in self.models:
            raise ValueError(f"No model loaded for {ticker}. Run train.py first.")

        df = feature_df.copy()

        # Inject live sentiment if provided
        if sentiment_scores:
            for col, val in sentiment_scores.items():
                if col in df.columns:
                    df.iloc[-1, df.columns.get_loc(col)] = val

        # Ensure all feature columns exist
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        # Take the last LOOKBACK_WINDOW rows
        window_df = df[ALL_FEATURES].tail(LOOKBACK_WINDOW)

        if len(window_df) < LOOKBACK_WINDOW:
            raise ValueError(
                f"Need at least {LOOKBACK_WINDOW} rows of data. Got {len(window_df)}."
            )

        # Scale
        scaler = self.scalers[ticker]
        X_scaled = scaler.transform(window_df.values)

        # Reshape for LSTM: (1, window, features)
        X_input = X_scaled.reshape(1, LOOKBACK_WINDOW, len(ALL_FEATURES))

        # Inference
        prob = float(self.models[ticker].predict(X_input, verbose=0)[0][0])
        direction = "BULLISH" if prob >= 0.5 else "BEARISH"
        confidence = round(max(prob, 1 - prob) * 100, 1)

        return {
            "ticker":        ticker,
            "direction":     direction,
            "probability_up": round(prob, 4),
            "confidence":    confidence,
            "sentiment_used": sentiment_scores is not None,
        }

    def predict_from_text(self, ticker: str, text: str, price_df: pd.DataFrame) -> dict:
        """
        Convenience method: score text with VADER, then predict.
        Used when the full FinBERT pipeline isn't running live.
        """
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        s = analyzer.polarity_scores(text)

        sentiment_scores = {
            "sent_positive": s["pos"],
            "sent_negative": s["neg"],
            "sent_neutral":  s["neu"],
            "sent_compound": s["compound"],
            "sent_positive_3d": s["pos"],
            "sent_negative_3d": s["neg"],
            "sent_momentum":    s["compound"],
            "news_count":       1,
        }

        # Build feature df
        df = add_technical_indicators(price_df)
        for col in sentiment_scores:
            df[col] = sentiment_scores[col]

        df = df.dropna(subset=ALL_FEATURES)
        return self.predict(ticker, df, sentiment_scores)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    predictor = SentimentEdgePredictor()

    ticker = "AAPL"
    price_df = yf.download(ticker, period="60d", auto_adjust=True, progress=False)

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = [c[0].lower() for c in price_df.columns]
    else:
        price_df.columns = [c.lower() for c in price_df.columns]

    test_headline = "Apple beats Q2 earnings expectations with record iPhone sales"

    result = predictor.predict_from_text(ticker, test_headline, price_df)
    print("\n── Prediction Result ──")
    for k, v in result.items():
        print(f"  {k:<20}: {v}")