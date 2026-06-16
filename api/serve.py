# api/serve.py
# Step 6 — Flask REST API: connects the trained LSTM model to the website

import json
import yfinance as yf
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TICKERS, MODEL_DIR
from features.engineer_features import add_technical_indicators
from features.sentiment import VADERScorer
from models.predict import SentimentEdgePredictor


app = Flask(__name__)
CORS(app)  # Allow requests from your website frontend

# Load model once at startup (expensive operation)
predictor = SentimentEdgePredictor()
vader = VADERScorer()


# ─────────────────────────────────────────
# HELPER: Fetch live prices
# ─────────────────────────────────────────

def get_price_features(ticker: str, period: str = "60d") -> pd.DataFrame:
    """Fetch recent prices and compute technical indicators."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No price data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = add_technical_indicators(df)
    return df


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms API is running."""
    return jsonify({
        "status": "ok",
        "models_loaded": list(predictor.models.keys()),
        "tickers": TICKERS,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Request body (JSON):
    {
        "ticker": "AAPL",
        "text":   "Apple beats earnings expectations..."
    }

    Response:
    {
        "ticker":        "AAPL",
        "direction":     "BULLISH",
        "confidence":    78.4,
        "probability_up": 0.784,
        "sentiment": {
            "positive": 0.82,
            "neutral":  0.12,
            "negative": 0.06,
            "compound": 0.76,
            "label":    "positive"
        }
    }
    """
    body = request.get_json(force=True)
    ticker = body.get("ticker", "AAPL").upper()
    text   = body.get("text", "").strip()

    if ticker not in TICKERS:
        return jsonify({"error": f"Ticker must be one of {TICKERS}"}), 400

    if not text:
        return jsonify({"error": "text field is required"}), 400

    # Score sentiment (VADER — fast, no GPU needed for live API)
    sent_df = vader.score_texts([text])
    sent_row = sent_df.iloc[0]

    sentiment_scores = {
        "sent_positive":    float(sent_row["positive"]),
        "sent_negative":    float(sent_row["negative"]),
        "sent_neutral":     float(sent_row["neutral"]),
        "sent_compound":    float(sent_row["compound"]),
        "sent_positive_3d": float(sent_row["positive"]),
        "sent_negative_3d": float(sent_row["negative"]),
        "sent_momentum":    float(sent_row["compound"]),
        "news_count":       1.0,
    }

    # Get price features
    try:
        price_df = get_price_features(ticker)
    except Exception as e:
        return jsonify({"error": f"Price fetch failed: {str(e)}"}), 500

    # Fill sentiment columns into price df
    for col, val in sentiment_scores.items():
        price_df[col] = val

    price_df = price_df.dropna()

    # Run LSTM prediction (fallback to sentiment-only if model not loaded)
    try:
        result = predictor.predict(ticker, price_df, sentiment_scores)
    except Exception as e:
        # Graceful fallback: use only sentiment signal
        compound = sentiment_scores["sent_compound"]
        direction = "BULLISH" if compound > 0.05 else "BEARISH" if compound < -0.05 else "NEUTRAL"
        conf = round(abs(compound) * 60 + 50, 1)
        result = {
            "ticker":         ticker,
            "direction":      direction,
            "probability_up": round(0.5 + compound * 0.3, 4),
            "confidence":     min(conf, 92),
            "sentiment_used": True,
            "_fallback":      str(e),
        }

    # Build full response
    response = {
        **result,
        "sentiment": {
            "positive": round(float(sent_row["positive"]), 4),
            "neutral":  round(float(sent_row["neutral"]),  4),
            "negative": round(float(sent_row["negative"]), 4),
            "compound": round(float(sent_row["compound"]), 4),
            "label":    str(sent_row["predicted_label"]),
        }
    }

    return jsonify(response)


@app.route("/api/price/<ticker>", methods=["GET"])
def price(ticker):
    """
    Return recent price data for chart rendering on the frontend.

    Response:
    {
        "ticker": "AAPL",
        "prices": [198.4, 201.2, ...],
        "dates":  ["2025-05-01", ...],
        "current": 211.40,
        "change_pct": 2.34
    }
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return jsonify({"error": f"Unsupported ticker: {ticker}"}), 400

    try:
        df = yf.download(ticker, period="30d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        prices = df["close"].round(2).tolist()
        dates  = [d.strftime("%Y-%m-%d") for d in df.index]
        current = prices[-1]
        change_pct = round((prices[-1] / prices[-2] - 1) * 100, 2) if len(prices) > 1 else 0

        return jsonify({
            "ticker":     ticker,
            "prices":     prices,
            "dates":      dates,
            "current":    current,
            "change_pct": change_pct,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results", methods=["GET"])
def results():
    """Return saved model evaluation results."""
    from config import RESULTS_PATH
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "No results found — run train.py first"}), 404


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SentimentEdge API")
    print("="*55)
    print("  Endpoints:")
    print("  GET  /api/health")
    print("  POST /api/predict    { ticker, text }")
    print("  GET  /api/price/<ticker>")
    print("  GET  /api/results")
    print("="*55)
    print("\n  Starting server on http://localhost:5000\n")

    app.run(debug=True, host="0.0.0.0", port=5000)