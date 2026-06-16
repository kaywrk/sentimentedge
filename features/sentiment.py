# features/sentiment.py
# Step 2 — Score headlines with FinBERT (contextual) and VADER (lexical)

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, FINBERT_MODEL, FINBERT_MAX_LEN, TICKERS


# ─────────────────────────────────────────
# FINBERT SCORER
# ─────────────────────────────────────────

class FinBERTScorer:
    """
    Scores text using ProsusAI/finbert — a BERT model fine-tuned on
    financial news. Returns probabilities for: positive, negative, neutral.

    Labels returned by the model: {'positive': 0, 'negative': 1, 'neutral': 2}
    """

    def __init__(self, model_name=FINBERT_MODEL, batch_size=32, device=None):
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FinBERT] Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Map model output indices → sentiment labels
        # ProsusAI/finbert label order: positive=0, negative=1, neutral=2
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        print("[FinBERT] Ready.")

    def score_batch(self, texts: list[str]) -> list[dict]:
        """
        Score a list of texts.
        Returns: [{"positive": float, "negative": float, "neutral": float}, ...]
        """
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=FINBERT_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for row in probs:
                results.append({
                    "positive": float(row[0]),
                    "negative": float(row[1]),
                    "neutral":  float(row[2]),
                })

        return results

    def score_texts(self, texts: list[str]) -> pd.DataFrame:
        """
        Score texts and return a DataFrame with columns:
        positive, negative, neutral, compound, predicted_label
        """
        scores = self.score_batch(texts)
        df = pd.DataFrame(scores)

        # Compound: positive - negative (range: -1 to +1)
        df["compound"] = df["positive"] - df["negative"]

        # Predicted label
        df["predicted_label"] = df[["positive", "negative", "neutral"]].idxmax(axis=1)

        return df


# ─────────────────────────────────────────
# VADER SCORER (for social media text)
# ─────────────────────────────────────────

class VADERScorer:
    """
    VADER is rule-based — fast, no GPU needed.
    Best for informal social media (Reddit, Twitter).
    Returns: positive, negative, neutral, compound (standard VADER output)
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        print("[VADER] Ready.")

    def score_texts(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        for text in texts:
            s = self.analyzer.polarity_scores(str(text))
            rows.append({
                "positive":  s["pos"],
                "negative":  s["neg"],
                "neutral":   s["neu"],
                "compound":  s["compound"],
                "predicted_label": (
                    "positive" if s["compound"] >= 0.05
                    else "negative" if s["compound"] <= -0.05
                    else "neutral"
                )
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────
# AGGREGATE DAILY SENTIMENT
# ─────────────────────────────────────────

def aggregate_daily_sentiment(news_df: pd.DataFrame, scorer, source_label: str) -> pd.DataFrame:
    """
    Score all headlines and aggregate to one row per (date, ticker).

    news_df must have: date, ticker, title
    Returns: DataFrame with date, ticker, sent_positive, sent_negative,
             sent_neutral, sent_compound, news_count
    """
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()

    print(f"[{source_label}] Scoring {len(news_df)} headlines...")

    scores = scorer.score_texts(news_df["title"].tolist())
    news_df = news_df.reset_index(drop=True)
    news_df = pd.concat([news_df, scores], axis=1)

    # Aggregate: mean scores + count per day per ticker
    agg = (
        news_df.groupby(["date", "ticker"])
        .agg(
            sent_positive=("positive", "mean"),
            sent_negative=("negative", "mean"),
            sent_neutral=("neutral",  "mean"),
            sent_compound=("compound", "mean"),
            news_count=("title", "count"),
        )
        .reset_index()
    )

    return agg


def build_daily_sentiment(news_path: str, use_finbert=True, use_vader=True):
    """
    Full pipeline: load news → score → aggregate → save.

    Strategy:
    - FinBERT for formal news headlines (more accurate on financial text)
    - VADER for informal social media posts (faster, handles slang)
    - Final sentiment = weighted average: 0.6 * FinBERT + 0.4 * VADER

    If only one scorer is available, uses that one alone.
    """
    print(f"\n[sentiment] Loading news from {news_path}...")
    news_df = pd.read_csv(news_path, parse_dates=["date"])

    # Filter to our tickers
    news_df = news_df[news_df["ticker"].isin(TICKERS)].copy()
    print(f"  {len(news_df)} headlines for {TICKERS}")

    dfs = []

    if use_finbert:
        try:
            scorer = FinBERTScorer()
            agg = aggregate_daily_sentiment(news_df, scorer, "FinBERT")
            agg["_source"] = "finbert"
            dfs.append(("finbert", agg))
        except Exception as e:
            print(f"  FinBERT failed: {e}. Skipping.")
            use_finbert = False

    if use_vader:
        scorer = VADERScorer()
        agg = aggregate_daily_sentiment(news_df, scorer, "VADER")
        agg["_source"] = "vader"
        dfs.append(("vader", agg))

    # Combine if both available
    if len(dfs) == 2:
        fb = dfs[0][1].set_index(["date", "ticker"])
        vd = dfs[1][1].set_index(["date", "ticker"])

        sent_cols = ["sent_positive", "sent_negative", "sent_neutral", "sent_compound"]
        combined = fb[sent_cols].copy()
        for col in sent_cols:
            combined[col] = 0.6 * fb[col] + 0.4 * vd[col]
        combined["news_count"] = fb["news_count"]
        combined = combined.reset_index()

    else:
        combined = dfs[0][1].drop(columns=["_source"])

    # Save
    os.makedirs(f"{DATA_DIR}sentiment", exist_ok=True)
    out_path = f"{DATA_DIR}sentiment/daily_sentiment.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n[sentiment] Saved → {out_path}")
    print(combined.head())

    return combined


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 2 — SENTIMENT SCORING")
    print("=" * 55)

    # Choose news source:
    news_source = f"{DATA_DIR}news/demo_news.csv"

    # For Kaggle dataset:
    # news_source = f"{DATA_DIR}news/financial_news_kaggle.csv"

    sentiment_df = build_daily_sentiment(
        news_path=news_source,
        use_finbert=True,
        use_vader=True,
    )

    print(f"\nSentiment scores summary:")
    print(sentiment_df.groupby("ticker")[["sent_compound", "news_count"]].describe())
    print("\nRun next: python features/engineer_features.py")