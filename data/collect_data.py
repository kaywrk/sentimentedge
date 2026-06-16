# data/collect_data.py
# Step 1 — Collect stock price data (yfinance) and news headlines (NewsAPI / CSV)

import os
import time
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TICKERS, START_DATE, END_DATE, DATA_DIR, NEWS_API_KEY


# ─────────────────────────────────────────
# 1A — STOCK PRICE DATA
# ─────────────────────────────────────────

def fetch_stock_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """
    Download daily OHLCV price data from Yahoo Finance for all tickers.
    Saves one CSV per ticker to data/prices/.
    Returns a dict: { 'AAPL': DataFrame, ... }
    """
    os.makedirs(f"{DATA_DIR}prices", exist_ok=True)
    price_data = {}

    for ticker in tickers:
        print(f"[prices] Fetching {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if df.empty:
            print(f"  WARNING: No data returned for {ticker}")
            continue

        # Flatten multi-level columns (yfinance returns MultiIndex sometimes)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df["ticker"] = ticker

        save_path = f"{DATA_DIR}prices/{ticker}_prices.csv"
        df.to_csv(save_path)
        price_data[ticker] = df
        print(f"  Saved {len(df)} rows → {save_path}")

    return price_data


def load_stock_prices(ticker):
    """Load a previously saved price CSV."""
    path = f"{DATA_DIR}prices/{ticker}_prices.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


# ─────────────────────────────────────────
# 1B — NEWS HEADLINES
# ─────────────────────────────────────────

def fetch_news_newsapi(ticker, start=START_DATE, end=END_DATE):
    """
    Fetch financial news headlines for a ticker using NewsAPI.
    Requires NEWS_API_KEY in your .env file.
    Saves to data/news/{ticker}_news.csv.

    Free tier limitation: NewsAPI only provides 1 month of history.
    For historical data, use the Kaggle 'Daily Financial News' dataset (see below).
    """
    if not NEWS_API_KEY:
        print("  NEWS_API_KEY not set. Skipping live NewsAPI fetch.")
        return None

    os.makedirs(f"{DATA_DIR}news", exist_ok=True)

    # Map tickers to company keywords for better search results
    queries = {
        "AAPL": "Apple stock OR Apple Inc earnings",
        "TSLA": "Tesla stock OR Tesla Inc earnings",
        "NVDA": "NVIDIA stock OR NVIDIA earnings",
    }
    query = queries.get(ticker, f"{ticker} stock")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start,
        "to": end,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY,
    }

    articles = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok" or not data.get("articles"):
            break

        for a in data["articles"]:
            articles.append({
                "date": pd.to_datetime(a["publishedAt"]).date(),
                "source": a["source"]["name"],
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["url"],
            })

        if len(data["articles"]) < 100:
            break
        page += 1
        time.sleep(0.5)   # respect rate limits

    if not articles:
        print(f"  No articles found for {ticker}")
        return None

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    save_path = f"{DATA_DIR}news/{ticker}_news.csv"
    df.to_csv(save_path, index=False)
    print(f"  Saved {len(df)} articles → {save_path}")
    return df


def load_kaggle_news(path="data/news/financial_news_kaggle.csv"):
    """
    Load the Kaggle 'Daily Financial News for Stock Market Prediction' dataset.
    Download from: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db
    Then save it to data/news/financial_news_kaggle.csv

    Expected columns: Date, Stock, Headline, Sentiment (optional)
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date", "headline": "title", "stock": "ticker"})
    df["date"] = pd.to_datetime(df["date"])
    print(f"[kaggle news] Loaded {len(df)} rows")
    return df


def load_wsb_reddit(path="data/news/wsb_reddit.csv"):
    """
    Load the Reddit WallStreetBets dataset.
    Download from: https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts
    Expected columns: title, score, created_utc, ticker (if available)
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # Parse timestamp
    if "created_utc" in df.columns:
        df["date"] = pd.to_datetime(df["created_utc"], unit="s")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Filter to posts mentioning our tickers
    mask = df["title"].str.contains("AAPL|Apple|TSLA|Tesla|NVDA|NVIDIA", case=False, na=False)
    df = df[mask].copy()

    # Tag tickers
    def tag_ticker(t):
        t = str(t)
        if "AAPL" in t or "Apple" in t: return "AAPL"
        if "TSLA" in t or "Tesla" in t: return "TSLA"
        if "NVDA" in t or "NVIDIA" in t: return "NVDA"
        return None

    df["ticker"] = df["title"].apply(tag_ticker)
    df = df[df["ticker"].notna()]
    df = df.rename(columns={"title": "title"})
    print(f"[reddit wsb] Loaded {len(df)} relevant posts")
    return df[["date", "ticker", "title"]]


def create_demo_news():
    """
    Generate synthetic demo headlines when real data is unavailable.
    Useful for testing the pipeline end-to-end before downloading datasets.
    """
    import random
    random.seed(42)

    pos_templates = [
        "{t} beats Q{q} earnings expectations, EPS at ${v:.2f} vs ${e:.2f} estimate",
        "{t} announces record revenue, shares surge {p:.1f}%",
        "Analysts upgrade {t} to Buy, raise price target to ${pt}",
        "{t} AI chip demand outpacing supply, guidance raised for FY2025",
        "{t} secures major partnership, stock climbs on news",
    ]
    neg_templates = [
        "{t} misses Q{q} revenue forecast by ${m:.1f}B",
        "{t} faces supply chain disruption, shares decline {p:.1f}%",
        "Analysts downgrade {t} citing valuation concerns",
        "{t} warns of margin pressure amid rising input costs",
        "Short interest in {t} rises to 52-week high",
    ]
    neutral_templates = [
        "{t} to report Q{q} earnings next week",
        "{t} CFO speaks at investor conference",
        "{t} files 10-Q with SEC, no material changes noted",
    ]

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    rows = []
    for date in dates:
        for ticker in TICKERS:
            n = random.randint(1, 4)
            for _ in range(n):
                pool = random.choices(
                    [pos_templates, neg_templates, neutral_templates],
                    weights=[0.4, 0.35, 0.25]
                )[0]
                template = random.choice(pool)
                title = template.format(
                    t=ticker,
                    q=random.randint(1, 4),
                    v=random.uniform(2, 12),
                    e=random.uniform(2, 12),
                    p=random.uniform(1, 8),
                    pt=random.randint(150, 2000),
                    m=random.uniform(0.2, 3.0),
                )
                rows.append({"date": date, "ticker": ticker, "title": title})

    df = pd.DataFrame(rows)
    os.makedirs(f"{DATA_DIR}news", exist_ok=True)
    df.to_csv(f"{DATA_DIR}news/demo_news.csv", index=False)
    print(f"[demo news] Created {len(df)} synthetic headlines")
    return df


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 1 — DATA COLLECTION")
    print("=" * 55)

    # 1A — Prices
    print("\n[1A] Fetching stock prices...")
    price_data = fetch_stock_prices()

    # 1B — News (choose your source)
    print("\n[1B] Loading news data...")
    # Option A: Use demo data (works immediately, no API key needed)
    news_df = create_demo_news()

    # Option B: Uncomment when you have a NewsAPI key
    # for ticker in TICKERS:
    #     fetch_news_newsapi(ticker)

    # Option C: Uncomment when you've downloaded the Kaggle dataset
    # news_df = load_kaggle_news()

    print("\nData collection complete.")
    print(f"  Price files: data/prices/")
    print(f"  News file:   data/news/demo_news.csv")
    print("\nRun next: python features/engineer_features.py")