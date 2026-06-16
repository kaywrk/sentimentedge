# SentimentEdge — Model Setup Guide

## Project Structure

```
model/
├── config.py                  ← All settings (tickers, dates, hyperparameters)
├── requirements.txt           ← Python packages
├── run_pipeline.py            ← Run everything in one command
│
├── data/
│   └── collect_data.py        ← Step 1: Download prices + news
│
├── features/
│   ├── sentiment.py           ← Step 2: FinBERT + VADER scoring
│   └── engineer_features.py  ← Step 3: Technical indicators + feature fusion
│
├── models/
│   ├── train.py               ← Step 4: Train & evaluate LSTM
│   ├── predict.py             ← Step 5: Load model + run inference
│   └── saved/                 ← Trained models saved here
│
└── api/
    └── serve.py               ← Step 6: Flask API for the website
```

---

## Quick Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create a .env file
```
NEWS_API_KEY=your_key_here    # optional, get from newsapi.org
```

### 3. Run the full pipeline (one command)
```bash
python run_pipeline.py
```

Or run step by step:
```bash
python data/collect_data.py
python features/sentiment.py
python features/engineer_features.py
python models/train.py
```

### 4. Start the API server
```bash
python api/serve.py
```

The server runs at **http://localhost:5000**

---

## Connecting to the Website

In `index.html`, replace the Claude API call with your Flask backend:

```javascript
// In the runAnalysis() function, replace the fetch call:
const response = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        ticker: currentTicker,
        text: document.getElementById('sentimentText').value
    })
});
const data = await response.json();

// data returns:
// { direction, confidence, probability_up, sentiment: { positive, negative, neutral } }
```

---

## Datasets to Download

| Dataset | Link | Save as |
|---|---|---|
| Financial PhraseBank | kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news | data/news/financial_news_kaggle.csv |
| Reddit WSB Posts | kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts | data/news/wsb_reddit.csv |
| Daily Financial News | kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db | data/news/financial_news_kaggle.csv |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/health | Check if server is running |
| POST | /api/predict | Predict stock direction from text |
| GET | /api/price/AAPL | Fetch recent price data for charts |
| GET | /api/results | View model evaluation metrics |

### POST /api/predict
```json
Request:  { "ticker": "AAPL", "text": "Apple beats earnings..." }
Response: { "direction": "BULLISH", "confidence": 78.4,
            "sentiment": { "positive": 0.82, "negative": 0.06 } }
```

---

## Expected Model Performance
Based on similar studies in the literature review:

| Metric | Price-Only Baseline | With Sentiment |
|---|---|---|
| Accuracy | ~52% | ~65–72% |
| AUC-ROC | ~0.52 | ~0.68–0.74 |
| F1 Score | ~0.51 | ~0.64–0.70 |

The sentiment fusion (FinBERT + VADER) accounts for most of the improvement over baseline.