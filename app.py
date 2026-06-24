"""
SentimentEdge — Streamlit Analytics Dashboard
Sentiment-Based Stock Movement Prediction
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentEdge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
API_URL   = "http://localhost:5000/api"
TICKERS   = ["AAPL", "TSLA", "NVDA"]
TK_NAMES  = {"AAPL": "Apple Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp."}
TK_COLORS = {"AAPL": "#60a5fa", "TSLA": "#f87171", "NVDA": "#4ade80"}

LOGOS = {
    "AAPL": "data:image/png;base64,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",
    "NVDA": "data:image/png;base64,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",
    "TSLA": "data:image/png;base64,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",
}


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

UP_COLOR  = "#10b981"
DN_COLOR  = "#f43f5e"
BLUE      = "#2563eb"
GOLD      = "#f59e0b"
BG        = "#07111f"
CARD      = "#0d1b2e"
TEXT      = "#e2e8f0"
MUTED     = "#7a9ab8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial", color=TEXT, size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, tickfont=dict(size=10, color=MUTED)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, tickfont=dict(size=10, color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hoverlabel=dict(bgcolor=CARD, bordercolor="rgba(255,255,255,0.1)", font=dict(size=12, color=TEXT)),
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Main app background */
.stApp {
    background-color: #07111F !important;
    color: #E2E8F0 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #07111F !important;
}

[data-testid="stMain"] {
    background-color: #07111F !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; padding-left: 1.5rem; padding-right: 1.5rem; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0d1b2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.7rem !important;
    color: #e8f4ff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a9ab8 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #060f1c !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* ── Buttons ── */
.stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.04em !important;
    padding: 10px 24px !important;
    transition: background 0.15s !important;
    width: 100%;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* ── Text area ── */
textarea {
    background: #111f34 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
}
textarea:focus { border-color: rgba(37,99,235,0.5) !important; }

/* ── Select boxes ── */
.stSelectbox div[data-baseweb="select"] > div {
    background: #111f34 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7a9ab8 !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,0.15) !important;
    color: #60a5fa !important;
    border-bottom: 2px solid #2563eb !important;
}

/* ── Radio buttons ── */
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 13px !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1rem 0; }

/* ── Cards via markdown ── */
.se-card {
    background: #0d1b2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.se-card-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a9ab8;
    margin-bottom: 4px;
}
.se-ticker-sym { font-size: 18px; font-weight: 700; color: #e8f4ff; }
.se-ticker-name { font-size: 11px; color: #7a9ab8; margin-bottom: 8px; }
.se-price { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 500; color: #e8f4ff; }
.se-up { color: #10b981; font-weight: 600; font-size: 13px; }
.se-dn { color: #f43f5e; font-weight: 600; font-size: 13px; }
.se-badge-up { background: rgba(16,185,129,0.12); color: #10b981; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
.se-badge-dn { background: rgba(244,63,94,0.12); color: #f43f5e; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
.se-badge-bull { background: rgba(16,185,129,0.12); color: #10b981; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; }
.se-badge-bear { background: rgba(244,63,94,0.12); color: #f43f5e; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; }
.se-badge-neut { background: rgba(100,116,139,0.15); color: #94a3b8; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; }
.se-mono { font-family: 'JetBrains Mono', monospace; }
.se-section-title { font-size: 22px; font-weight: 700; color: #e8f4ff; margin-bottom: 4px; }
.se-section-sub { font-size: 13px; color: #7a9ab8; margin-bottom: 20px; }
.verdict-bull { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 20px; text-align: center; }
.verdict-bear { background: rgba(244,63,94,0.06); border: 1px solid rgba(244,63,94,0.3); border-radius: 12px; padding: 20px; text-align: center; }
.verdict-neut { background: rgba(100,116,139,0.08); border: 1px solid rgba(100,116,139,0.2); border-radius: 12px; padding: 20px; text-align: center; }
.verdict-arrow { font-size: 40px; line-height: 1; }
.verdict-word-bull { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 600; color: #10b981; letter-spacing: 0.1em; }
.verdict-word-bear { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 600; color: #f43f5e; letter-spacing: 0.1em; }
.verdict-word-neut { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 600; color: #94a3b8; letter-spacing: 0.1em; }
.log-row { border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0; display: flex; gap: 10px; align-items: flex-start; }
.log-time { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #4a6380; min-width: 50px; }
.log-text { font-size: 11px; color: #c8d8e8; flex: 1; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page":        "Dashboard",
        "ticker":      "AAPL",
        "is_live":     False,
        "price_cache": {},
        "results":     {},
        "analysis_log": [],
        "last_result": None,
        "anal_count":  0,
        "signals":     {"AAPL": {"dir": "BULLISH", "conf": 74}, "TSLA": {"dir": "BEARISH", "conf": 68}, "NVDA": {"dir": "BULLISH", "conf": 81}},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────
# BACKEND HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_prices(ticker: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/price/{ticker}", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_results() -> dict:
    try:
        r = requests.get(f"{API_URL}/results", timeout=5)
        if r.ok:
            data = r.json()
            if "error" not in data:
                return data
    except Exception:
        pass
    return {}


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.ok
    except Exception:
        return False


def predict_backend(ticker: str, text: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"ticker": ticker, "text": text},
            timeout=10,
        )
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────────
def demo_prices(ticker: str, days: int = 30) -> dict:
    rng = np.random.default_rng({"AAPL": 42, "TSLA": 17, "NVDA": 91}.get(ticker, 0))
    base = {"AAPL": 195, "TSLA": 245, "NVDA": 1020}.get(ticker, 200)
    prices, dates = [], []
    p = base
    end = datetime.today()
    for i in range(days * 2):
        d = end - timedelta(days=(days * 2 - i))
        if d.weekday() >= 5:
            continue
        p *= 1 + (rng.random() - 0.49) * 0.025
        prices.append(round(p, 2))
        dates.append(d.strftime("%b %d"))
        if len(prices) >= days:
            break
    change = round((prices[-1] / prices[0] - 1) * 100, 2) if prices else 0
    return {"prices": prices, "dates": dates, "current": prices[-1] if prices else base, "change_pct": change}


def get_prices(ticker: str, days: int = 30) -> dict:
    if st.session_state.is_live:
        cached = fetch_live_prices(ticker)
        if cached:
            return cached
    return demo_prices(ticker, days)


def demo_sentiment(n: int) -> list:
    rng = np.random.default_rng(21)
    return list(np.round(rng.random(n) * 1.6 - 0.7, 3))


def demo_rsi(n: int) -> list:
    rng = np.random.default_rng(7)
    v, result = 52.0, []
    for _ in range(n):
        v = max(15, min(85, v + (rng.random() - 0.5) * 8))
        result.append(round(v, 1))
    return result


def local_sentiment(text: str) -> dict:
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    direction = "BULLISH" if compound > 0.05 else "BEARISH" if compound < -0.05 else "NEUTRAL"
    confidence = round(54 + abs(compound) * 38)
    return {
        "direction":     direction,
        "confidence":    confidence,
        "probability_up": round(0.5 + compound * 0.35, 4),
        "sentiment": {
            "positive": round(scores["pos"], 3),
            "negative": round(scores["neg"], 3),
            "neutral":  round(scores["neu"], 3),
            "compound": round(compound, 3),
            "label":    direction.lower(),
        },
    }


# ─────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────
def price_sentiment_chart(ticker: str, days: int = 30) -> go.Figure:
    d = get_prices(ticker, days)
    prices  = d["prices"]
    dates   = d["dates"]
    n       = len(prices)
    sent    = demo_sentiment(n)
    up      = prices[-1] >= prices[0]
    pc      = UP_COLOR if up else DN_COLOR

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price area
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        name="Price",
        line=dict(color=pc, width=2.5),
        fill="tozeroy",
        fillcolor=f"{'rgba(16,185,129' if up else 'rgba(244,63,94'}, 0.08)",
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
    ), secondary_y=False)

    # Sentiment overlay
    fig.add_trace(go.Scatter(
        x=dates, y=sent,
        name="Sentiment",
        line=dict(color="#818cf8", width=1.5, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Sent: %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    layout = {**PLOTLY_LAYOUT,
        "yaxis":  dict(tickprefix="$", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color=MUTED), showline=False),
        "yaxis2": dict(range=[-1.2, 1.2], tickformat=".1f", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9, color="#818cf8"), showline=False),
        "legend": dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED)),
        "hovermode": "x unified",
    }
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig


def rsi_chart(ticker: str, days: int = 30) -> go.Figure:
    d = get_prices(ticker, days)
    dates = d["dates"]
    rsi   = demo_rsi(len(dates))
    last  = rsi[-1]
    color = DN_COLOR if last > 70 else UP_COLOR if last < 30 else BLUE

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=rsi, name="RSI",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"{'rgba(244,63,94' if last > 70 else 'rgba(37,99,235'}, 0.06)",
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ))
    for level, lc, label in [(70, DN_COLOR, "Overbought"), (30, UP_COLOR, "Oversold")]:
        fig.add_hline(y=level, line_dash="dot", line_color=lc, line_width=1, opacity=0.5,
                      annotation_text=label, annotation_font_size=9, annotation_font_color=lc)

    fig.update_layout(**{**PLOTLY_LAYOUT, "yaxis": dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color=MUTED))})
    return fig, last


def macd_chart(ticker: str, days: int = 30) -> go.Figure:
    d     = get_prices(ticker, days)
    dates = d["dates"]
    n     = len(dates)
    rng   = np.random.default_rng(13)
    v     = 0.0
    macd_vals, signal_vals, hist_vals = [], [], []
    for _ in range(n):
        v += (rng.random() - 0.5) * 0.8
        sig = v * 0.85 + rng.random() * 0.1 - 0.05
        macd_vals.append(round(v, 3))
        signal_vals.append(round(sig, 3))
        hist_vals.append(round(v - sig, 3))

    colors = [UP_COLOR if h >= 0 else DN_COLOR for h in hist_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=hist_vals, name="Histogram", marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=dates, y=macd_vals,   name="MACD",   line=dict(color=GOLD,  width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=signal_vals, name="Signal", line=dict(color=BLUE, width=1.5, dash="dot")))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.update_layout(**{**PLOTLY_LAYOUT,
        "legend": dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED))
    })
    return fig


def sentiment_comparison_chart() -> go.Figure:
    data = {
        "AAPL": {"pos": 0.72, "neu": 0.18, "neg": 0.10},
        "TSLA": {"pos": 0.32, "neu": 0.28, "neg": 0.40},
        "NVDA": {"pos": 0.81, "neu": 0.13, "neg": 0.06},
    }
    fig = go.Figure()
    for label, color, key in [("Positive", UP_COLOR, "pos"), ("Neutral", "#64748b", "neu"), ("Negative", DN_COLOR, "neg")]:
        fig.add_trace(go.Bar(
            name=label,
            x=TICKERS,
            y=[data[t][key] for t in TICKERS],
            marker_color=color,
            opacity=0.8,
            text=[f"{data[t][key]*100:.0f}%" for t in TICKERS],
            textposition="inside",
            textfont=dict(size=10, color="white"),
        ))
    fig.update_layout(**{**PLOTLY_LAYOUT,
        "barmode": "stack",
        "yaxis": dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color=MUTED)),
        "legend": dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED)),
    })
    return fig


def volatility_chart() -> go.Figure:
    d   = get_prices("AAPL", 30)
    n   = len(d["dates"])
    fig = go.Figure()
    for tk, seed, c in [("AAPL", 5, TK_COLORS["AAPL"]), ("TSLA", 9, TK_COLORS["TSLA"]), ("NVDA", 15, TK_COLORS["NVDA"])]:
        rng = np.random.default_rng(seed)
        vol = [round(abs(np.sin(i * 0.3) * 0.04 + 0.12 + rng.random() * 0.03) * (1.5 if tk == "TSLA" else 1.3 if tk == "NVDA" else 1), 3) for i in range(n)]
        fig.add_trace(go.Scatter(
            x=d["dates"], y=vol, name=tk,
            line=dict(color=c, width=2), fill="tozeroy",
            fillcolor=hex_to_rgba(c, 0.06),
            hovertemplate=f"{tk}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(**{**PLOTLY_LAYOUT,
        "yaxis": dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color=MUTED)),
        "legend": dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED)),
    })
    return fig


def radar_chart(results: dict) -> go.Figure:
    cats = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    fig = go.Figure()
    for tk, color in TK_COLORS.items():
        r   = results.get(tk, {})
        vals = [
            round((r.get("accuracy",  0.68) * 100)),
            round((r.get("precision", 0.68) * 100)),
            round((r.get("recall",    0.65) * 100)),
            round((r.get("f1_score",  0.67) * 100)),
            round((r.get("auc_roc",   0.70) * 100)),
        ] if r else [68, 71, 65, 68, 72] if tk == "AAPL" else [65, 63, 67, 64, 70] if tk == "TSLA" else [71, 73, 69, 70, 74]
        vals.append(vals[0])
        c_cats = cats + [cats[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=c_cats, name=tk,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.1),
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[50, 85], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=9, color=MUTED), showline=False),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=10, color=MUTED)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        legend=dict(orientation="h", y=-0.12, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED)),
        margin=dict(l=20, r=20, t=30, b=40),
        hoverlabel=dict(bgcolor=CARD, font=dict(size=12, color=TEXT)),
    )
    return fig


def accuracy_bar_chart(results: dict) -> go.Figure:
    metrics = {}
    for tk in TICKERS:
        r = results.get(tk, {})
        metrics[tk] = {
            "acc": round((r.get("accuracy",  {"AAPL":0.68,"TSLA":0.65,"NVDA":0.71}[tk]) * 100), 1),
            "f1":  round((r.get("f1_score",  {"AAPL":0.67,"TSLA":0.64,"NVDA":0.70}[tk]) * 100), 1),
            "auc": round((r.get("auc_roc",   {"AAPL":0.72,"TSLA":0.70,"NVDA":0.74}[tk]) * 100), 1),
        }
    fig = go.Figure()
    for label, key, opacity in [("Accuracy", "acc", 0.9), ("F1 Score", "f1", 0.6), ("AUC-ROC", "auc", 0.4)]:
        fig.add_trace(go.Bar(
            name=label, x=TICKERS,
            y=[metrics[t][key] for t in TICKERS],
            marker_color=[TK_COLORS[t] for t in TICKERS],
            opacity=opacity,
            text=[f"{metrics[t][key]}%" for t in TICKERS],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
        ))
    fig.update_layout(**{**PLOTLY_LAYOUT,
        "barmode": "group",
        "yaxis": dict(range=[50, 90], ticksuffix="%", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color=MUTED)),
        "legend": dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED)),
    })
    return fig


def confidence_gauge(conf: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        number=dict(suffix="%", font=dict(size=28, color=TEXT, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(size=9, color=MUTED)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                dict(range=[0, 55],  color="rgba(244,63,94,0.08)"),
                dict(range=[55, 70], color="rgba(245,158,11,0.08)"),
                dict(range=[70, 100],color="rgba(16,185,129,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=0.75, value=conf),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=160, margin=dict(l=20, r=20, t=20, b=10),
                      font=dict(color=TEXT))
    return fig


def sentiment_gauge(compound: float) -> go.Figure:
    color = UP_COLOR if compound > 0.1 else DN_COLOR if compound < -0.1 else BLUE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(compound, 3),
        number=dict(font=dict(size=28, color=TEXT, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor=MUTED, tickfont=dict(size=9, color=MUTED)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                dict(range=[-1, -0.05],  color="rgba(244,63,94,0.08)"),
                dict(range=[-0.05, 0.05],color="rgba(100,116,139,0.08)"),
                dict(range=[0.05, 1],    color="rgba(16,185,129,0.08)"),
            ],
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=160, margin=dict(l=20, r=20, t=20, b=10),
                      font=dict(color=TEXT))
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:20px;font-weight:700;color:#e8f4ff;letter-spacing:-.02em;margin-bottom:4px">Sentiment<span style="color:#2563eb;font-weight:300">Edge</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#7a9ab8;margin-bottom:20px">Stock Movement Prediction</div>', unsafe_allow_html=True)

    # Connection status
    is_live = check_backend()
    st.session_state.is_live = is_live
    if is_live:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;background:#0d1b2e;border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:8px 12px;margin-bottom:16px"><div style="width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 6px #10b981"></div><span style="font-size:11px;color:#10b981;font-weight:500">Backend Live</span></div>', unsafe_allow_html=True)
        results = fetch_model_results()
        if results:
            st.session_state.results = results
    else:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;background:#0d1b2e;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px;margin-bottom:16px"><div style="width:8px;height:8px;border-radius:50%;background:#f43f5e"></div><span style="font-size:11px;color:#7a9ab8;font-weight:500">Demo Mode</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4a6380;margin-bottom:8px">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(" ", ["Dashboard", "Analysis", "Market"], label_visibility="collapsed",
                    index=["Dashboard", "Analysis", "Market"].index(st.session_state.page))
    st.session_state.page = page

    st.markdown("---")
    st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4a6380;margin-bottom:8px">Active Ticker</div>', unsafe_allow_html=True)
    ticker = st.selectbox(" ", TICKERS, index=TICKERS.index(st.session_state.ticker), label_visibility="collapsed")
    st.session_state.ticker = ticker

    st.markdown("---")
    st.markdown(f'<div style="font-size:10px;color:#4a6380">Session analyses: <span style="color:#60a5fa;font-family:JetBrains Mono">{st.session_state.anal_count}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px;color:#4a6380;margin-top:4px">Time: <span style="color:#7a9ab8;font-family:JetBrains Mono">{datetime.now().strftime("%H:%M:%S")} ET</span></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ══ PAGE: DASHBOARD ══
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "Dashboard":
    st.markdown('<div class="se-section-title">Market Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="se-section-sub">{"Live data from Flask backend" if is_live else "Demo mode — start backend with: py -3.11 api/serve.py"}</div>', unsafe_allow_html=True)

    # ── KPI ROW ──
    k1, k2, k3, k4 = st.columns(4)
    smap = {"AAPL": 0.42, "TSLA": -0.28, "NVDA": 0.67}
    avg_sent = round(sum(smap.values()) / len(smap), 2)
    results  = st.session_state.results
    avg_acc  = round(sum(r.get("accuracy", 0.68) for r in results.values()) / 3 * 100, 1) if results else 68.0
    best_tk  = max(results, key=lambda t: results[t].get("accuracy", 0)) if results else "NVDA"
    bull_cnt = sum(1 for v in st.session_state.signals.values() if v["dir"] == "BULLISH")

    with k1:
        st.metric("Avg Sentiment Score", f"{avg_sent:+.2f}", "Bullish" if avg_sent > 0 else "Bearish")
    with k2:
        st.metric("Model Accuracy", f"{avg_acc:.1f}%", f"{best_tk} best")
    with k3:
        st.metric("Bullish Signals", f"{bull_cnt} / {len(TICKERS)}", "Active now")
    with k4:
        st.metric("Analyses Run", st.session_state.anal_count, "This session")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TICKER CARDS ──
    c1, c2, c3 = st.columns(3)
    for col, tk in zip([c1, c2, c3], TICKERS):
        d   = get_prices(tk, 14)
        px  = d["current"]
        chg = d["change_pct"]
        up  = chg >= 0
        sig = st.session_state.signals.get(tk, {})
        with col:
            badge = f'<span class="se-badge-{"up" if up else "dn"}">{"▲" if up else "▼"} {abs(chg):.2f}%</span>'
            sig_badge = f'<span class="se-badge-{"bull" if sig.get("dir")=="BULLISH" else "bear" if sig.get("dir")=="BEARISH" else "neut"}">{sig.get("dir","—")}</span>'
            st.markdown(f"""
            <div class="se-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div style="width:38px;height:38px;border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;padding:4px"><img src="{LOGOS[tk]}" width="28" height="28" style="object-fit:contain"></div>
                {badge}
              </div>
              <div class="se-ticker-sym">{tk}</div>
              <div class="se-ticker-name">{TK_NAMES[tk]}</div>
              <div class="se-price">${px:,.2f}</div>
              <div style="margin-top:8px">{sig_badge} <span style="font-size:10px;color:#4a6380">{sig.get("conf",70)}% conf</span></div>
            </div>""", unsafe_allow_html=True)

    # ── MAIN CHART + SIGNALS ──
    st.markdown("---")
    ch_col, sig_col = st.columns([2, 1])

    with ch_col:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:8px">Price & Sentiment Overlay</div>', unsafe_allow_html=True)
        days_map = {"5D": 5, "1M": 30, "3M": 90, "6M": 180}
        range_sel = st.radio("Range", list(days_map.keys()), horizontal=True, index=1, label_visibility="collapsed")
        fig = price_sentiment_chart(st.session_state.ticker, days_map[range_sel])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with sig_col:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:12px">Live Signals</div>', unsafe_allow_html=True)
        for tk in TICKERS:
            sig  = st.session_state.signals.get(tk, {})
            d_   = get_prices(tk, 2)
            color = "#10b981" if sig.get("dir") == "BULLISH" else "#f43f5e" if sig.get("dir") == "BEARISH" else "#94a3b8"
            st.markdown(f"""
            <div style="background:#0d1b2e;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:12px 14px;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between">
              <div style="display:flex;align-items:center;gap:10px">
                <div style="width:32px;height:32px;border-radius:7px;background:#fff;display:flex;align-items:center;justify-content:center;padding:3px"><img src="{LOGOS[tk]}" width="24" height="24" style="object-fit:contain"></div>
                <div><div style="font-weight:600;color:#e8f4ff;font-size:13px">{tk}</div><div style="font-size:10px;color:#4a6380">${d_["current"]:,.2f}</div></div>
              </div>
              <div style="text-align:right">
                <div style="font-size:11px;font-weight:700;color:{color};background:{color}1a;padding:3px 8px;border-radius:5px">{sig.get("dir","—")}</div>
                <div style="font-size:10px;color:#4a6380;margin-top:3px">{sig.get("conf",70)}% conf</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── BOTTOM ROW ──
    st.markdown("---")
    bot1, bot2 = st.columns(2)
    with bot1:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:10px">Stock Overview</div>', unsafe_allow_html=True)
        rows = []
        for tk in TICKERS:
            d_ = get_prices(tk, 2)
            sig_ = st.session_state.signals.get(tk, {})
            rows.append({
                "":          f'<img src="{LOGOS[tk]}" width="20" height="20" style="object-fit:contain;vertical-align:middle;background:#fff;border-radius:4px;padding:2px">',
                "Ticker":    tk,
                "Company":   TK_NAMES[tk],
                "Price":     f"${d_['current']:,.2f}",
                "Change":    f"{d_['change_pct']:+.2f}%",
                "Sentiment": f"{smap.get(tk, 0):+.2f}",
                "Signal":    sig_.get("dir", "—"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={
                         "":      st.column_config.Column("", width="small"),
                         "Signal": st.column_config.TextColumn("Signal"),
                     })
    with bot2:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:10px">Sentiment Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(sentiment_comparison_chart(), use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# ══ PAGE: ANALYSIS ══
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Analysis":
    st.markdown('<div class="se-section-title">Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="se-section-sub">Paste any financial text — headline, tweet, or Reddit post — and get a sentiment score and next-day direction prediction.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#7a9ab8;margin-bottom:8px">Select Ticker</div>', unsafe_allow_html=True)
        anl_tk = st.selectbox("Ticker", TICKERS, index=TICKERS.index(st.session_state.ticker), label_visibility="collapsed", key="anl_tk")

        st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#7a9ab8;margin-bottom:8px;margin-top:14px">Financial Text</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Text",
            placeholder=f"e.g. {anl_tk} beats Q2 earnings expectations — EPS $1.53 vs $1.38 estimate, strong guidance…",
            height=120, label_visibility="collapsed",
        )

        src = st.selectbox("Source", ["News Headline", "Tweet / X Post", "Reddit Post"], label_visibility="collapsed")

        run = st.button("Analyze & Predict Direction", use_container_width=True)

        if run and text_input.strip():
            with st.spinner("Running sentiment analysis…"):
                result = predict_backend(anl_tk, text_input.strip()) if is_live else None
                source_used = "Flask backend" if result else "local VADER"
                if not result:
                    result = local_sentiment(text_input.strip())
                st.session_state.last_result = result
                st.session_state.anal_count += 1
                # Update signals
                st.session_state.signals[anl_tk] = {"dir": result["direction"], "conf": result["confidence"], "src": src}
                # Add to log
                st.session_state.analysis_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "ticker": anl_tk,
                    "text": text_input.strip()[:72],
                    "dir": result["direction"],
                    "conf": result["confidence"],
                })
            st.success(f"Done via {source_used}", icon="✓")

        elif run:
            st.warning("Please enter some text before running analysis.")

        # RSI chart for context
        st.markdown("---")
        st.markdown(f'<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:8px">{anl_tk} — RSI Technical Context</div>', unsafe_allow_html=True)
        rsi_fig, last_rsi = rsi_chart(anl_tk)
        zone = "Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral"
        st.markdown(f'<div style="font-size:11px;color:#7a9ab8;margin-bottom:6px">Current RSI: <span class="se-mono" style="color:{"#f43f5e" if last_rsi > 70 else "#10b981" if last_rsi < 30 else "#60a5fa"}">{last_rsi:.1f}</span> — {zone}</div>', unsafe_allow_html=True)
        st.plotly_chart(rsi_fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        result = st.session_state.last_result
        if result:
            s    = result["sentiment"]
            d    = result["direction"]
            conf = result["confidence"]
            comp = s["compound"]

            # Verdict
            cls   = "bull" if d == "BULLISH" else "bear" if d == "BEARISH" else "neut"
            arrow = "↑" if d == "BULLISH" else "↓" if d == "BEARISH" else "→"
            st.markdown(f"""
            <div class="verdict-{cls}" style="margin-bottom:16px">
              <div class="verdict-arrow">{arrow}</div>
              <div class="verdict-word-{cls}">{d}</div>
              <div style="font-size:11px;color:#7a9ab8;margin-top:6px">{anl_tk} · next-day direction signal</div>
            </div>""", unsafe_allow_html=True)

            # Gauges
            g1, g2 = st.columns(2)
            with g1:
                st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#7a9ab8;text-align:center">Confidence</div>', unsafe_allow_html=True)
                c = UP_COLOR if conf >= 70 else GOLD if conf >= 55 else DN_COLOR
                st.plotly_chart(confidence_gauge(conf, c), use_container_width=True, config={"displayModeBar": False})
            with g2:
                st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#7a9ab8;text-align:center">Compound Score</div>', unsafe_allow_html=True)
                st.plotly_chart(sentiment_gauge(comp), use_container_width=True, config={"displayModeBar": False})

            # Sentiment bars
            st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#7a9ab8;margin-bottom:10px">Sentiment Breakdown</div>', unsafe_allow_html=True)
            for label, val, color in [
                ("Positive", s["positive"], UP_COLOR),
                ("Neutral",  s["neutral"],  "#64748b"),
                ("Negative", s["negative"], DN_COLOR),
            ]:
                pct = round(val * 100, 1)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                  <div style="font-size:11px;color:#7a9ab8;min-width:56px;font-family:'JetBrains Mono'">{label}</div>
                  <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:7px;overflow:hidden">
                    <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;transition:width .7s ease"></div>
                  </div>
                  <div style="font-size:11px;color:#e8f4ff;min-width:36px;text-align:right;font-family:'JetBrains Mono'">{pct}%</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1b2e;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:40px;text-align:center;color:#4a6380;margin-bottom:16px">
              <div style="font-size:28px;margin-bottom:10px">📊</div>
              <div style="font-size:13px">Enter text on the left and click<br><strong style="color:#7a9ab8">Analyze & Predict</strong> to see results</div>
            </div>""", unsafe_allow_html=True)

        # Analysis Log
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:10px;margin-top:8px">Analysis Log</div>', unsafe_allow_html=True)
        log = st.session_state.analysis_log
        if log:
            rows = []
            for e in log[:12]:
                badge_color = "#10b981" if e["dir"] == "BULLISH" else "#f43f5e" if e["dir"] == "BEARISH" else "#94a3b8"
                rows.append({
                    "Time":   e["time"],
                    "Ticker": e["ticker"],
                    "Text":   e["text"] + ("…" if len(e["text"]) == 72 else ""),
                    "Signal": e["dir"],
                    "Conf.":  f"{e['conf']}%",
                })
            df_log = pd.DataFrame(rows)
            st.dataframe(df_log, hide_index=True, use_container_width=True,
                         column_config={
                             "Signal": st.column_config.TextColumn("Signal"),
                             "Time": st.column_config.TextColumn("Time", width="small"),
                             "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                         })
        else:
            st.markdown('<div style="color:#4a6380;font-size:12px;text-align:center;padding:20px">No analyses yet</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ══ PAGE: MARKET PERFORMANCE ══
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Market":
    st.markdown('<div class="se-section-title">Model Performance</div>', unsafe_allow_html=True)
    results = st.session_state.results
    src_label = "Live — from /api/results" if results else "Demo values — run train.py to populate"
    st.markdown(f'<div class="se-section-sub">{src_label}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:8px">Accuracy & F1 by Ticker</div>', unsafe_allow_html=True)
        st.plotly_chart(accuracy_bar_chart(results), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:8px">Performance Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(results), use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:8px">Realised Volatility (10d)</div>', unsafe_allow_html=True)
        st.plotly_chart(volatility_chart(), use_container_width=True, config={"displayModeBar": False})
    with c4:
        st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:8px">MACD — Active Ticker</div>', unsafe_allow_html=True)
        st.plotly_chart(macd_chart(st.session_state.ticker), use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown('<div style="font-size:14px;font-weight:600;color:#e8f4ff;margin-bottom:12px">Full Metrics Table</div>', unsafe_allow_html=True)
    table_rows = []
    for tk in TICKERS:
        r = results.get(tk, {})
        table_rows.append({
            "Ticker":    tk,
            "Company":   TK_NAMES[tk],
            "Accuracy":  f"{round(r.get('accuracy',  {'AAPL':0.68,'TSLA':0.65,'NVDA':0.71}[tk])*100, 1)}%",
            "Precision": f"{round(r.get('precision', {'AAPL':0.71,'TSLA':0.63,'NVDA':0.73}[tk])*100, 1)}%",
            "Recall":    f"{round(r.get('recall',    {'AAPL':0.65,'TSLA':0.67,'NVDA':0.69}[tk])*100, 1)}%",
            "F1 Score":  f"{r.get('f1_score',  {'AAPL':0.674,'TSLA':0.641,'NVDA':0.703}[tk]):.3f}",
            "AUC-ROC":   f"{r.get('auc_roc',   {'AAPL':0.718,'TSLA':0.697,'NVDA':0.741}[tk]):.3f}",
            "Test Samples": r.get("test_samples", "—"),
            "Source":    "Live" if results else "Demo",
        })
    df_metrics = pd.DataFrame(table_rows)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # Model insight
    st.markdown("---")
    best = max(["AAPL","TSLA","NVDA"], key=lambda t: float(df_metrics[df_metrics["Ticker"]==t]["AUC-ROC"].values[0]))
    st.info(f"**Best performing model: {best}** — {TK_NAMES[best]}. Higher AUC-ROC indicates stronger discriminative ability between UP and DOWN days. Sentiment fusion (FinBERT + VADER) accounts for the improvement over price-only baseline (~52% accuracy).", icon="💡")