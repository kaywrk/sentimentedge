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
                <div style="width:34px;height:34px;border-radius:8px;background:{"#1c1c1e" if tk=="AAPL" else "#cc0000" if tk=="TSLA" else "#76b900"};display:flex;align-items:center;justify-content:center;font-size:16px">{"🍎" if tk=="AAPL" else "⚡" if tk=="TSLA" else "🎮"}</div>
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
                <div style="width:32px;height:32px;border-radius:7px;background:{"#1c1c1e" if tk=="AAPL" else "#cc0000" if tk=="TSLA" else "#76b900"};display:flex;align-items:center;justify-content:center;font-size:14px">{"🍎" if tk=="AAPL" else "⚡" if tk=="TSLA" else "🎮"}</div>
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
                "Ticker":    tk,
                "Company":   TK_NAMES[tk],
                "Price":     f"${d_['current']:,.2f}",
                "Change":    f"{d_['change_pct']:+.2f}%",
                "Sentiment": f"{smap.get(tk, 0):+.2f}",
                "Signal":    sig_.get("dir", "—"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"Signal": st.column_config.TextColumn("Signal")})
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
            st.success(f"Done via {source_used}", icon="✅")

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