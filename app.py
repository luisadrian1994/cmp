import time
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# ---------- SETTINGS ----------
# -----------------------------
DEFAULT_LOOKBACK_YEARS = 2
MIN_HISTORY_DAYS = 200  # need enough bars for long MAs
CROSS_WINDOW_DAYS = 7   # how recent a cross must be to count as "starting"
TREND_MIN_DAYS = 15     # require at least this many days for "ongoing" trend
UNIVERSE = "S&P 500"    # default universe on load

# -----------------------------
# ----- HELPER FUNCTIONS ------
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def fetch_sp500_tickers():
    """
    Pull S&P 500 tickers from Wikipedia.
    If it fails, fall back to a compact list.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = requests.get(url, timeout=10).text
        tables = pd.read_html(html)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Some tickers have dots (e.g., BRK.B). yfinance expects '-' instead of '.'
        tickers = [t.replace('.', '-') for t in tickers]
        return sorted(set(tickers))
    except Exception:
        # Small fallback list to ensure app still works offline
        return sorted(set("""
AAPL MSFT AMZN NVDA GOOGL META BRK-B UNH XOM JNJ V JPM PG AVGO LLY
""".split()))

def get_price_history(ticker, years=DEFAULT_LOOKBACK_YEARS):
    end = datetime.now()
    start = end - timedelta(days=365 * years + 30)
    try:
        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        data = data.rename(columns=str.title)  # ensure 'Close', 'High', 'Low', 'Volume'
        return data
    except Exception:
        return None

def compute_indicators(df: pd.DataFrame):
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["SMA200"] = sma(out["Close"], 200)
    out["RSI14"] = rsi(out["Close"], 14)
    out["MACD"], out["MACDsig"], out["MACDhist"] = macd(out["Close"])
    return out

def last_cross_dates(short: pd.Series, long: pd.Series):
    """
    Detect last bullish and bearish cross dates (short vs long).
    Returns (bullish_date, bearish_date, days_since_bull, days_since_bear)
    """
    cross = np.sign(short - long)
    cross_shift = cross.shift(1)
    # Bullish cross: went from negative to positive
    bull_idx = (cross > 0) & (cross_shift < 0)
    # Bearish cross: went from positive to negative
    bear_idx = (cross < 0) & (cross_shift > 0)
    bull_date = bull_idx[bull_idx].index.max() if bull_idx.any() else None
    bear_date = bear_idx[bear_idx].index.max() if bear_idx.any() else None

    today = short.index.max()
    days_since_bull = (today - bull_date).days if bull_date is not None else None
    days_since_bear = (today - bear_date).days if bear_date is not None else None
    return bull_date, bear_date, days_since_bull, days_since_bear

def classify_trend(row_window: pd.DataFrame):
    """
    Classify uptrend / downtrend / neutral using EMAs + price relative to SMA200.
    """
    close = row_window["Close"].iloc[-1]
    ema20 = row_window["EMA20"].iloc[-1]
    ema50 = row_window["EMA50"].iloc[-1]
    sma200 = row_window["SMA200"].iloc[-1]
    macd_hist = row_window["MACDhist"].iloc[-1]

    if np.isnan(ema20) or np.isnan(ema50) or np.isnan(sma200):
        return "Insufficient data"

    # Baseline trend
    if ema20 > ema50 and close > sma200:
        trend = "Uptrend"
    elif ema20 < ema50 and close < sma200:
        trend = "Downtrend"
    else:
        # Mixed state
        trend = "Neutral"

    return trend

def trend_stage(df: pd.DataFrame, trend: str):
    """
    Stage estimation using recency of cross + RSI + MACD histogram slope + extension from EMA50.
    """
    bull_date, bear_date, days_bull, days_bear = last_cross_dates(df["EMA20"], df["EMA50"])
    rsi_now = float(df["RSI14"].iloc[-1])
    macd_hist = df["MACDhist"].iloc[-10:]  # recent window
    macd_slope = macd_hist.diff().iloc[-3:].mean()  # average recent slope
    close = float(df["Close"].iloc[-1])
    ema50_now = float(df["EMA50"].iloc[-1])
    ext50 = (close - ema50_now) / ema50_now if ema50_now else 0.0
    days = days_bull if trend == "Uptrend" else days_bear

    if trend == "Uptrend":
        if days is not None and days <= 7 and macd_hist.iloc[-1] > 0 and rsi_now >= 45 and rsi_now <= 70:
            return "Beginning"
        if days is not None and days >= 45:
            return "Ending"
        if rsi_now > 70 or ext50 > 0.10 or macd_slope < 0:
            return "Ending"
        return "Middle"

    if trend == "Downtrend":
        if days is not None and days <= 7 and macd_hist.iloc[-1] < 0 and rsi_now <= 55 and rsi_now >= 25:
            return "Beginning"
        if days is not None and days >= 45:
            return "Ending"
        if rsi_now < 30 or ext50 < -0.10 or macd_slope > 0:
            return "Ending"
        return "Middle"

    return "—"

def is_starting_uptrend(df: pd.DataFrame):
    """
    Conditions that suggest the *start* of an uptrend in the last few days:
    - 20EMA crossed above 50EMA within CROSS_WINDOW_DAYS
    - MACD histogram turned positive within CROSS_WINDOW_DAYS
    - RSI between ~45 and 65 (not overbought yet)
    - Price above 200SMA
    """
    bull_date, _, days_bull, _ = last_cross_dates(df["EMA20"], df["EMA50"])
    if days_bull is None or days_bull > CROSS_WINDOW_DAYS:
        return False

    recent = df.tail(CROSS_WINDOW_DAYS + 2)
    macd_pos_recent = recent["MACDhist"].iloc[-1] > 0 and (recent["MACDhist"] > 0).any()
    rsi_now = recent["RSI14"].iloc[-1]
    above_200 = recent["Close"].iloc[-1] > recent["SMA200"].iloc[-1]
    return macd_pos_recent and (45 <= rsi_now <= 65) and above_200

def is_starting_downtrend(df: pd.DataFrame):
    """
    Mirror conditions for downtrend start:
    - 20EMA crossed below 50EMA within CROSS_WINDOW_DAYS
    - MACD histogram turned negative within CROSS_WINDOW_DAYS
    - RSI between ~35 and 55 (not oversold yet)
    - Price below 200SMA
    """
    _, bear_date, _, days_bear = last_cross_dates(df["EMA20"], df["EMA50"])
    if days_bear is None or days_bear > CROSS_WINDOW_DAYS:
        return False

    recent = df.tail(CROSS_WINDOW_DAYS + 2)
    macd_neg_recent = recent["MACDhist"].iloc[-1] < 0 and (recent["MACDhist"] < 0).any()
    rsi_now = recent["RSI14"].iloc[-1]
    below_200 = recent["Close"].iloc[-1] < recent["SMA200"].iloc[-1]
    return macd_neg_recent and (35 <= rsi_now <= 55) and below_200

def starting_score(df: pd.DataFrame, bullish=True):
    """
    Rank candidates: more recent cross, stronger (but not extreme) momentum,
    and volume heartbeat.
    """
    bull_date, bear_date, days_bull, days_bear = last_cross_dates(df["EMA20"], df["EMA50"])
    days = days_bull if bullish else days_bear
    if days is None:
        return -np.inf

    # recency score: newer is better
    recency = max(0, (CROSS_WINDOW_DAYS + 1) - days)

    # momentum: use |MACDhist| scaled, but penalize extremes (too extended)
    macd_h = abs(df["MACDhist"].iloc[-1])
    rsi_now = df["RSI14"].iloc[-1]
    penalty = 0.0
    if bullish and rsi_now > 68: penalty = 2.0
    if (not bullish) and rsi_now < 32: penalty = 2.0

    # volume pulse: compare last volume to 20d average
    vol = df["Volume"].iloc[-1] if "Volume" in df.columns else np.nan
    vol20 = df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else np.nan
    vol_boost = 0.0
    if not np.isnan(vol) and not np.isnan(vol20) and vol20 > 0:
        vol_boost = min(3.0, (vol / vol20) - 1.0)  # cap boost

    return recency + (macd_h * 2) + vol_boost - penalty

def analyze_ticker(ticker: str, years=DEFAULT_LOOKBACK_YEARS):
    hist = get_price_history(ticker, years)
    if hist is None or len(hist) < MIN_HISTORY_DAYS:
        return None, None
    df = compute_indicators(hist)
    trend = classify_trend(df)
    stage = trend_stage(df, trend)
    return df, {"trend": trend, "stage": stage}

def make_chart(df: pd.DataFrame, ticker: str):
    df = df.dropna().copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200", mode="lines"))

    fig.update_layout(
        title=f"{ticker} — Price with EMA20 / EMA50 / SMA200",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        height=520,
    )
    return fig

# -----------------------------
# ------------ UI -------------
# -----------------------------
st.set_page_config(page_title="Trend Finder — 3‑Month Swing Scanner", page_icon="📈", layout="wide")
st.title("📈 Trend Finder — 3‑Month Swing Scanner")

colA, colB = st.columns([2, 1])
with colA:
    st.write(
        "Find **top 5 stocks starting an uptrend** (buy-and-hold ~3 months), "
        "**top 5 starting a downtrend**, and analyze any ticker’s **trend** and **stage**."
    )
with colB:
    st.info("Educational only. Not investment advice.")

with st.sidebar:
    st.header("Scanner Settings")
    universe_choice = st.selectbox("Universe", ["S&P 500", "Custom List"])
    custom_list = ""
    if universe_choice == "Custom List":
        custom_list = st.text_area(
            "Enter tickers separated by space or comma",
            value="AAPL MSFT AMZN NVDA GOOGL META JPM JNJ TSLA AMD"
        )
    years = st.slider("Price history (years)", 1, 5, DEFAULT_LOOKBACK_YEARS, 1)
    st.caption("We compute EMAs/RSI/MACD from this history window.")
    scan_button = st.button("🔎 Run Scan")

    st.header("Single Ticker Check")
    ticker_input = st.text_input("Ticker (e.g., AAPL)", value="AAPL")
    check_button = st.button("Check Ticker")

# Universe
if universe_choice == "S&P 500":
    tickers = fetch_sp500_tickers()
else:
    tickers = [t.strip().upper() for t in custom_list.replace(",", " ").split() if t.strip()]

# -------- Single Ticker Check --------
if check_button and ticker_input.strip():
    t = ticker_input.strip().upper()
    with st.spinner(f"Analyzing {t}..."):
        df, info = analyze_ticker(t, years=years)
    if df is None:
        st.error(f"Could not retrieve enough data for {t}.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Trend", info["trend"])
        with c2:
            st.metric("Stage", info["stage"])
        with c3:
            st.caption("Trend uses EMA20 vs EMA50 and price vs SMA200. Stage uses recency of crossover, RSI, MACD histogram slope, and extension from EMA50.")
        st.plotly_chart(make_chart(df, t), use_container_width=True)

        # Show latest indicators
        last = df.iloc[-1]
        meta = pd.DataFrame({
            "Close": [round(last["Close"], 2)],
            "EMA20": [round(last["EMA20"], 2)],
            "EMA50": [round(last["EMA50"], 2)],
            "SMA200": [round(last["SMA200"], 2)],
            "RSI14": [round(last["RSI14"], 1)],
            "MACD": [round(last["MACD"], 3)],
            "MACDsig": [round(last["MACDsig"], 3)],
            "MACDhist": [round(last["MACDhist"], 3)]
        })
        st.dataframe(meta, use_container_width=True)

st.markdown("---")

# -------- Scanner --------
if scan_button:
    st.subheader(f"Scanner: {UNIVERSE if universe_choice=='S&P 500' else 'Custom List'}")

    progress = st.progress(0)
    results = []
    total = len(tickers)
    scanned = 0

    for t in tickers:
        scanned += 1
        progress.progress(min(1.0, scanned / total))
        df = get_price_history(t, years=years)
        if df is None or len(df) < MIN_HISTORY_DAYS:
            continue
        df = compute_indicators(df).dropna()
        if len(df) < MIN_HISTORY_DAYS:
            continue

        # Determine candidates
        up_start = is_starting_uptrend(df)
        down_start = is_starting_downtrend(df)
        trend_now = classify_trend(df)
        stage_now = trend_stage(df, trend_now)

        row = {
            "Ticker": t,
            "Close": float(df["Close"].iloc[-1]),
            "Trend": trend_now,
            "Stage": stage_now,
            "BullStart": up_start,
            "BearStart": down_start,
            "BullScore": starting_score(df, bullish=True),
            "BearScore": starting_score(df, bullish=False),
        }
        results.append(row)

    progress.progress(1.0)
    st.success(f"Scanned {len(results)} tickers.")

    if not results:
        st.warning("No results. Try a different universe or shorter lookback.")
    else:
        res_df = pd.DataFrame(results)

        # Top 5 starting uptrends
        bull = res_df[res_df["BullStart"]].copy()
        bull = bull.sort_values(by=["BullScore", "Close"], ascending=[False, False]).head(5)

        # Top 5 starting downtrends
        bear = res_df[res_df["BearStart"]].copy()
        bear = bear.sort_values(by=["BearScore", "Close"], ascending=[False, True]).head(5)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔼 Top 5 Starting Uptrends")
            if bull.empty:
                st.info("No fresh uptrend starts found under current rules.")
            else:
                st.dataframe(
                    bull[["Ticker", "Close", "Trend", "Stage", "BullScore"]].reset_index(drop=True),
                    use_container_width=True
                )
        with col2:
            st.markdown("### 🔽 Top 5 Starting Downtrends")
            if bear.empty:
                st.info("No fresh downtrend starts found under current rules.")
            else:
                st.dataframe(
                    bear[["Ticker", "Close", "Trend", "Stage", "BearScore"]].reset_index(drop=True),
                    use_container_width=True
                )

        st.caption(
            "Scoring favors **more recent EMA20/EMA50 crossovers**, **healthy (not extreme) momentum**, "
            "and a mild **volume pulse**. ‘Stage’ uses recency of cross, RSI bounds, MACD‑hist slope, "
            "and distance from EMA50."
        )

        # Optional: click-to-chart
        st.markdown("#### Quick Chart")
        pick = st.selectbox("Choose a ticker to view", options=sorted(res_df["Ticker"].unique()))
        if pick:
            df, _ = analyze_ticker(pick, years=years)
            if df is not None:
                st.plotly_chart(make_chart(df, pick), use_container_width=True)

# Footer disclaimer
st.markdown(
    """
    <hr/>
    <small>
    This tool is for **educational purposes** only and does not constitute financial advice or a recommendation to buy/sell any security.
    Markets involve risk. Always do your own research.
    </small>
    """,
    unsafe_allow_html=True
)
