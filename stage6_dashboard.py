"""
=============================================================
STAGE 6: STREAMLIT DASHBOARD & AUTOMATIC ALERT MONITOR
=============================================================
PURPOSE:
    Two-tab interactive dashboard:

    TAB 1 — Stock Analysis
        Select a sector and stock, view price chart,
        risk regime, class probabilities, and recent
        predictions.

    TAB 2 — Auto Alert Monitor
        Automatically scans ALL 50 S&P 500 stocks and
        displays a live alert table showing every stock
        currently in High Risk or Medium Risk regime,
        sorted by High-Risk probability (highest first).
        Includes a Refresh button and auto-refresh timer.

HOW TO RUN:
    streamlit run stage6_dashboard.py

NOTE ON DATA DELAY:
    Yahoo Finance introduces a ~15-20 minute delay.
    This dashboard reflects near-real-time prices.
=============================================================
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Regime Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# SECTOR MAP
# ─────────────────────────────────────────────────────────────
SECTORS = {
    "💻 Technology": {
        "AAPL":  "Apple Inc.",
        "MSFT":  "Microsoft Corporation",
        "NVDA":  "NVIDIA Corporation",
        "GOOGL": "Alphabet Inc. (Class A)",
        "META":  "Meta Platforms Inc.",
        "AVGO":  "Broadcom Inc.",
        "CRM":   "Salesforce Inc.",
        "ORCL":  "Oracle Corporation",
        "AMD":   "Advanced Micro Devices",
        "CSCO":  "Cisco Systems Inc.",
        "ADBE":  "Adobe Inc.",
        "INTU":  "Intuit Inc.",
        "QCOM":  "Qualcomm Inc.",
        "IBM":   "International Business Machines",
        "TXN":   "Texas Instruments Inc.",
    },
    "🛒 Consumer Discretionary": {
        "AMZN":  "Amazon.com Inc.",
        "TSLA":  "Tesla Inc.",
        "MCD":   "McDonald's Corporation",
        "HD":    "Home Depot Inc.",
    },
    "🏥 Healthcare": {
        "LLY":   "Eli Lilly and Company",
        "UNH":   "UnitedHealth Group",
        "JNJ":   "Johnson & Johnson",
        "MRK":   "Merck & Co.",
        "ABBV":  "AbbVie Inc.",
        "TMO":   "Thermo Fisher Scientific",
        "ABT":   "Abbott Laboratories",
        "AMGN":  "Amgen Inc.",
        "DHR":   "Danaher Corporation",
    },
    "🏦 Financials": {
        "BRK-B": "Berkshire Hathaway (Class B)",
        "JPM":   "JPMorgan Chase & Co.",
        "V":     "Visa Inc.",
        "MA":    "Mastercard Inc.",
        "BAC":   "Bank of America Corporation",
        "SPGI":  "S&P Global Inc.",
    },
    "⛽ Energy": {
        "XOM":   "ExxonMobil Corporation",
        "CVX":   "Chevron Corporation",
    },
    "🛍️ Consumer Staples": {
        "PG":    "Procter & Gamble",
        "COST":  "Costco Wholesale Corporation",
        "KO":    "The Coca-Cola Company",
        "PEP":   "PepsiCo Inc.",
        "WMT":   "Walmart Inc.",
        "PM":    "Philip Morris International",
    },
    "📡 Communication Services": {
        "NFLX":  "Netflix Inc.",
    },
    "🏭 Industrials": {
        "CAT":   "Caterpillar Inc.",
        "GE":    "GE Aerospace",
        "BA":    "Boeing Company",
        "RTX":   "RTX Corporation (Raytheon)",
        "LIN":   "Linde PLC",
        "ACN":   "Accenture PLC",
    },
    "⚡ Utilities": {
        "NEE":   "NextEra Energy Inc.",
    },
}

# Flat ticker -> (company, sector) lookup
ALL_STOCKS = {}
for sector_name, stocks in SECTORS.items():
    for ticker, company in stocks.items():
        ALL_STOCKS[ticker] = {"company": company, "sector": sector_name}

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Log_Return", "MA10", "MA20", "Volatility_20", "Momentum_10"
]
CLASS_NAMES   = ["Low Risk", "Medium Risk", "High Risk"]
CLASS_COLOURS = ["#2ecc71", "#f39c12", "#e74c3c"]
WINDOW_SIZE   = 60
MODEL_PATH    = "models/cnn_lstm_best.keras"
SCALER_PATH   = "data/processed/sequences/scaler.pkl"


# ─────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False, ttl=900)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download OHLCV data. Cached for 15 minutes."""
    try:
        df = yf.download(ticker, period=period,
                         auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Log_Return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["MA10"]          = df["Close"].rolling(10).mean()
    df["MA20"]          = df["Close"].rolling(20).mean()
    df["Volatility_20"] = df["Log_Return"].rolling(20).std()
    df["Momentum_10"]   = df["Close"] - df["Close"].shift(10)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# BATCHED PREDICTION
# ─────────────────────────────────────────────────────────────
def run_predictions(df: pd.DataFrame, model, scaler):
    """Run all window predictions in one batched call."""
    scaled = df.copy()
    scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    feature_arr = scaled[FEATURE_COLS].values

    n_windows = len(feature_arr) - WINDOW_SIZE
    if n_windows <= 0:
        return [], [], []

    X_all = np.array([
        feature_arr[i: i + WINDOW_SIZE]
        for i in range(n_windows)
    ], dtype=np.float32)

    probas_all = model.predict(X_all, batch_size=256, verbose=0)
    labels_all = np.argmax(probas_all, axis=1)

    dates  = [df["Date"].iloc[i + WINDOW_SIZE] for i in range(n_windows)]
    labels = [int(l) for l in labels_all]
    probas = [probas_all[i] for i in range(n_windows)]

    return dates, labels, probas


def get_latest_prediction(ticker: str, model, scaler):
    """
    Fetch data for one ticker and return only the latest
    prediction (label + probabilities). Used by the
    Auto Alert Monitor to quickly scan all 50 stocks.
    """
    df_raw = fetch_stock_data(ticker, period="6mo")
    if df_raw.empty or len(df_raw) < WINDOW_SIZE + 25:
        return None

    df_feat = compute_features(df_raw)
    if len(df_feat) < WINDOW_SIZE + 1:
        return None

    # Only predict on the last window for speed
    scaled = df_feat.copy()
    scaled[FEATURE_COLS] = scaler.transform(df_feat[FEATURE_COLS])
    feature_arr = scaled[FEATURE_COLS].values

    window = feature_arr[-WINDOW_SIZE:]
    X      = window[np.newaxis, :, :].astype(np.float32)
    proba  = model.predict(X, verbose=0)[0]
    label  = int(np.argmax(proba))

    return {
        "label"      : label,
        "regime"     : CLASS_NAMES[label],
        "p_low"      : float(proba[0]),
        "p_medium"   : float(proba[1]),
        "p_high"     : float(proba[2]),
        "close"      : float(df_raw["Close"].iloc[-1]),
        "date"       : df_raw["Date"].iloc[-1].strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────────────────────
# ALERT CHECK
# ─────────────────────────────────────────────────────────────
def check_alerts(labels, probas, threshold):
    if not labels:
        return {}
    latest_label = labels[-1]
    latest_proba = probas[-1]
    prev_label   = labels[-2] if len(labels) > 1 else None
    return {
        "high_risk_alert": (latest_proba[2] >= threshold),
        "regime_changed" : (prev_label is not None and
                            latest_label != prev_label),
        "latest_label"   : latest_label,
        "latest_proba"   : latest_proba,
        "prev_label"     : prev_label,
    }


# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────
def hex_to_rgba(hex_colour: str, alpha: float = 0.6) -> str:
    hex_colour = hex_colour.lstrip("#")
    r = int(hex_colour[0:2], 16)
    g = int(hex_colour[2:4], 16)
    b = int(hex_colour[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def regime_badge(label: int) -> str:
    """Return a coloured emoji badge for a regime label."""
    return ["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"][label]


# ─────────────────────────────────────────────────────────────
# PLOT: PRICE + REGIME
# ─────────────────────────────────────────────────────────────
def plot_price_and_regime(df_raw, dates, labels):
    colours = [CLASS_COLOURS[l] for l in labels]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=[
            "Price Chart (Candlestick + Moving Averages)",
            "Risk Regime Classification"
        ]
    )

    fig.add_trace(go.Candlestick(
        x=df_raw["Date"],
        open=df_raw["Open"], high=df_raw["High"],
        low=df_raw["Low"],   close=df_raw["Close"],
        name="Price", showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_raw["Date"],
        y=df_raw["Close"].rolling(10).mean(),
        mode="lines", name="MA10",
        line=dict(color="orange", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_raw["Date"],
        y=df_raw["Close"].rolling(20).mean(),
        mode="lines", name="MA20",
        line=dict(color="royalblue", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=pd.Series(dates),
        y=[1] * len(labels),
        marker_color=colours,
        name="Regime",
        showlegend=False,
        hovertext=[CLASS_NAMES[l] for l in labels],
        hoverinfo="x+text"
    ), row=2, col=1)

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(l=60, r=20, t=40, b=20),
    )
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    return fig


# ─────────────────────────────────────────────────────────────
# PLOT: PROBABILITIES
# ─────────────────────────────────────────────────────────────
def plot_probabilities(dates, probas):
    prob_arr = np.array(probas)
    fig = go.Figure()

    for i, (name, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS)):
        fig.add_trace(go.Scatter(
            x=dates, y=prob_arr[:, i],
            name=name,
            stackgroup="one",
            fillcolor=hex_to_rgba(colour, alpha=0.6),
            line=dict(color=colour, width=1),
            mode="lines"
        ))

    fig.update_layout(
        title="Model Class Probabilities Over Time",
        xaxis_title="Date",
        yaxis_title="Probability",
        height=300,
        margin=dict(l=60, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────
# TAB 1 — STOCK ANALYSIS
# ─────────────────────────────────────────────────────────────
def render_stock_analysis(model, scaler):

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Stock Analysis Settings")

        sector_names    = list(SECTORS.keys())
        selected_sector = st.selectbox("① Select Sector",
                                       options=sector_names, index=0)

        sector_stocks   = SECTORS[selected_sector]
        ticker_options  = {f"{t}  —  {n}": t
                           for t, n in sector_stocks.items()}
        selected_label  = st.selectbox("② Select Stock",
                                       options=list(ticker_options.keys()),
                                       index=0)
        ticker = ticker_options[selected_label]

        period = st.selectbox("③ Data Period",
                              options=["6mo", "1y", "2y", "5y"],
                              index=2)

        alert_threshold = st.slider(
            "④ High-Risk Alert Threshold",
            min_value=0.30, max_value=0.95,
            value=0.60, step=0.05,
            help="Alert fires when P(High Risk) >= this value."
        )

        st.divider()
        st.markdown(f"**{selected_sector}** — {len(sector_stocks)} stocks")
        for t, n in sector_stocks.items():
            marker = "▶" if t == ticker else "·"
            st.markdown(f"`{marker}` **{t}** — {n}")

        st.divider()
        st.markdown("**Model:** Hybrid CNN-LSTM")
        st.markdown("**Window:** 60 trading days")
        st.markdown("**Data:** Yahoo Finance (~15 min delay)")

    # ── Fetch & predict ───────────────────────────────────
    company_name = sector_stocks[ticker]
    st.subheader(f"{selected_sector}  ›  {ticker} — {company_name}")

    with st.spinner(f"Fetching {ticker} data..."):
        df_raw = fetch_stock_data(ticker, period)

    if df_raw.empty or len(df_raw) < WINDOW_SIZE + 25:
        st.error(f"Not enough data for {ticker}. Try a longer period.")
        return

    df_feat = compute_features(df_raw)

    with st.spinner("Running predictions..."):
        dates, labels, probas = run_predictions(df_feat, model, scaler)

    if not dates:
        st.error("Not enough data for predictions.")
        return

    # ── Alerts ────────────────────────────────────────────
    alert = check_alerts(labels, probas, alert_threshold)

    if alert.get("high_risk_alert"):
        st.error(
            f"🚨 **HIGH-RISK ALERT — {ticker} ({company_name})**\n\n"
            f"P(High Risk) = **{alert['latest_proba'][2]:.1%}** "
            f"exceeds threshold of {alert_threshold:.0%}. "
            f"Consider reviewing portfolio exposure."
        )
    elif alert.get("regime_changed"):
        prev   = CLASS_NAMES[alert["prev_label"]]
        latest = CLASS_NAMES[alert["latest_label"]]
        st.warning(f"⚡ **REGIME TRANSITION — {ticker}**\n\n{prev}  →  **{latest}**")
    else:
        st.success(
            f"✅ Current Regime: **{CLASS_NAMES[alert['latest_label']]}** "
            f"— No alert for {ticker}."
        )

    # ── Metrics ───────────────────────────────────────────
    latest_proba = alert["latest_proba"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sector",         selected_sector.split(" ", 1)[1])
    c2.metric("Latest Close",   f"${float(df_raw['Close'].iloc[-1]):.2f}")
    c3.metric("P(Low Risk)",    f"{latest_proba[0]:.1%}")
    c4.metric("P(Medium Risk)", f"{latest_proba[1]:.1%}")
    c5.metric("P(High Risk)",   f"{latest_proba[2]:.1%}",
              delta="ALERT" if alert["high_risk_alert"] else "OK",
              delta_color="inverse" if alert["high_risk_alert"] else "off")

    # ── Charts ────────────────────────────────────────────
    st.subheader(f"📈 {ticker} Price Chart & Risk Regime")
    st.plotly_chart(plot_price_and_regime(df_raw, dates, labels),
                    use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("🟩 **Low Risk** — Low volatility")
    c2.markdown("🟧 **Medium Risk** — Moderate volatility")
    c3.markdown("🟥 **High Risk** — High volatility")

    st.subheader("🔬 Model Class Probabilities")
    st.plotly_chart(plot_probabilities(dates, probas),
                    use_container_width=True)

    # ── Recent table ──────────────────────────────────────
    st.subheader("📋 Recent Predictions (Last 10 Days)")
    n = min(10, len(dates))
    st.dataframe(pd.DataFrame({
        "Date"      : [d.strftime("%Y-%m-%d") for d in dates[-n:]],
        "Regime"    : [CLASS_NAMES[l] for l in labels[-n:]],
        "P(Low)"    : [f"{p[0]:.2%}" for p in probas[-n:]],
        "P(Medium)" : [f"{p[1]:.2%}" for p in probas[-n:]],
        "P(High)"   : [f"{p[2]:.2%}" for p in probas[-n:]],
    }).set_index("Date"), use_container_width=True)

    # ── Sector reference ──────────────────────────────────
    st.divider()
    st.subheader("🗂️ All Sectors & Companies")
    for sec, stocks in SECTORS.items():
        with st.expander(f"{sec}  ({len(stocks)} companies)"):
            st.dataframe(
                pd.DataFrame([{"Ticker": t, "Company": n}
                               for t, n in stocks.items()]).set_index("Ticker"),
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────────
# TAB 2 — AUTO ALERT MONITOR
# ─────────────────────────────────────────────────────────────
def render_auto_alert_monitor(model, scaler):
    """
    Automatically scan all 50 stocks and display a live
    alert table ranked by High-Risk probability.
    """

    st.subheader("🚨 Automatic Alert Monitor — All 50 S&P 500 Stocks")
    st.markdown(
        "This tab scans **all 50 stocks** at once and automatically flags "
        "any stock currently in **High Risk** or **Medium Risk** regime. "
        "Results are sorted by High-Risk probability, highest first."
    )

    # ── Controls row ──────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        threshold = st.slider(
            "High-Risk Threshold",
            min_value=0.30, max_value=0.95,
            value=0.50, step=0.05,
            help="Stocks with P(High Risk) >= this value trigger a red alert.",
            key="monitor_threshold"
        )
    with col2:
        auto_refresh = st.selectbox(
            "Auto-Refresh Every",
            options=["Off", "5 minutes", "15 minutes", "30 minutes"],
            index=0,
            key="auto_refresh"
        )
    with col3:
        scan_button = st.button("🔍 Scan All Stocks Now",
                                use_container_width=True,
                                type="primary")

    # ── Auto-refresh logic ────────────────────────────────
    refresh_map = {
        "Off": None, "5 minutes": 300,
        "15 minutes": 900, "30 minutes": 1800
    }
    refresh_secs = refresh_map[auto_refresh]

    # Store last scan time in session state
    if "last_scan_time" not in st.session_state:
        st.session_state.last_scan_time  = None
        st.session_state.scan_results    = None

    # Determine whether to run a scan
    should_scan = scan_button
    if (refresh_secs is not None and
            st.session_state.last_scan_time is not None):
        elapsed = time.time() - st.session_state.last_scan_time
        if elapsed >= refresh_secs:
            should_scan = True

    # Show last scan time
    if st.session_state.last_scan_time:
        elapsed_min = (time.time() - st.session_state.last_scan_time) / 60
        st.caption(f"Last scan: {elapsed_min:.1f} minutes ago  "
                   f"| Next auto-refresh: "
                   f"{'Off' if not refresh_secs else f'{refresh_secs//60} min'}")

    # ── RUN SCAN ──────────────────────────────────────────
    if should_scan or st.session_state.scan_results is None:
        results = []
        progress_bar = st.progress(0, text="Scanning stocks...")
        total = len(ALL_STOCKS)

        for idx, (ticker, info) in enumerate(ALL_STOCKS.items()):
            progress_bar.progress(
                (idx + 1) / total,
                text=f"Scanning {ticker} ({idx+1}/{total})..."
            )
            pred = get_latest_prediction(ticker, model, scaler)
            if pred is None:
                continue

            results.append({
                "Ticker"   : ticker,
                "Company"  : info["company"],
                "Sector"   : info["sector"],
                "Regime"   : regime_badge(pred["label"]),
                "P(High)"  : pred["p_high"],
                "P(Medium)": pred["p_medium"],
                "P(Low)"   : pred["p_low"],
                "Close"    : pred["close"],
                "Date"     : pred["date"],
                "_label"   : pred["label"],
            })

        progress_bar.empty()
        st.session_state.scan_results   = results
        st.session_state.last_scan_time = time.time()

    results = st.session_state.scan_results

    if not results:
        st.warning("No results yet. Click 'Scan All Stocks Now' to begin.")
        return

    df_results = pd.DataFrame(results)
    df_results.sort_values("P(High)", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    # ── Summary Metrics ───────────────────────────────────
    n_high   = int((df_results["_label"] == 2).sum())
    n_medium = int((df_results["_label"] == 1).sum())
    n_low    = int((df_results["_label"] == 0).sum())
    n_alert  = int((df_results["P(High)"] >= threshold).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔴 High Risk Stocks",   n_high)
    m2.metric("🟡 Medium Risk Stocks", n_medium)
    m3.metric("🟢 Low Risk Stocks",    n_low)
    m4.metric("🚨 Alert Triggered",    n_alert,
              help=f"Stocks with P(High Risk) >= {threshold:.0%}")

    st.divider()

    # ── HIGH RISK ALERT TABLE ─────────────────────────────
    df_high = df_results[df_results["P(High)"] >= threshold].copy()

    if df_high.empty:
        st.success(
            f"✅ No stocks currently exceed the High-Risk threshold "
            f"of {threshold:.0%}. Market conditions appear stable."
        )
    else:
        st.error(f"🚨 **{len(df_high)} stock(s) triggered a HIGH-RISK ALERT**")
        display_high = df_high[[
            "Ticker", "Company", "Sector", "Regime",
            "P(High)", "P(Medium)", "P(Low)", "Close", "Date"
        ]].copy()
        display_high["P(High)"]   = display_high["P(High)"].map("{:.1%}".format)
        display_high["P(Medium)"] = display_high["P(Medium)"].map("{:.1%}".format)
        display_high["P(Low)"]    = display_high["P(Low)"].map("{:.1%}".format)
        display_high["Close"]     = display_high["Close"].map("${:.2f}".format)
        st.dataframe(display_high.set_index("Ticker"),
                     use_container_width=True)

    st.divider()

    # ── MEDIUM RISK TABLE ─────────────────────────────────
    df_medium = df_results[
        (df_results["_label"] == 1) &
        (df_results["P(High)"] < threshold)
    ].copy()

    st.subheader(f"🟡 Medium Risk Stocks ({len(df_medium)})")
    if df_medium.empty:
        st.info("No stocks currently in Medium Risk regime.")
    else:
        display_med = df_medium[[
            "Ticker", "Company", "Sector", "Regime",
            "P(High)", "P(Medium)", "P(Low)", "Close", "Date"
        ]].copy()
        display_med["P(High)"]   = display_med["P(High)"].map("{:.1%}".format)
        display_med["P(Medium)"] = display_med["P(Medium)"].map("{:.1%}".format)
        display_med["P(Low)"]    = display_med["P(Low)"].map("{:.1%}".format)
        display_med["Close"]     = display_med["Close"].map("${:.2f}".format)
        st.dataframe(display_med.set_index("Ticker"),
                     use_container_width=True)

    st.divider()

    # ── LOW RISK TABLE ────────────────────────────────────
    df_low = df_results[df_results["_label"] == 0].copy()
    st.subheader(f"🟢 Low Risk Stocks ({len(df_low)})")
    if df_low.empty:
        st.info("No stocks currently in Low Risk regime.")
    else:
        with st.expander("Show Low Risk stocks"):
            display_low = df_low[[
                "Ticker", "Company", "Sector",
                "P(High)", "P(Medium)", "P(Low)", "Close", "Date"
            ]].copy()
            display_low["P(High)"]   = display_low["P(High)"].map("{:.1%}".format)
            display_low["P(Medium)"] = display_low["P(Medium)"].map("{:.1%}".format)
            display_low["P(Low)"]    = display_low["P(Low)"].map("{:.1%}".format)
            display_low["Close"]     = display_low["Close"].map("${:.2f}".format)
            st.dataframe(display_low.set_index("Ticker"),
                         use_container_width=True)

    # ── Risk Distribution Chart ───────────────────────────
    st.divider()
    st.subheader("📊 Risk Distribution Across All Stocks")

    fig = go.Figure(go.Bar(
        x=["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
        y=[n_high, n_medium, n_low],
        marker_color=[CLASS_COLOURS[2], CLASS_COLOURS[1], CLASS_COLOURS[0]],
        text=[n_high, n_medium, n_low],
        textposition="auto"
    ))
    fig.update_layout(
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=20, b=40),
        yaxis_title="Number of Stocks",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Auto-refresh trigger ──────────────────────────────
    if refresh_secs:
        time.sleep(1)
        st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.title("📊 Hybrid CNN-LSTM: Risk Regime Detector")
    st.markdown(
        "Deep learning risk classification for **50 S&P 500 stocks** "
        "across **9 sectors**, trained on 10 years of historical data."
    )

    # ── Load model & scaler ───────────────────────────────
    try:
        model  = load_model()
        scaler = load_scaler()
    except Exception as e:
        st.error(
            f"⚠️ Could not load model or scaler.\n\n"
            f"Make sure you have run Stages 1–4 first.\n\n`{e}`"
        )
        st.stop()

    # ── Two tabs ──────────────────────────────────────────
    tab1, tab2 = st.tabs([
        "📈 Stock Analysis",
        "🚨 Auto Alert Monitor"
    ])

    with tab1:
        render_stock_analysis(model, scaler)

    with tab2:
        render_auto_alert_monitor(model, scaler)

    st.caption(
        "⚠️ For educational purposes only. Not financial advice. "
        "Yahoo Finance data may be delayed by 15–20 minutes."
    )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()