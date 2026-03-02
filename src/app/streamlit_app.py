"""
Market Regime Monitor — Streamlit Dashboard
============================================
Real-time / near-real-time HMM regime visualization.

Run:
    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Market Regime Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import (
    DEFAULT_EXCHANGE,
    DEFAULT_N_STATES,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    MAX_STATES,
    MIN_STATES,
    P_MIN,
    REGIME_COLORS,
    ROLLING_RETRAIN,
    RETRAIN_EVERY,
    TRAIN_WINDOW,
    VALID_TIMEFRAMES,
)
from src.data.io import get_data
from src.features.build_features import build_features, get_feature_matrix, FEATURE_COLS
from src.models.hmm_predict import (
    compute_state_stats,
    decode_states,
    get_current_signal,
)
from src.models.hmm_train import train_hmm, save_model, load_model
from src.models.model_selection import auto_select_states


# ─── Caching helpers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def cached_fetch(symbol, timeframe, since, exchange_id):
    return get_data(symbol=symbol, timeframe=timeframe, since=since, exchange_id=exchange_id)


@st.cache_data(ttl=600, show_spinner="Building features...")
def cached_features(df_json, train_window):
    df = pd.read_json(df_json)
    df.index = pd.to_datetime(df.index, utc=True)
    feat = build_features(df.tail(train_window))
    return feat


def _get_or_train_model(X, n_states, auto_states, cov_type, symbol, timeframe):
    """Load from session or train fresh."""
    cache_key = f"hmm_{symbol}_{timeframe}_{n_states}_{auto_states}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    with st.spinner("Training HMM model..."):
        if auto_states:
            model, actual_n, ranking = auto_select_states(X, cov_type=cov_type)
            st.session_state["state_ranking"] = ranking
            st.session_state["actual_n_states"] = actual_n
        else:
            model, _ = train_hmm(X, n_states=n_states, cov_type=cov_type)
            st.session_state["actual_n_states"] = n_states

        st.session_state[cache_key] = model

    return model


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _state_color(state_label: str) -> str:
    for key, color in REGIME_COLORS.items():
        if key.lower() in state_label.lower():
            return color
    return REGIME_COLORS["Unknown"]


def plot_price_with_regimes(df: pd.DataFrame, feat: pd.DataFrame, state_seq: np.ndarray,
                             state_stats: pd.DataFrame) -> go.Figure:
    """Price chart with colored background per regime state."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Price + Regimes", "Volume"),
    )

    # Align state_seq with feat index
    state_index = feat.index
    df_aligned = df.reindex(state_index)

    # Add regime background shading
    n = len(state_index)
    if n > 0:
        prev_state = state_seq[0]
        seg_start = state_index[0]

        for i in range(1, n):
            if state_seq[i] != prev_state or i == n - 1:
                end_idx = state_index[i] if state_seq[i] != prev_state else state_index[-1]
                label = state_stats.loc[prev_state, "label"] if prev_state in state_stats.index else "Unknown"
                color = _state_color(label)

                fig.add_vrect(
                    x0=seg_start,
                    x1=end_idx,
                    fillcolor=color,
                    opacity=0.18,
                    line_width=0,
                    row=1, col=1,
                )
                seg_start = state_index[i]
                prev_state = state_seq[i]

    # Candlestick (or line if no OHLC)
    if "open" in df_aligned.columns:
        fig.add_trace(
            go.Candlestick(
                x=df_aligned.index,
                open=df_aligned["open"],
                high=df_aligned["high"],
                low=df_aligned["low"],
                close=df_aligned["close"],
                name="Price",
                increasing_line_color="#00c896",
                decreasing_line_color="#ff4b4b",
                showlegend=False,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=df_aligned.index, y=df_aligned["close"], name="Close", line=dict(color="#5599ff")),
            row=1, col=1,
        )

    # Volume
    vol_colors = ["#00c896" if r >= 0 else "#ff4b4b"
                  for r in feat["log_return"].values]
    fig.add_trace(
        go.Bar(x=df_aligned.index, y=df_aligned["volume"], name="Volume",
               marker_color=vol_colors, opacity=0.6, showlegend=False),
        row=2, col=1,
    )

    # Legend entries for regimes
    for s, row in state_stats.iterrows():
        label = row["label"]
        color = _state_color(label)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=f"State {s}: {label}",
                showlegend=True,
            ),
            row=1, col=1,
        )

    fig.update_layout(
        height=550,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_posteriors(feat: pd.DataFrame, posteriors: np.ndarray,
                    state_stats: pd.DataFrame) -> go.Figure:
    """Stacked area chart of posterior probabilities over time."""
    fig = go.Figure()
    n_states = posteriors.shape[1]

    for s in range(n_states):
        label = state_stats.loc[s, "label"] if s in state_stats.index else f"State {s}"
        color = _state_color(label)
        fig.add_trace(go.Scatter(
            x=feat.index,
            y=posteriors[:, s],
            name=f"S{s}: {label}",
            mode="lines",
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color,
        ))

    fig.update_layout(
        title="Posterior Probabilities by State",
        height=280,
        template="plotly_dark",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_transition_matrix(model) -> go.Figure:
    """Heatmap of HMM transition matrix."""
    trans = model.transmat_
    n = trans.shape[0]
    labels = [f"S{i}" for i in range(n)]
    fig = px.imshow(
        trans,
        text_auto=".2f",
        color_continuous_scale="Blues",
        labels=dict(x="To State", y="From State", color="Prob"),
        x=labels,
        y=labels,
        title="Transition Matrix",
    )
    fig.update_layout(
        height=350,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_showscale=False,
    )
    return fig


def plot_state_returns(state_stats: pd.DataFrame) -> go.Figure:
    """Bar chart: mean return per state."""
    colors = [_state_color(row["label"]) for _, row in state_stats.iterrows()]
    fig = go.Figure(go.Bar(
        x=[f"S{s}: {row['label']}" for s, row in state_stats.iterrows()],
        y=state_stats["mean_return"].values * 100,
        marker_color=colors,
        text=[f"{v*100:.3f}%" for v in state_stats["mean_return"].values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Mean Return per State (%)",
        height=280,
        template="plotly_dark",
        yaxis_title="Mean Log Return (%)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ─── Current state panel ──────────────────────────────────────────────────────

def render_current_state(signal: dict):
    """Render the 'Current Regime' panel."""
    label = signal["label"]
    conf = signal["confidence"]
    color = _state_color(label)

    st.markdown(
        f"""
        <div style="
            border: 2px solid {color};
            border-radius: 12px;
            padding: 16px 20px;
            background: {color}22;
            margin-bottom: 8px;
        ">
        <h2 style="color:{color}; margin:0 0 4px 0;">{label}</h2>
        <span style="font-size:0.9rem; color:#aaa;">Regime Signal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{conf:.1%}")
    col2.metric("Expected Return", f"{signal['expected_return']*100:.3f}%")
    col3.metric("Volatility", f"{signal['expected_vol']*100:.3f}%")
    col4.metric("P(Up)", f"{signal['proba_up']:.1%}")

    if signal["signal"] == "Neutral":
        st.warning(f"⚠️ Confidence below threshold ({conf:.1%} < p_min). Signal is NEUTRAL.")
    elif "Bull" in label:
        st.success(f"🟢 Bullish regime detected with {conf:.1%} confidence.")
    elif "Bear" in label:
        st.error(f"🔴 Bearish regime detected with {conf:.1%} confidence.")
    else:
        st.info(f"🟡 Sideways / consolidation regime.")


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("📈 Market Regime Monitor — HMM Dashboard")
    st.markdown(
        "Hidden Markov Model regime detection. "
        "**Not financial advice** — for research and monitoring purposes only."
    )

    # ─── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
        exchange_id = st.selectbox("Exchange", ["binance", "bybit", "okx", "kraken"], index=0)
        timeframe = st.selectbox("Timeframe", VALID_TIMEFRAMES, index=VALID_TIMEFRAMES.index(DEFAULT_TIMEFRAME))

        st.divider()

        auto_states = st.toggle("Auto-select # states (BIC)", value=False)
        n_states = st.slider(
            "# States", min_value=MIN_STATES, max_value=MAX_STATES,
            value=DEFAULT_N_STATES, disabled=auto_states
        )
        cov_type = st.selectbox("Covariance type", ["diag", "full"], index=0)

        st.divider()

        train_window = st.number_input("Train window (bars)", min_value=500, max_value=10000,
                                        value=TRAIN_WINDOW, step=100)
        p_min = st.slider("Min confidence (p_min)", min_value=0.3, max_value=0.9,
                           value=P_MIN, step=0.05)
        since = st.date_input("Fetch data since", value=datetime.today() - timedelta(days=365))

        st.divider()

        use_rolling = st.toggle("Rolling retrain", value=False)
        if use_rolling:
            retrain_every = st.number_input("Retrain every N bars", min_value=50,
                                             max_value=500, value=RETRAIN_EVERY)
        else:
            retrain_every = RETRAIN_EVERY

        run_btn = st.button("🚀 Run / Refresh", type="primary", use_container_width=True)
        clear_cache_btn = st.button("🗑️ Clear cache & retrain", use_container_width=True)

    if clear_cache_btn:
        keys_to_delete = [k for k in st.session_state if k.startswith("hmm_")]
        for k in keys_to_delete:
            del st.session_state[k]
        st.cache_data.clear()
        st.success("Cache cleared.")
        st.rerun()

    if not run_btn and "last_result" not in st.session_state:
        st.info("👈 Configure settings and click **Run / Refresh** to start.")
        return

    # ─── Data & model pipeline ───────────────────────────────────────────────
    try:
        with st.status("Loading data...", expanded=False):
            df = cached_fetch(symbol, timeframe, str(since), exchange_id)
            st.write(f"✓ {len(df)} bars loaded ({df.index[0].date()} → {df.index[-1].date()})")

        with st.status("Building features...", expanded=False):
            feat = build_features(df.tail(train_window), scale=True)
            X = get_feature_matrix(feat)
            st.write(f"✓ {len(feat)} feature rows, {X.shape[1]} features")

        model = _get_or_train_model(X, n_states, auto_states, cov_type, symbol, timeframe)
        actual_n = st.session_state.get("actual_n_states", n_states)

        with st.status("Decoding states...", expanded=False):
            state_seq, posteriors = decode_states(model, X)
            state_stats = compute_state_stats(model, X, state_seq, X[:, 0])
            signal = get_current_signal(model, X, state_stats, p_min=p_min)
            st.write(f"✓ {actual_n} states decoded")

        st.session_state["last_result"] = {
            "df": df, "feat": feat, "X": X, "model": model,
            "state_seq": state_seq, "posteriors": posteriors,
            "state_stats": state_stats, "signal": signal, "actual_n": actual_n,
        }

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
        return

    res = st.session_state["last_result"]
    df = res["df"]
    feat = res["feat"]
    state_seq = res["state_seq"]
    posteriors = res["posteriors"]
    state_stats = res["state_stats"]
    signal = res["signal"]
    model = res["model"]
    actual_n = res["actual_n"]

    # ─── Current regime panel ────────────────────────────────────────────────
    st.subheader("🎯 Current Regime")
    render_current_state(signal)

    st.divider()

    # ─── Main charts ─────────────────────────────────────────────────────────
    st.subheader("📊 Price + Regime History")
    fig_price = plot_price_with_regimes(df, feat, state_seq, state_stats)
    st.plotly_chart(fig_price, use_container_width=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📉 Posterior Probabilities")
        fig_post = plot_posteriors(feat, posteriors, state_stats)
        st.plotly_chart(fig_post, use_container_width=True)

    with col_right:
        st.subheader("🔄 Transition Matrix")
        fig_trans = plot_transition_matrix(model)
        st.plotly_chart(fig_trans, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Mean Return by State")
        fig_ret = plot_state_returns(state_stats)
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_b:
        st.subheader("📋 State Statistics Table")
        display_stats = state_stats.copy()
        display_stats["mean_return"] = (display_stats["mean_return"] * 100).round(4).astype(str) + "%"
        display_stats["vol"] = (display_stats["vol"] * 100).round(4).astype(str) + "%"
        display_stats["pct_up"] = (display_stats["pct_up"] * 100).round(1).astype(str) + "%"
        display_stats["pct"] = display_stats["pct"].astype(str) + "%"
        st.dataframe(display_stats[["label", "count", "pct", "mean_return", "vol", "sharpe", "pct_up", "avg_duration"]],
                     use_container_width=True)

    # ─── Auto-state ranking ──────────────────────────────────────────────────
    if auto_states and "state_ranking" in st.session_state:
        with st.expander("🏆 Auto-State Selection Ranking (BIC)"):
            ranking = st.session_state["state_ranking"]
            rank_df = pd.DataFrame([
                {"n_states": r["n_states"], "BIC": round(r["bic"], 1),
                 "AIC": round(r["aic"], 1), "Log-Likelihood": round(r["log_likelihood"], 1)}
                for r in ranking
            ])
            st.dataframe(rank_df, use_container_width=True)

    # ─── Feature importance hint ──────────────────────────────────────────────
    with st.expander("🔍 Feature Means per State (HMM Means)"):
        means_df = pd.DataFrame(
            model.means_,
            columns=FEATURE_COLS,
            index=[f"State {i}" for i in range(model.n_components)],
        )
        st.dataframe(means_df.round(4), use_container_width=True)

    # ─── Alert / regime change detection ─────────────────────────────────────
    if len(state_seq) >= 2:
        if state_seq[-1] != state_seq[-2]:
            new_label = state_stats.loc[state_seq[-1], "label"] if state_seq[-1] in state_stats.index else "Unknown"
            st.toast(f"🔔 Regime changed → **{new_label}**", icon="🔔")

    # ─── Footer ──────────────────────────────────────────────────────────────
    st.divider()
    last_bar = feat.index[-1]
    st.caption(
        f"Last bar: {last_bar} UTC | "
        f"Model: {actual_n}-state GaussianHMM ({cov_type} cov) | "
        f"Train window: {len(feat)} bars | "
        f"p_min: {p_min} | "
        f"⚠️ Not financial advice"
    )

    st.caption("Auto-refresh: use browser reload or click 'Run / Refresh' manually.")


if __name__ == "__main__":
    main()
