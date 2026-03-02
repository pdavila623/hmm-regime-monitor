# Market Regime Monitor — HMM

Detect and visualize crypto market regimes (Bull / Bear / Sideways / Volatile) using a **Hidden Markov Model (HMM)** with a real-time Streamlit dashboard.

> ⚠️ **Not financial advice.** This is a research/monitoring tool only.

---

## Features

- **Multi-timeframe**: 5m, 15m, 1h, 4h, 1d
- **Multi-state**: 3–7 regimes (manual or auto-selected via BIC)
- **Rich features**: log return, rolling vol, trend slope, RSI, ATR, z-score, MACD histogram
- **Robust training**: multiple EM restarts, KMeans init, covariance floor
- **Walk-forward backtest**: direction accuracy, balanced accuracy, log-loss vs EMA baseline
- **Dashboard**: price + regime overlay, posterior probs, transition matrix, state stats

---

## Installation

```bash
# Clone / download the project
cd market_regime_hmm

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Fetch data

```bash
# Fetch BTC/USDT 1h from Binance since 2022
python -m src.data.ccxt_fetch --symbol BTC/USDT --tf 1h --since 2022-01-01

# Different timeframe
python -m src.data.ccxt_fetch --symbol ETH/USDT --tf 4h --since 2021-01-01

# Load from local CSV (no internet needed)
# CSV must have columns: timestamp (index), open, high, low, close, volume
```

### 2. Train HMM

```bash
# Train with 5 states (default)
python -m src.models.hmm_train --symbol BTC/USDT --tf 1h --states 5

# Auto-select best # of states (3–7) via BIC
python -m src.models.hmm_train --symbol BTC/USDT --tf 1h --auto-states
```

### 3. Evaluate (walk-forward)

```bash
# Walk-forward with 5 states
python -m src.eval.walkforward --symbol BTC/USDT --tf 1h --states 5

# Auto-select states per fold
python -m src.eval.walkforward --symbol BTC/USDT --tf 1h --auto
```

### 4. Launch Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## Dashboard Guide

| Panel | Description |
|---|---|
| **Current Regime** | Signal (Bull/Bear/Sideways), confidence, expected return, vol |
| **Price + Regimes** | Candlestick with colored background per state |
| **Posterior Probs** | Stacked area chart — how certain the model is per state over time |
| **Transition Matrix** | Heatmap of state-to-state transition probabilities |
| **State Statistics** | Mean return, vol, Sharpe, % up, avg duration per state |

**Controls (sidebar):**
- Symbol, exchange, timeframe
- \# states (3–7) or auto-BIC
- Train window (bars), p_min (confidence threshold)
- Rolling retrain toggle

---

## Model Design & Decisions

### Why HMM?
Markets cycle through regimes (trending, volatile, ranging). HMMs model this as a latent Markov chain — the hidden state drives observable returns/volatility.

### Why these features?
| Feature | Why |
|---|---|
| `log_return` | Core stationary signal; prices are non-stationary |
| `rolling_vol` | Volatility clustering is the clearest regime signal |
| `trend_strength` | OLS slope on log-close — captures trend direction |
| `rsi_norm` | Momentum: overbought/oversold conditions |
| `atr_norm` | Price-normalized range volatility |
| `zscore_return` | Normalizes return distribution to detect outlier bars |
| `macd_hist_norm` | Short-term momentum crossover signal |

All features are **return/rate-based** (not price levels) to satisfy HMM stationarity assumption.

### Why `diag` covariance?
- `full` covariance has O(n_states × n_features²) parameters → overfits with limited data
- `diag` assumes feature independence per state → regularized, numerically stable
- Empirically: BIC usually prefers `diag` for 3–7 states on financial data

### Why multiple restarts?
EM (Expectation-Maximization) converges to local optima. With 15 random restarts from slightly perturbed KMeans initializations, we take the best log-likelihood solution, reducing the chance of degenerate solutions.

### Why BIC for model selection?
BIC = -2·LL + k·log(n) penalizes model complexity more than AIC. This prevents selecting too many states (overfitting) when the data doesn't support them.

### Why walk-forward evaluation?
Standard cross-validation leaks future data into training. Walk-forward evaluation strictly respects temporal ordering: the model never sees test data during training.

---

## Project Structure

```
market_regime_hmm/
├── requirements.txt
├── README.md
├── cache/                    # Parquet cache of OHLCV data
├── models_saved/             # Trained HMM pickles
├── src/
│   ├── config.py             # All defaults, documented
│   ├── data/
│   │   ├── ccxt_fetch.py     # CCXT data ingestion + cache
│   │   └── io.py             # Unified data loader
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── hmm_train.py      # Training, BIC/AIC, save/load
│   │   ├── hmm_predict.py    # Viterbi decode, regime labeling, signals
│   │   └── model_selection.py # Auto n_states via BIC
│   ├── eval/
│   │   ├── metrics.py        # Accuracy, log-loss, stability metrics
│   │   └── walkforward.py    # Walk-forward evaluation
│   └── app/
│       └── streamlit_app.py  # Dashboard
└── tests/
    └── test_features_and_model.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Limitations

1. **No lookahead by design** — signals use only past data, but HMM state labels are assigned post-hoc on training data.
2. **Non-stationary markets** — model may drift; use rolling retrain option.
3. **Regime labels are heuristic** — Bull/Bear/Sideways labels are based on state statistics, not guaranteed.
4. **Crypto-specific** — features and defaults tuned for crypto; may need adjustment for equities.
5. **No order execution** — this is a monitoring tool, not a trading system.

---

## Configuration

All defaults are in `src/config.py` with inline reasoning comments. Key params:

| Parameter | Default | Why |
|---|---|---|
| `TRAIN_WINDOW` | 2000 bars | ~83 days on 1h; enough for multi-cycle view |
| `N_RESTARTS` | 15 | Avoid EM local optima |
| `DEFAULT_COV_TYPE` | "diag" | Regularized, stable |
| `P_MIN` | 0.6 | Confidence threshold before signaling |
| `RETRAIN_EVERY` | 100 bars | Adapt to regime drift without thrashing |
