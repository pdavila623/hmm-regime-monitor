"""
Central configuration for Market Regime HMM.
All defaults are documented with reasoning.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"
MODELS_DIR = ROOT / "models_saved"
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ─── Data defaults ────────────────────────────────────────────────────────────
DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
VALID_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# ─── Feature engineering ──────────────────────────────────────────────────────
ROLLING_VOL_WINDOW = 20      # ~1 trading day on 1h bars
TREND_WINDOW = 20            # EMA/regression slope window
RSI_PERIOD = 14              # Standard RSI
ATR_PERIOD = 14              # ATR smoothing
ZSCORE_WINDOW = 60           # Window for return z-score normalization
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ─── HMM defaults ─────────────────────────────────────────────────────────────
# WHY 2000 bars?
#   - Enough to see multiple regimes (bull/bear/sideways cycles)
#   - Not so large that old regimes distort current dynamics
#   - At 1h: ~83 days; at 4h: ~333 days; at 1d: ~8 years
TRAIN_WINDOW = 2000

# WHY "diag" covariance?
#   - "full" can overfit with many features & limited data (n << p²)
#   - "diag" assumes feature independence per state → regularized, stable
#   - Tested empirically: BIC usually favors "diag" for n_states=3-7
DEFAULT_COV_TYPE = "diag"

DEFAULT_N_STATES = 5         # 5 gives Bull/Bear/Volatile Bull/Volatile Bear/Sideways
N_ITER = 200                 # Max EM iterations
N_RESTARTS = 15              # WHY 15? Avoid local optima in EM; best ll chosen
MIN_STATES = 3
MAX_STATES = 7

# WHY p_min=0.6?
#   - Posterior < 0.6 means HMM is uncertain between states → "Neutral"
#   - 0.6 is a reasonable confidence threshold (not too strict, not too loose)
P_MIN = 0.6

# ─── Rolling retrain ──────────────────────────────────────────────────────────
# WHY rolling retrain?
#   - Market regimes are non-stationary; a model trained on old data may drift
#   - Rolling window keeps the model adapted to recent structure
ROLLING_RETRAIN = True
RETRAIN_EVERY = 100          # Retrain every 100 new bars
RETRAIN_WINDOW = 2000        # Fixed window (not expanding) to avoid stale data

# ─── Walk-forward evaluation ──────────────────────────────────────────────────
WF_TRAIN_SIZE = 1500
WF_TEST_SIZE = 200
WF_STEP = 200

# ─── Regularization ──────────────────────────────────────────────────────────
COV_FLOOR = 1e-4             # Floor on covariance diagonals to prevent degeneracy

# ─── Regime labels (mapped post-hoc by return/vol profile) ───────────────────
REGIME_COLORS = {
    "Bull": "#00c896",
    "Bear": "#ff4b4b",
    "Sideways": "#ffd700",
    "Volatile Bull": "#00a0ff",
    "Volatile Bear": "#ff8c00",
    "Unknown": "#888888",
    "Neutral": "#cccccc",
}
