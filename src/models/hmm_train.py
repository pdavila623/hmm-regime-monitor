"""
HMM training module.

Key design decisions:
- hmmlearn GaussianHMM: mature, fast, supports multiple covariance types
- "diag" covariance: regularized, avoids overfitting with many features
- Multiple random restarts (N_RESTARTS): EM is sensitive to initialization;
  taking the best log-likelihood across restarts avoids poor local optima
- KMeans initialization: better starting point than random
- Covariance floor: prevents numerical degeneracy (singular covariance matrices)

Usage:
    python -m src.models.hmm_train --symbol BTC/USDT --tf 1h --states 5
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
from loguru import logger
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

from src.config import (
    COV_FLOOR,
    DEFAULT_COV_TYPE,
    DEFAULT_N_STATES,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    MODELS_DIR,
    N_ITER,
    N_RESTARTS,
    TRAIN_WINDOW,
)
from src.data.io import get_data, ensure_min_rows
from src.features.build_features import build_features, get_feature_matrix

app = typer.Typer()


def _apply_cov_floor(model: GaussianHMM, floor: float = COV_FLOOR) -> GaussianHMM:
    """Apply a minimum floor to covariance diagonals to prevent degeneracy."""
    if model.covariance_type == "diag":
        model.covars_ = np.maximum(model.covars_, floor)
    elif model.covariance_type == "full":
        for i in range(len(model.covars_)):
            np.fill_diagonal(model.covars_[i], np.maximum(np.diag(model.covars_[i]), floor))
    return model


def _kmeans_init(X: np.ndarray, n_states: int) -> np.ndarray:
    """Initialize HMM means via KMeans clustering."""
    km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    km.fit(X)
    return km.cluster_centers_


def train_hmm(
    X: np.ndarray,
    n_states: int = DEFAULT_N_STATES,
    cov_type: str = DEFAULT_COV_TYPE,
    n_iter: int = N_ITER,
    n_restarts: int = N_RESTARTS,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[GaussianHMM, float]:
    """
    Train GaussianHMM with multiple restarts and KMeans initialization.

    Args:
        X: Feature matrix (n_samples, n_features)
        n_states: Number of hidden states
        cov_type: 'diag' (recommended) or 'full'
        n_iter: Max EM iterations per restart
        n_restarts: Number of random restarts
        random_state: Base random seed
        verbose: Print restart scores

    Returns:
        (best_model, best_log_likelihood)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if np.isnan(X).any():
        raise ValueError("Feature matrix contains NaN values")

    best_model = None
    best_score = -np.inf
    means_init = _kmeans_init(X, n_states)

    for restart in range(n_restarts):
        seed = (random_state or 0) + restart
        rng = np.random.RandomState(seed)

        # Add small noise to KMeans init for diversity
        noise_scale = X.std(axis=0).mean() * 0.1
        init_means = means_init + rng.randn(*means_init.shape) * noise_scale

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=seed,
            tol=1e-4,
            init_params="stc",  # init startprob, transmat, covars; use our means
            params="stmc",       # learn all params
        )
        model.means_init = init_means

        try:
            model.fit(X)
            _apply_cov_floor(model)
            score = model.score(X)

            if verbose:
                logger.debug(f"  Restart {restart+1}/{n_restarts}: score={score:.2f}, converged={model.monitor_.converged}")

            if score > best_score:
                best_score = score
                best_model = model

        except Exception as e:
            logger.warning(f"  Restart {restart+1} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed. Check your data.")

    logger.info(
        f"Best HMM: states={n_states}, cov={cov_type}, "
        f"score={best_score:.2f}, converged={best_model.monitor_.converged}"
    )
    return best_model, best_score


def compute_bic_aic(
    model: GaussianHMM, X: np.ndarray
) -> Tuple[float, float]:
    """
    Compute BIC and AIC for model selection.

    BIC penalizes model complexity more → prefers simpler models → less overfit.
    AIC is less conservative.
    We use BIC as primary criterion.

    n_params for GaussianHMM (diag):
        - means: n_states * n_features
        - covars: n_states * n_features
        - transmat: n_states * (n_states - 1)  [rows sum to 1]
        - startprob: n_states - 1
    """
    n = len(X)
    k = model.n_components
    d = X.shape[1]

    if model.covariance_type == "diag":
        n_params = k * d + k * d + k * (k - 1) + (k - 1)
    else:  # full
        n_params = k * d + k * d * (d + 1) // 2 + k * (k - 1) + (k - 1)

    log_likelihood = model.score(X) * n
    bic = -2 * log_likelihood + n_params * np.log(n)
    aic = -2 * log_likelihood + 2 * n_params
    return bic, aic


def save_model(model: GaussianHMM, symbol: str, timeframe: str, n_states: int) -> Path:
    """Save trained model to disk."""
    safe_sym = symbol.replace("/", "_")
    path = MODELS_DIR / f"hmm_{safe_sym}_{timeframe}_s{n_states}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {path}")
    return path


def load_model(symbol: str, timeframe: str, n_states: int) -> Optional[GaussianHMM]:
    """Load trained model from disk."""
    safe_sym = symbol.replace("/", "_")
    path = MODELS_DIR / f"hmm_{safe_sym}_{timeframe}_s{n_states}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded: {path}")
    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────
@app.command()
def main(
    symbol: str = typer.Option(DEFAULT_SYMBOL),
    tf: str = typer.Option(DEFAULT_TIMEFRAME),
    states: int = typer.Option(DEFAULT_N_STATES, min=3, max=7),
    auto_states: bool = typer.Option(False, help="Auto-select n_states via BIC"),
    cov_type: str = typer.Option(DEFAULT_COV_TYPE),
    since: str = typer.Option("2022-01-01"),
    train_window: int = typer.Option(TRAIN_WINDOW),
):
    """Train HMM on historical data."""
    from src.models.model_selection import auto_select_states

    df = get_data(symbol=symbol, timeframe=tf, since=since)
    df = ensure_min_rows(df)
    feat = build_features(df.tail(train_window))
    X = get_feature_matrix(feat)

    if auto_states:
        model, n_states, ranking = auto_select_states(X, cov_type=cov_type)
        typer.echo("\nState selection ranking (by BIC):")
        for row in ranking:
            typer.echo(f"  {row['n_states']} states: BIC={row['bic']:.1f}, AIC={row['aic']:.1f}")
    else:
        n_states = states
        model, score = train_hmm(X, n_states=n_states, cov_type=cov_type)

    save_model(model, symbol, tf, n_states)
    typer.echo(f"\n✓ Model trained: {n_states} states | Saved to models_saved/")


if __name__ == "__main__":
    app()
